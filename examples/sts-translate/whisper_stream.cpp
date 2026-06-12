// whisper_stream — streaming STT wrapper implementation. See whisper_stream.h.

#include "whisper_stream.h"
#include "whisper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

namespace {

int64_t now_us() {
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration_cast<std::chrono::microseconds>(
               clk::now().time_since_epoch()).count();
}

std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\n\r");
    return s.substr(a, b - a + 1);
}

// Stateful linear resampler: arbitrary src rate -> 16 kHz, mono. Holds a small
// carry of unconsumed input across calls so partial frames don't glitch. Linear
// interpolation is plenty for 16 kHz ASR (whisper's own front-end band-limits).
struct Resampler {
    double fs_in  = 0.0;
    double fs_out = WHISPER_SAMPLE_RATE;
    double pos    = 0.0;          // fractional read position within `in`
    std::vector<float> in;        // pending mono input samples

    void reset(double src_rate) { fs_in = src_rate; pos = 0.0; in.clear(); }

    void process(const float * mono, int n, std::vector<float> & out) {
        in.insert(in.end(), mono, mono + n);
        if (fs_in <= 0.0 || fs_in == fs_out) {        // identity
            out.insert(out.end(), in.begin(), in.end());
            in.clear();
            return;
        }
        const double step = fs_in / fs_out;
        while (pos + 1.0 < (double) in.size()) {
            const int    i = (int) pos;
            const float  f = (float) (pos - i);
            out.push_back(in[i] * (1.0f - f) + in[i + 1] * f);
            pos += step;
        }
        const int consumed = (int) pos;               // keep the tail for next call
        if (consumed > 0) {
            in.erase(in.begin(), in.begin() + consumed);
            pos -= consumed;
        }
    }
};

} // namespace

struct whisper_stream {
    whisper_stream_params p{};
    whisper_context * ctx = nullptr;
    Resampler rs;

    std::vector<float> audio;     // rolling 16 kHz mono buffer
    int64_t base = 0;             // absolute index of audio[0] (trimmed prefix count)

    // geometry in 16 kHz samples
    int window_n = 0, hop_n = 0, endpoint_n = 0, max_utt_n = 0, vad_frame_n = 0;

    // VAD / endpointing state
    bool    in_utt = false;
    int64_t utt_start = 0;        // absolute index where current utterance began
    int64_t silence_run = 0;      // consecutive silent 16 kHz samples
    int64_t last_partial_at = 0;  // absolute index of last partial decode
    int     vad_acc_n = 0;        // samples accumulated toward the next VAD frame
    double  vad_acc_sq = 0.0;     // sum of squares for the current VAD frame

    // results
    std::string partial;
    bool partial_dirty = false;
    std::deque<std::string> finals;
    double last_decode_ms = 0.0;

    int64_t abs_end() const { return base + (int64_t) audio.size(); }

    // decode [a,b) of the absolute stream; returns trimmed text
    std::string decode(int64_t a, int64_t b, bool streaming) {
        a = std::max(a, base);
        b = std::min(b, abs_end());
        if (b - a <= 0) return "";
        const float * src = audio.data() + (a - base);
        const int     n   = (int) (b - a);

        whisper_full_params w = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        w.print_progress = w.print_special = w.print_realtime = w.print_timestamps = false;
        w.no_timestamps  = true;
        w.translate      = p.translate != 0;
        w.language       = p.language;
        w.detect_language = false;
        w.n_threads      = p.n_threads;
        // Partials use the trimmed audio_ctx for low latency; FINALs use the FULL
        // encoder context — a finalized utterance can be many seconds (well past
        // what a trimmed ctx covers), and a too-small ctx there triggers decoder
        // repetition runaway. Quality matters more than latency on the (rare) final.
        w.audio_ctx      = streaming ? p.audio_ctx : 0;
        w.temperature_inc = 0.0f;
        if (streaming) { w.single_segment = true; w.no_context = true; w.max_tokens = 64; }

        const int64_t t0 = now_us();
        if (whisper_full(ctx, w, src, n) != 0) return "";
        last_decode_ms = (now_us() - t0) / 1000.0;

        std::string text;
        for (int s = 0; s < whisper_full_n_segments(ctx); ++s)
            text += whisper_full_get_segment_text(ctx, s);
        return trim(text);
    }

    void finalize(int64_t utt_end) {
        if (!in_utt) return;
        std::string fin = decode(utt_start, utt_end, /*streaming=*/false);
        if (!fin.empty()) finals.push_back(fin);
        in_utt = false;
        partial.clear();
        partial_dirty = true;     // signal partial cleared
    }

    // Feed newly-resampled 16 kHz mono samples through VAD + decode scheduling.
    void ingest(const std::vector<float> & chunk) {
        for (float v : chunk) {
            audio.push_back(v);
            vad_acc_sq += (double) v * v;
            if (++vad_acc_n >= vad_frame_n) {
                const double rms = std::sqrt(vad_acc_sq / vad_acc_n);
                const bool   voiced = rms >= p.vad_threshold;
                const int64_t frame_end = abs_end();
                if (voiced) {
                    if (!in_utt) { in_utt = true; utt_start = frame_end - vad_acc_n; last_partial_at = utt_start; }
                    silence_run = 0;
                } else if (in_utt) {
                    silence_run += vad_acc_n;
                }
                vad_acc_n = 0; vad_acc_sq = 0.0;
            }
        }

        const int64_t end = abs_end();

        // endpoint: enough trailing silence after speech, or utterance too long
        if (in_utt && (silence_run >= endpoint_n ||
                       (end - utt_start) >= max_utt_n)) {
            finalize(end - silence_run);
        }

        // partial: emit a windowed hypothesis on the hop cadence while speaking
        if (in_utt && (end - last_partial_at) >= hop_n) {
            std::string pt = decode(end - window_n, end, /*streaming=*/true);
            last_partial_at = end;
            if (pt != partial) { partial = pt; partial_dirty = true; }
        }

        trim_buffer();
    }

    // keep memory bounded: drop old audio we no longer need (never inside an utterance)
    void trim_buffer() {
        const int64_t keep = (int64_t) window_n + max_utt_n;
        if (!in_utt && (int64_t) audio.size() > keep) {
            const int64_t drop = (int64_t) audio.size() - keep;
            audio.erase(audio.begin(), audio.begin() + drop);
            base += drop;
        }
    }
};

// ── C ABI ──────────────────────────────────────────────────────────────────────

extern "C" {

whisper_stream_params whisper_stream_default_params(void) {
    whisper_stream_params p{};
    p.model = nullptr; p.language = "en";
    p.translate = 0; p.use_gpu = 1; p.flash_attn = 0;
    p.n_threads = 4; p.audio_ctx = 0;
    p.window_ms = 1500; p.hop_ms = 200; p.endpoint_ms = 600; p.max_utterance_ms = 15000;
    p.vad_threshold = 0.012f; p.vad_frame_ms = 20;
    return p;
}

whisper_stream * whisper_stream_init(const whisper_stream_params * params) {
    if (!params || !params->model) return nullptr;
    auto * s = new whisper_stream();
    s->p = *params;

    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = params->use_gpu != 0;
    cp.flash_attn = params->flash_attn != 0;
    s->ctx = whisper_init_from_file_with_params(params->model, cp);
    if (!s->ctx) { delete s; return nullptr; }

    const int sr = WHISPER_SAMPLE_RATE;
    s->window_n     = std::max(1, params->window_ms      * sr / 1000);
    s->hop_n        = std::max(1, params->hop_ms         * sr / 1000);
    s->endpoint_n   = std::max(1, params->endpoint_ms    * sr / 1000);
    s->max_utt_n    = std::max(1, params->max_utterance_ms * sr / 1000);
    s->vad_frame_n  = std::max(1, params->vad_frame_ms   * sr / 1000);
    s->rs.reset(0.0);
    return s;
}

void whisper_stream_free(whisper_stream * s) {
    if (!s) return;
    if (s->ctx) whisper_free(s->ctx);
    delete s;
}

static int push_mono(whisper_stream * s, const std::vector<float> & mono, int src_rate) {
    if (!s) return 1;
    if (s->rs.fs_in != (double) src_rate) s->rs.reset(src_rate);
    std::vector<float> out;
    s->rs.process(mono.data(), (int) mono.size(), out);
    s->ingest(out);
    return 0;
}

int whisper_stream_push_f32(whisper_stream * s, const float * pcm,
                            int n_frames, int n_channels, int src_rate) {
    if (!s || !pcm || n_frames <= 0 || n_channels <= 0) return 1;
    std::vector<float> mono((size_t) n_frames);
    if (n_channels == 1) {
        std::copy(pcm, pcm + n_frames, mono.begin());
    } else {
        for (int i = 0; i < n_frames; ++i) {
            float acc = 0.0f;
            for (int c = 0; c < n_channels; ++c) acc += pcm[(size_t) i * n_channels + c];
            mono[i] = acc / n_channels;
        }
    }
    return push_mono(s, mono, src_rate);
}

int whisper_stream_push_i16(whisper_stream * s, const int16_t * pcm,
                            int n_frames, int n_channels, int src_rate) {
    if (!s || !pcm || n_frames <= 0 || n_channels <= 0) return 1;
    std::vector<float> mono((size_t) n_frames);
    for (int i = 0; i < n_frames; ++i) {
        int acc = 0;
        for (int c = 0; c < n_channels; ++c) acc += pcm[(size_t) i * n_channels + c];
        mono[i] = (float) acc / n_channels / 32768.0f;
    }
    return push_mono(s, mono, src_rate);
}

int whisper_stream_poll_partial(whisper_stream * s, char * buf, int buf_size) {
    if (!s || !buf || buf_size <= 0 || !s->partial_dirty) return 0;
    std::snprintf(buf, buf_size, "%s", s->partial.c_str());
    s->partial_dirty = false;
    return 1;
}

int whisper_stream_poll_final(whisper_stream * s, char * buf, int buf_size) {
    if (!s || !buf || buf_size <= 0 || s->finals.empty()) return 0;
    std::snprintf(buf, buf_size, "%s", s->finals.front().c_str());
    s->finals.pop_front();
    return 1;
}

void whisper_stream_flush(whisper_stream * s) {
    if (!s) return;
    if (s->in_utt) s->finalize(s->abs_end());
}

double whisper_stream_last_decode_ms(const whisper_stream * s) {
    return s ? s->last_decode_ms : 0.0;
}

} // extern "C"
