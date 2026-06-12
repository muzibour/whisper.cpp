// sts-translate — P0 feasibility spike for the real-time multilingual
// Speech-to-Speech (STS) translation pipeline.
//
// See docs/realtime-sts-translation-plan.md (v2). This spike answers ONE
// question with real numbers: can whisper.cpp serve as a *streaming* STT stage
// inside the existing ai_conference app's "Fast (~150 ms)" tier, beating the
// incumbent sherpa-onnx Whisper (offline, whole-segment) backend?
//
// What it does:
//   • Reads a 16 kHz mono WAV file.
//   • Simulates real-time streaming: slides a fixed-length window over the
//     audio, advancing by a hop each step, and runs whisper.cpp on each window
//     (single_segment, greedy, hardcoded language, trimmed audio_ctx) — exactly
//     how a live sender pipeline would feed it. Emits a PARTIAL hypothesis per
//     hop and times each decode.
//   • Runs the result through the two downstream STS stages as TIMED SEAMS:
//       MT  (source→target text)  — passthrough stub here; real engine is
//                                    Marian/OPUS-MT or the SeamlessM4T text path.
//       TTS (target speech)       — stub here; real engine is VITS/Piper/Kokoro.
//     The seams exist so the harness measures the *whole* per-chunk budget and
//     so P1/P2 can drop real engines in without reshaping this file.
//   • Prints per-stage timing stats (min/avg/p50/p90/max) and the streaming
//     real-time factor, then a one-shot "batch" baseline (whole file at once)
//     for comparison.
//
// This is a measurement tool, not a product. It has no SDL/audio-out/network
// dependencies — it links only `common` + `whisper`.

#include "common.h"
#include "common-whisper.h"
#include "whisper.h"
#include "whisper_stream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::high_resolution_clock;

static int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               clk::now().time_since_epoch())
        .count();
}

// ── CLI parameters ────────────────────────────────────────────────────────────
struct sts_params {
    std::string model     = "models/ggml-base.en.bin";
    std::string fname     = "samples/jfk.wav";
    std::string language  = "en";   // hardcode source lang (skip auto-detect)
    std::string target    = "en";   // target language (MT seam target)

    int  n_threads  = std::min(4, (int) std::thread::hardware_concurrency());
    int  length_ms  = 1500;         // window length decoded each step
    int  step_ms    = 500;          // new audio per step (hop)
    int  keep_ms    = 200;          // context carried from the previous window
    int  audio_ctx  = 0;            // 0 = model default; trim to speed short chunks

    bool translate  = false;        // whisper X→en task (only en target)
    bool use_gpu    = true;
    bool flash_attn = false;
    bool batch_only = false;        // skip streaming, just run the batch baseline

    std::string tts_out;            // if set, render final text to this WAV (TTS seam)

    bool stream_mode = false;       // P1: drive the whisper_stream wrapper instead of file-slicing
    int  src_rate    = 48000;       // simulated live-source sample rate (resampled to 16 kHz)
    int  endpoint_ms = 600;         // trailing silence that finalizes a segment (VAD)
};

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "\nusage: %s [options]\n\n"
        "  -m  MODEL       model path            (default: models/ggml-base.en.bin)\n"
        "  -f  FILE        16 kHz mono WAV       (default: samples/jfk.wav)\n"
        "  -l  LANG        source language       (default: en; \"auto\" to detect)\n"
        "  -tl LANG        target language (MT)  (default: en)\n"
        "  -t  N           threads               (default: %d)\n"
        "      --length MS window length decoded each step (default: 1500)\n"
        "      --step   MS hop / new audio per step        (default: 500)\n"
        "      --keep   MS context carried over            (default: 200)\n"
        "      --audio-ctx N  trim encoder ctx (0=default) (default: 0)\n"
        "      --translate    whisper X->en task\n"
        "      --no-gpu       disable GPU\n"
        "      --flash-attn   enable flash attention\n"
        "      --batch        batch baseline only (no streaming sim)\n"
        "      --tts-out FILE render final text to a WAV (Windows SAPI; TTS seam demo)\n"
        "      --stream       drive the whisper_stream wrapper (live-stream sim: resample+VAD)\n"
        "      --src-rate HZ  simulated live-source rate for --stream  (default: 48000)\n"
        "      --endpoint MS  trailing silence that finalizes a segment (default: 600)\n\n",
        argv0, std::min(4, (int) std::thread::hardware_concurrency()));
}

static bool parse_args(int argc, char ** argv, sts_params & p) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")             p.model     = next("-m");
        else if (a == "-f")             p.fname     = next("-f");
        else if (a == "-l")             p.language  = next("-l");
        else if (a == "-tl")            p.target    = next("-tl");
        else if (a == "-t")             p.n_threads = std::stoi(next("-t"));
        else if (a == "--length")       p.length_ms = std::stoi(next("--length"));
        else if (a == "--step")         p.step_ms   = std::stoi(next("--step"));
        else if (a == "--keep")         p.keep_ms   = std::stoi(next("--keep"));
        else if (a == "--audio-ctx")    p.audio_ctx = std::stoi(next("--audio-ctx"));
        else if (a == "--translate")    p.translate = true;
        else if (a == "--no-gpu")       p.use_gpu   = false;
        else if (a == "--flash-attn")   p.flash_attn = true;
        else if (a == "--batch")        p.batch_only = true;
        else if (a == "--tts-out")      p.tts_out   = next("--tts-out");
        else if (a == "--stream")       p.stream_mode = true;
        else if (a == "--src-rate")     p.src_rate  = std::stoi(next("--src-rate"));
        else if (a == "--endpoint")     p.endpoint_ms = std::stoi(next("--endpoint"));
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); print_usage(argv[0]); return false; }
    }
    if (p.translate) p.target = "en"; // whisper translate only yields English
    return true;
}

// ── Timing accumulator ────────────────────────────────────────────────────────
struct stage_stats {
    std::string name;
    std::vector<double> ms; // one sample per step

    void add(double v) { ms.push_back(v); }

    double pct(double q) const {
        if (ms.empty()) return 0.0;
        std::vector<double> s = ms;
        std::sort(s.begin(), s.end());
        double idx = q * (s.size() - 1);
        size_t lo = (size_t) std::floor(idx), hi = (size_t) std::ceil(idx);
        return s[lo] + (s[hi] - s[lo]) * (idx - lo);
    }
    double min() const { return ms.empty() ? 0 : *std::min_element(ms.begin(), ms.end()); }
    double max() const { return ms.empty() ? 0 : *std::max_element(ms.begin(), ms.end()); }
    double avg() const {
        return ms.empty() ? 0 : std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
    }
};

static void print_stats(const stage_stats & s) {
    printf("  %-10s  n=%3zu  min=%6.1f  avg=%6.1f  p50=%6.1f  p90=%6.1f  max=%6.1f  ms\n",
           s.name.c_str(), s.ms.size(), s.min(), s.avg(), s.pct(0.50), s.pct(0.90), s.max());
}

// ── Downstream STS seams (stubs — real engines land in P1/P2) ─────────────────
// MT: source text → target-language text. Real impl: Marian/OPUS-MT (preferred,
// permissive) or the SeamlessM4T text path. Passthrough here.
static std::string mt_translate(const std::string & src_text,
                                const std::string & /*src_lang*/,
                                const std::string & /*tgt_lang*/) {
    return src_text; // seam: plug a real translator here
}

// TTS: target text → PCM. Real impl: VITS/Piper/Kokoro. No-op here (the spike
// measures the STT budget; TTS cost is engine-specific and benchmarked in P2).
// Kept as a no-op in the per-chunk loop so the timing stats stay clean.
static void tts_synthesize(const std::string & /*text*/) {
    // seam: plug a real synthesizer here
}

// Real TTS-to-WAV, invoked once on the final text (see --tts-out). Uses Windows
// SAPI (System.Speech) via PowerShell so the spike needs no TTS model download;
// the production engine is VITS/Piper/Kokoro wired into the seam above in P2.
// NOTE: voice is the system default (English) — with the MT stub the text is still
// English, so this proves the speech-out plumbing, not the translation itself.
static bool tts_write_wav(const std::string & text, const std::string & out_wav) {
    if (out_wav.empty()) return false;
    if (text.empty()) { fprintf(stderr, "TTS: empty text, nothing to synthesize\n"); return false; }
#ifdef _WIN32
    // Stage the text in a sidecar file to dodge shell-quoting issues.
    const std::string txt = out_wav + ".txt";
    { std::ofstream f(txt, std::ios::binary);
      if (!f) { fprintf(stderr, "TTS: cannot write %s\n", txt.c_str()); return false; }
      f << text; }
    const std::string ps =
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.SetOutputToWaveFile('" + out_wav + "'); "
        "$s.Speak([IO.File]::ReadAllText('" + txt + "')); "
        "$s.Dispose()";
    const std::string cmd = "powershell -NoProfile -Command \"" + ps + "\"";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) { fprintf(stderr, "TTS: SAPI synthesis failed (rc=%d)\n", rc); return false; }
    fprintf(stderr, "TTS: wrote %s\n", out_wav.c_str());
    return true;
#else
    (void) text;
    fprintf(stderr, "TTS: --tts-out uses Windows SAPI, not available on this platform (%s)\n",
            out_wav.c_str());
    return false;
#endif
}

static std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\n\r");
    return s.substr(a, b - a + 1);
}

static whisper_full_params make_wparams(const sts_params & p, bool streaming) {
    whisper_full_params w = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    w.print_progress   = false;
    w.print_special    = false;
    w.print_realtime   = false;
    w.print_timestamps = false;
    w.translate        = p.translate;
    w.language         = p.language.c_str();
    w.detect_language  = false;
    w.n_threads        = p.n_threads;
    w.audio_ctx        = p.audio_ctx;
    w.no_timestamps    = true;
    w.temperature_inc  = 0.0f;        // disable temperature fallback — keep latency bounded
    if (streaming) {
        w.single_segment = true;      // one rolling hypothesis per window
        w.no_context     = true;      // each window decoded independently
        w.max_tokens     = 64;
    }
    return w;
}

// ── P1 streaming demo ─────────────────────────────────────────────────────────
// Drives the whisper_stream wrapper exactly as the live app would: it turns the
// 16 kHz file into a synthetic *raw stream* (upsampled to --src-rate, stereo) and
// pushes it in 20 ms frames, so the wrapper exercises downmix + resample + VAD
// endpointing — no file semantics. Prints PARTIALs as they evolve and FINALs at
// each endpoint, plus the per-partial decode-latency budget.
static int run_stream_demo(const sts_params & p, const std::vector<float> & pcm16) {
    const int    sr16   = WHISPER_SAMPLE_RATE;
    const double in_sec = (double) pcm16.size() / sr16;

    // Build a synthetic source: resample 16 kHz mono -> src_rate, duplicate to stereo.
    std::vector<float> src; // interleaved L/R at p.src_rate
    const double ratio = (double) p.src_rate / sr16;
    const int    out_n = (int) (pcm16.size() * ratio);
    src.resize((size_t) out_n * 2);
    for (int i = 0; i < out_n; ++i) {
        const double t  = i / ratio;       // position in 16 kHz samples
        const int    j  = (int) t;
        const double f  = t - j;
        const float  a  = pcm16[std::min((size_t) j, pcm16.size() - 1)];
        const float  b  = pcm16[std::min((size_t) j + 1, pcm16.size() - 1)];
        const float  v  = a * (1.0f - (float) f) + b * (float) f;
        src[(size_t) i * 2 + 0] = v;
        src[(size_t) i * 2 + 1] = v;
    }

    whisper_stream_params wp = whisper_stream_default_params();
    wp.model       = p.model.c_str();
    wp.language    = p.language.c_str();
    wp.translate   = p.translate ? 1 : 0;
    wp.use_gpu     = p.use_gpu ? 1 : 0;
    wp.flash_attn  = p.flash_attn ? 1 : 0;
    wp.n_threads   = p.n_threads;
    wp.audio_ctx   = p.audio_ctx;
    wp.window_ms   = p.length_ms;
    wp.hop_ms      = p.step_ms;
    wp.endpoint_ms = p.endpoint_ms;

    whisper_stream * st = whisper_stream_init(&wp);
    if (!st) { fprintf(stderr, "error: whisper_stream_init failed (model '%s')\n", p.model.c_str()); return 3; }

    printf("\n=== STS P1 streaming wrapper ===\n");
    printf("model      : %s\n", p.model.c_str());
    printf("source     : %s  (%.2f s) -> simulated %d Hz stereo stream -> 16 kHz mono\n",
           p.fname.c_str(), in_sec, p.src_rate);
    printf("lang       : %s -> %s%s   gpu: %s   audio_ctx: %d\n",
           p.language.c_str(), p.target.c_str(), p.translate ? " (whisper translate)" : "",
           p.use_gpu ? "on" : "off", p.audio_ctx);
    printf("window/hop : %d / %d ms   endpoint silence: %d ms\n", p.length_ms, p.step_ms, p.endpoint_ms);
    printf("\n--- live stream (P=partial, FINAL=endpointed segment) ---\n");

    const int frame_ms = 20;
    const int frame_n  = p.src_rate * frame_ms / 1000;   // samples-per-channel per push
    stage_stats stt{"STT(partial)"};
    char buf[1024];
    std::string last_shown;

    for (int off = 0; off < out_n; off += frame_n) {
        const int n = std::min(frame_n, out_n - off);
        whisper_stream_push_f32(st, src.data() + (size_t) off * 2, n, 2, p.src_rate);

        const double t_now = (double) (off + n) / p.src_rate;
        if (whisper_stream_poll_partial(st, buf, sizeof(buf))) {
            const double d = whisper_stream_last_decode_ms(st);
            stt.add(d);
            if (buf[0] && last_shown != buf) {
                printf("  [%5.1fs | P     | %6.1f ms] %s\n", t_now, d, buf);
                last_shown = buf;
            }
        }
        while (whisper_stream_poll_final(st, buf, sizeof(buf)))
            printf("  [%5.1fs | FINAL |         ] %s\n", t_now, buf);
    }

    whisper_stream_flush(st);
    while (whisper_stream_poll_final(st, buf, sizeof(buf)))
        printf("  [%5.1fs | FINAL |         ] %s\n", in_sec, buf);

    printf("\n--- partial-decode latency ---\n");
    print_stats(stt);
    printf("budget check (Fast tier, partial p90 < 150 ms): %s (p90 = %.1f ms)\n",
           stt.pct(0.90) < 150.0 ? "PASS" : "FAIL", stt.pct(0.90));

    whisper_stream_free(st);
    printf("\n");
    return 0;
}

int main(int argc, char ** argv) {
    sts_params p;
    if (!parse_args(argc, argv, p)) return 1;

    // ── Load audio ────────────────────────────────────────────────────────────
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(p.fname, pcmf32, pcmf32s, /*stereo=*/false)) {
        fprintf(stderr, "error: failed to read audio '%s' (need 16 kHz mono WAV)\n", p.fname.c_str());
        return 2;
    }
    const int    sr        = WHISPER_SAMPLE_RATE;
    const double audio_sec = (double) pcmf32.size() / sr;

    // ── P1: streaming wrapper mode (no file semantics — drives whisper_stream) ──
    if (p.stream_mode) {
        return run_stream_demo(p, pcmf32);
    }

    // ── Init whisper ──────────────────────────────────────────────────────────
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = p.use_gpu;
    cparams.flash_attn = p.flash_attn;

    const int64_t t_load0 = now_us();
    whisper_context * ctx = whisper_init_from_file_with_params(p.model.c_str(), cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to load model '%s'\n", p.model.c_str());
        return 3;
    }
    const double load_ms = (now_us() - t_load0) / 1000.0;

    printf("\n=== STS P0 spike ===\n");
    printf("model      : %s\n", p.model.c_str());
    printf("audio      : %s  (%.2f s, %zu samples @ %d Hz)\n",
           p.fname.c_str(), audio_sec, pcmf32.size(), sr);
    printf("lang       : %s -> %s%s\n", p.language.c_str(), p.target.c_str(),
           p.translate ? "  (whisper translate)" : "");
    printf("threads    : %d   gpu: %s   flash_attn: %s   audio_ctx: %d\n",
           p.n_threads, p.use_gpu ? "on" : "off", p.flash_attn ? "on" : "off", p.audio_ctx);
    printf("model load : %.1f ms\n", load_ms);

    // ── Streaming simulation ──────────────────────────────────────────────────
    if (!p.batch_only) {
        const int length_n = (p.length_ms * sr) / 1000;
        const int step_n   = std::max(1, (p.step_ms * sr) / 1000);
        (void) p.keep_ms; // keep_ms is folded into length_ms here; exposed for P1 tuning

        printf("\n--- streaming sim (window=%d ms, hop=%d ms) ---\n", p.length_ms, p.step_ms);

        stage_stats stt{"STT"}, mt{"MT"}, tts{"TTS"}, total{"chunk"};
        std::string last_partial;

        // warm-up decode (first call pays one-time GPU/graph alloc; not counted)
        {
            int warm_n = std::min((int) pcmf32.size(), step_n);
            whisper_full_params w = make_wparams(p, true);
            whisper_full(ctx, w, pcmf32.data(), warm_n);
        }

        int step_idx = 0;
        for (int pos = step_n; pos <= (int) pcmf32.size() + step_n - 1; pos += step_n) {
            const int end   = std::min(pos, (int) pcmf32.size());
            const int begin = std::max(0, end - length_n);
            const int n     = end - begin;
            if (n <= 0) break;

            // STT (real)
            const int64_t t0 = now_us();
            whisper_full_params w = make_wparams(p, true);
            if (whisper_full(ctx, w, pcmf32.data() + begin, n) != 0) {
                fprintf(stderr, "warning: whisper_full failed at step %d\n", step_idx);
                continue;
            }
            std::string partial;
            const int nseg = whisper_full_n_segments(ctx);
            for (int s = 0; s < nseg; ++s) partial += whisper_full_get_segment_text(ctx, s);
            partial = trim(partial);
            const double stt_ms = (now_us() - t0) / 1000.0;

            // MT (seam)
            const int64_t t1 = now_us();
            std::string translated = mt_translate(partial, p.language, p.target);
            const double mt_ms = (now_us() - t1) / 1000.0;

            // TTS (seam)
            const int64_t t2 = now_us();
            tts_synthesize(translated);
            const double tts_ms = (now_us() - t2) / 1000.0;

            stt.add(stt_ms); mt.add(mt_ms); tts.add(tts_ms);
            total.add(stt_ms + mt_ms + tts_ms);

            if (partial != last_partial) {
                printf("  [%5.1fs | %6.1f ms] %s\n", end / (double) sr, stt_ms, partial.c_str());
                last_partial = partial;
            }
            ++step_idx;
        }

        printf("\n--- per-chunk timing (%d steps) ---\n", step_idx);
        print_stats(stt); print_stats(mt); print_stats(tts); print_stats(total);

        const double wall_ms = total.avg() * step_idx;
        const double rtf     = (wall_ms / 1000.0) / audio_sec;
        printf("\nstreaming wall time : %.0f ms for %.2f s audio  (RTF %.2fx, %s real-time)\n",
               wall_ms, audio_sec, rtf, rtf < 1.0 ? "faster than" : "SLOWER than");
        printf("budget check (Fast tier, target STT p90 < 150 ms): %s (p90 = %.1f ms)\n",
               stt.pct(0.90) < 150.0 ? "PASS" : "FAIL", stt.pct(0.90));
    }

    // ── Batch baseline (whole file at once) ───────────────────────────────────
    {
        printf("\n--- batch baseline (whole file, one decode) ---\n");
        whisper_full_params w = make_wparams(p, false);
        const int64_t t0 = now_us();
        if (whisper_full(ctx, w, pcmf32.data(), (int) pcmf32.size()) != 0) {
            fprintf(stderr, "error: batch whisper_full failed\n");
        } else {
            const double batch_ms = (now_us() - t0) / 1000.0;
            std::string text;
            for (int s = 0; s < whisper_full_n_segments(ctx); ++s)
                text += whisper_full_get_segment_text(ctx, s);
            printf("  text : %s\n", trim(text).c_str());
            printf("  time : %.0f ms for %.2f s audio (RTF %.2fx)\n",
                   batch_ms, audio_sec, (batch_ms / 1000.0) / audio_sec);

            // TTS seam: render the final (MT'd) text to a real WAV if requested.
            if (!p.tts_out.empty()) {
                const std::string tgt = mt_translate(trim(text), p.language, p.target);
                printf("\n--- TTS seam (final text -> %s) ---\n", p.tts_out.c_str());
                tts_write_wav(tgt, p.tts_out);
            }
        }
    }

    whisper_free(ctx);
    printf("\n");
    return 0;
}
