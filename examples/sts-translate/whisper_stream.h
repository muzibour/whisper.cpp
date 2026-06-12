// whisper_stream — a minimal streaming STT wrapper over whisper.cpp for the
// real-time STS pipeline (P1). It turns a *raw, unbounded PCM stream* (e.g.
// mediasoup-decoded Opus frames) into low-latency partial hypotheses plus
// VAD-endpointed final segments — with no files involved.
//
// Pipeline inside:
//   push_*  -> downmix to mono -> linear resample to 16 kHz f32 -> ring buffer
//           -> energy VAD (speech/silence)  -> on hop: windowed PARTIAL decode
//                                           -> on trailing silence: FINAL decode
//   poll_partial / poll_final read the results out.
//
// This is the engine that EngineType::OfflineWhisperCpp's accept_audio() drives
// in the app: feed it network audio frames, read finals to send to MT/TTS.
//
// Pure C ABI so it links into any host (C/C++/FFI). Single-threaded: decoding
// happens synchronously inside push_* when a hop's worth of audio has arrived;
// in the app you'd run push_* on a worker thread fed by the network callback.

#ifndef WHISPER_STREAM_H
#define WHISPER_STREAM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct whisper_stream whisper_stream;

typedef struct whisper_stream_params {
    const char * model;        // path to ggml model
    const char * language;     // source language, e.g. "en" (hardcoded; no auto-detect)
    int   translate;           // 1 = whisper X->English task
    int   use_gpu;             // 1 = use GPU backend if available
    int   flash_attn;          // 1 = flash attention
    int   n_threads;           // decode threads
    int   audio_ctx;           // encoder context cap (0 = full 30 s)

    int   window_ms;           // trailing audio decoded for each PARTIAL   (default 1500)
    int   hop_ms;              // min new audio between PARTIAL decodes      (default 200)
    int   endpoint_ms;         // trailing silence that finalizes a segment  (default 600)
    int   max_utterance_ms;    // hard cap on a single utterance length      (default 15000)

    float vad_threshold;       // RMS energy above which a frame is "speech" (default 0.012)
    int   vad_frame_ms;        // granularity of the VAD energy check         (default 20)
} whisper_stream_params;

// Sensible defaults (see above).
whisper_stream_params whisper_stream_default_params(void);

// Create / destroy. Returns NULL on model-load failure.
whisper_stream * whisper_stream_init(const whisper_stream_params * params);
void             whisper_stream_free(whisper_stream * s);

// Push raw audio from the live source. n_frames = samples per channel;
// channels are interleaved; src_rate is the source sample rate (e.g. 48000).
// Audio is downmixed to mono and resampled to 16 kHz internally. Returns 0 on ok.
int whisper_stream_push_i16(whisper_stream * s, const int16_t * pcm,
                            int n_frames, int n_channels, int src_rate);
int whisper_stream_push_f32(whisper_stream * s, const float * pcm,
                            int n_frames, int n_channels, int src_rate);

// Read the latest PARTIAL hypothesis. Returns 1 and copies into buf if the
// partial changed since the last poll, else 0.
int whisper_stream_poll_partial(whisper_stream * s, char * buf, int buf_size);

// Pop the next FINAL (endpointed) segment, if any. Returns 1 and copies into
// buf, else 0. Call in a loop to drain all pending finals.
int whisper_stream_poll_final(whisper_stream * s, char * buf, int buf_size);

// Force-finalize any in-progress utterance (call at end-of-stream).
void whisper_stream_flush(whisper_stream * s);

// Last decode time in milliseconds (for benchmarking the per-chunk budget).
double whisper_stream_last_decode_ms(const whisper_stream * s);

#ifdef __cplusplus
}
#endif

#endif // WHISPER_STREAM_H
