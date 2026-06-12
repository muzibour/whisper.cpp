# sts-translate — P0 feasibility spike

A measurement tool for the **real-time multilingual speech-to-speech (STS)
translation** effort. See [`docs/realtime-sts-translation-plan.md`](../../docs/realtime-sts-translation-plan.md).

It answers one question with real numbers: **can whisper.cpp act as a *streaming*
STT stage inside the existing `ai_conference` app's "Fast (~150 ms)" tier**,
beating the incumbent sherpa-onnx Whisper (offline, whole-segment) backend?

It is a benchmark harness, not a product — no SDL / audio-out / network deps.
It links only `common` + `whisper`.

## What it does

1. Reads a 16 kHz mono WAV.
2. **Streaming sim:** slides a fixed-length window over the audio, advancing by a
   hop each step, and runs `whisper_full` per window (`single_segment`, greedy,
   hardcoded language, optional trimmed `audio_ctx`) — emitting an evolving
   PARTIAL hypothesis and timing each decode.
3. Runs each result through two **timed downstream seams**:
   - `mt_translate()` — source→target text. Stub (passthrough) today; real engine
     is Marian/OPUS-MT or the SeamlessM4T text path.
   - `tts_synthesize()` — target speech. Stub today; real engine is VITS/Piper/Kokoro.
4. Prints per-stage timing stats (min/avg/p50/p90/max), streaming real-time
   factor, a Fast-tier budget check, and a one-shot **batch baseline** for comparison.

## Build & run

```bash
cmake -B build -DWHISPER_SDL2=OFF
cmake --build build --target whisper-sts-translate --config Release -j

# needs a REAL model — the bundled models/for-tests-*.bin are 0.5 MB dummies
sh ./models/download-ggml-model.sh base.en

./build/bin/Release/whisper-sts-translate \
    -m models/ggml-base.en.bin -f samples/jfk.wav \
    -l en --step 500 --length 1500 --audio-ctx 256
```

Key flags: `-l` source lang (hardcode — skip auto-detect), `-tl` target lang,
`--length`/`--step`/`--keep` streaming window geometry, `--audio-ctx` (the latency
lever — see below), `--translate` (whisper X→en), `--no-gpu`, `--batch`,
`--tts-out FILE` (render the final text to a WAV via the TTS seam — Windows SAPI).

### Running on Windows (cmd / PowerShell, GPU build)

The CUDA build is `build-cu13` (configured with `-DGGML_CUDA=ON`). The exe
finds its sibling `ggml*.dll` / `whisper.dll` automatically (same folder), but the
**CUDA runtime DLLs live in `…\CUDA\v13.3\bin\x64`** (cudart/cublas) — that dir must
be on `PATH` or the exe fails to start. Run from the repo root:

**Command Prompt (cmd.exe):**

```bat
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA%\bin\x64;%CUDA%\bin;%PATH%"
build-cu13\bin\Release\whisper-sts-translate.exe ^
    -m models\ggml-base.en.bin -f samples\jfk.wav ^
    -l en --step 500 --length 1500 --audio-ctx 0
```

**PowerShell (pwsh):**

```powershell
$CUDA = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$env:Path = "$CUDA\bin\x64;$CUDA\bin;$env:Path"
.\build-cu13\bin\Release\whisper-sts-translate.exe `
    -m models\ggml-base.en.bin -f samples\jfk.wav `
    -l en --step 500 --length 1500 --audio-ctx 0
```

For a **CPU-only** build (the default `build` dir, no CUDA), drop the two PATH lines
and point at `build\bin\Release\whisper-sts-translate.exe`.

### Runner scripts (all the examples, three shells)

`run-examples.{sh,ps1,bat}` wrap every example below — they auto-locate the binary
(CUDA build preferred, CPU fallback) and set up the CUDA DLL `PATH`. Pass an example
name, `all`, or nothing to list them:

```bash
sh   examples/sts-translate/run-examples.sh            # bash / MSYS / Linux / macOS
pwsh examples/sts-translate/run-examples.ps1  all      # PowerShell
examples\sts-translate\run-examples.bat       fast     # cmd.exe
```

Examples: `full` (ac=0), `fast` (ac=256), `accuracy` (large-v3-turbo ac=512),
`aggressive` (ac=32, degraded), `sts` (en→es pipeline), `translate` (X→en), `batch`.

Each run echoes the **input WAV** and an **output file** path (saved under
`examples/sts-translate/out/<example>.txt`) and prints a **legend** of the
abbreviations (STT/MT/TTS, RTF, p50/p90, `ac`, flags). Run `… legend` for just the
glossary. The **`sts`** example additionally renders the final text to
**`out/sts.wav`** through the TTS seam (Windows SAPI) — proving the speech-out
plumbing end-to-end (still English, since the MT stage is a stub).

### Reproduce the benchmark tables

`bench-gpu.sh` re-runs the `audio_ctx` sweep for both models and prints
README-ready markdown rows (it handles the CUDA-13 `bin/x64` DLL PATH quirk):

```bash
sh examples/sts-translate/bench-gpu.sh
# override: BIN=... WAV=... CUDA=... CTXS="0 256 512" sh examples/sts-translate/bench-gpu.sh
```

Paste its output into the tables below to refresh the numbers.

## P1 — streaming wrapper (`whisper_stream`)

The spike above slices a *file*. The real app gets an **unbounded raw PCM stream**
(mediasoup-decoded Opus frames, ~48 kHz, possibly stereo) — never a file. P1 adds a
thin streaming layer that bridges that gap: [`whisper_stream.h`](whisper_stream.h) /
[`whisper_stream.cpp`](whisper_stream.cpp).

```text
push_i16/f32(frames, channels, src_rate)
   └─ downmix to mono ─ linear resample to 16 kHz ─ ring buffer
        └─ VAD (Silero / energy) ─┬─ every hop:    windowed PARTIAL decode (trimmed audio_ctx)
                                  └─ trailing silence: full-context FINAL decode (audio_ctx=0)
poll_partial() / poll_final() / flush()      ← drain results
```

It is the **C ABI** that `EngineType::OfflineWhisperCpp`'s `accept_audio()` drives in
the app — feed it network frames, read finals to hand to MT/TTS. Key points:

- **No files, no fixed length.** You `push` frames as they arrive; memory is bounded
  by a rolling ring buffer.
- **Partials vs finals.** Partials are low-latency windowed hypotheses (trimmed
  `audio_ctx`); a **final** is emitted when the VAD sees `--endpoint` ms of silence and
  is decoded at **full context** for quality (a too-small ctx on a multi-second
  utterance causes decoder-repetition runaway — learned the hard way).
- **Resampler + downmix** handle the 48 kHz-stereo → 16 kHz-mono conversion whisper
  requires.
- **VAD backend (`--vad-model`).** Endpointing uses **Silero** (neural) when a VAD
  ggml model is supplied — streamed through whisper.cpp's stateful
  `whisper_vad_detect_speech_no_reset` over 32 ms windows, far more robust to
  noise/music than energy. Without `--vad-model` it falls back to a simple **RMS
  energy** detector. The Silero graph runs on **CPU** even in the GPU build (its
  LSTM trips a CUDA backend assertion in this whisper.cpp, and it's a 0.88 MB model
  off the STT critical path). Threshold via `--vad-thold` (prob for Silero, RMS for
  energy; auto-defaults 0.5 / 0.012).

### Try it

`--stream` turns the input WAV into a *synthetic* 48 kHz stereo stream (to exercise
resample + downmix) and drives the wrapper frame-by-frame:

```bash
whisper-sts-translate -m models/ggml-base.en.bin -f samples/jfk.wav \
    -l en --stream --src-rate 48000 --step 200 --audio-ctx 0
# or: sh examples/sts-translate/run-examples.sh stream

# with neural Silero endpointing (download a VAD model, or use the test fixture):
whisper-sts-translate -m models/ggml-base.en.bin -f samples/jfk.wav \
    -l en --stream --audio-ctx 0 \
    --vad-model models/for-tests-silero-v6.2.0-ggml.bin
```

Output shows `P` (partial) lines evolving and `FINAL` lines at each VAD endpoint, e.g.:

```text
[  2.9s | FINAL] And so, my fellow Americans.
[  5.0s | FINAL] Ask not!
[ 11.0s | FINAL] What your country can do for you, ask what you can do for your country.
```

Tuning flags: `--src-rate HZ` (simulated source rate), `--step MS` (partial hop),
`--endpoint MS` (silence to finalize), `--audio-ctx N` (partial latency/quality),
`--vad-model F` / `--vad-thold X` (neural Silero endpointing — see above).
Still STT-only — the finals are what you'd feed into the MT → TTS stages.

## Understanding `--audio-ctx` (the latency lever)

`audio_ctx` caps **how many encoder positions whisper computes** — i.e. how much
of the audio the encoder actually does work on.

Whisper's encoder is built around a fixed **30 s** input:

```text
30 s audio → log-mel (3000 frames @ 10 ms) → 2 conv (stride 2) → 1500 encoder positions
```

So the full context is **`n_audio_ctx = 1500` positions, each ≈ 20 ms** of audio
(1500 × 20 ms = 30 s). `--audio-ctx N` computes only the first `N` positions;
`--audio-ctx 0` means "use the default 1500" (full 30 s).

| `--audio-ctx` | positions | audio it covers | relative encoder work |
|--------------:|----------:|----------------:|:----------------------|
| `0` (=1500)   | 1500      | 30 s            | full (baseline)       |
| `768`         | 768       | ~15 s           | ~½                    |
| `512`         | 512       | ~10 s           | ~⅓                    |
| `256`         | 256       | ~5.1 s          | ~⅙                    |
| `128`         | 128       | ~2.6 s          | ~1/12                 |

**Why it moves latency.** Encoder self-attention is `O(n²)` in positions and the
rest scales with `n`, so 1500 → 256 slashes per-chunk encoder cost (on CPU base.en
that was 924 ms → 125 ms p90).

**Why trimming is usually free for streaming.** Whisper *always pads its input to
30 s internally*. A 1.5 s streaming window only needs ~75 real positions
(1.5 s ÷ 20 ms); the other ~1425 are computing on **silent padding** — pure waste.
`--audio-ctx 256` keeps a comfortable margin over a 1.5 s window (covers 5.1 s)
while skipping most of that padding, so quality is unaffected.

### Rule of thumb

Keep `audio_ctx ≥ window_seconds ÷ 0.02`, with margin. For a 1500 ms window the
content needs ~75 positions, so 128–256 is safe; going below your window's real
content blinds the encoder to the end of the audio and quality drops.

### Examples

```bash
# Full quality, no trimming (best when you have GPU headroom — see findings)
whisper-sts-translate -m models/ggml-base.en.bin -f samples/jfk.wav \
    -l en --length 1500 --step 500 --audio-ctx 0

# CPU Fast-tier sweet spot for base.en: ~6× less encoder work, quality intact
whisper-sts-translate -m models/ggml-base.en.bin -f samples/jfk.wav \
    -l en --length 1500 --step 500 --audio-ctx 256

# Accuracy model: 256 is TOO small here (repetition artifact) — 512 is the sweet spot
whisper-sts-translate -m models/ggml-large-v3-turbo-q5_0.bin -f samples/jfk.wav \
    -l en --length 1500 --step 500 --audio-ctx 512

# Too aggressive: encoder can't see a 1.5 s window's tail → garbled / dropped words
whisper-sts-translate -m models/ggml-base.en.bin -f samples/jfk.wav \
    -l en --length 1500 --step 500 --audio-ctx 32
```

**Caveats (measured below):** the sweet spot is **per-model** — on `large-v3-turbo`,
`audio_ctx=256` is both *worse and slower* than `512`, because degraded encoding makes
the decoder hallucinate repetitions (`S NOT S NOT …`) that run the token loop away
(decoder cost, not encoder, then dominates). And on **GPU** the full 1500 positions
already run in ~48 ms for base.en, so `audio_ctx` stops mattering for latency — just
use `0` and keep best quality. In code this is the `audio_ctx` field of
`whisper_full_params` (maps to `whisper_hparams.n_audio_ctx`).

## P0 findings (base.en, CPU-only build, 4 threads, AVX2)

The streaming cost is dominated by the **encoder**, and the encoder runs its
*full 30 s context every chunk* unless `audio_ctx` is trimmed. On an 11 s clip,
window 1500 ms / hop 500 ms:

| `audio_ctx` | STT p50 | STT p90 | RTF | Fast-tier (<150 ms p90) | Quality |
|------------:|--------:|--------:|----:|:----------------------:|:--------|
| 0 (full)    | 886 ms  | 924 ms  | 1.78× | ❌ | full |
| 512         | 225 ms  | 236 ms  | ~0.4× | ❌ | full |
| **256**     | **109 ms** | **125 ms** | **~0.23×** | ✅ | full |
| 128         | 64 ms   | 199 ms  | 0.17× | ✅* | slight loss |

**Conclusion (CPU):** whisper.cpp streaming STT hits the Fast (~150 ms) tier on
**CPU-only** base.en at `audio_ctx≈256` with transcription quality intact
(`audio_ctx=256` is the sweet spot: best quality under budget).

## P0 findings (GPU: RTX 2000 Ada Laptop 8 GB, CUDA 13.3, sm_89)

Built with `-DGGML_CUDA=ON`. Same clip / window / hop. The GPU changes the picture:

| model | `audio_ctx` | STT p50 | STT p90 | (CPU p90) | note |
|-------|------------:|--------:|--------:|----------:|:-----|
| base.en | **0 (full 30 s)** | 42 ms | **48 ms** | 924 ms | full context, best quality, stable |
| base.en | 256 | 17 ms | 25 ms | 125 ms | trimmed |
| base.en | 512 | 20 ms | 27 ms | 236 ms | trimmed |
| large-v3-turbo-q5_0 | 0 (full) | 429 ms | 441 ms | — | accuracy model, **balanced** tier |
| large-v3-turbo-q5_0 | 256 | 155 ms | 160 ms | — | over budget + repetition artifact |
| large-v3-turbo-q5_0 | **512** | 81 ms | **86 ms** | — | clean output, **under Fast budget** ✅ |

**GPU conclusions:**
1. **On GPU, `audio_ctx` is no longer the latency lever it is on CPU.** base.en
   runs the *full* 30 s encoder context (`audio_ctx=0`, best quality) at p90 48 ms —
   ~19× faster than CPU. Prefer `audio_ctx=0` for small models on GPU; you no longer
   trade quality for latency, and ~100 ms of budget is left for MT/TTS.
2. **The accuracy model is real-time-capable on an 8 GB laptop GPU.**
   `large-v3-turbo-q5_0` runs at p90 86 ms (`audio_ctx=512`, clean) for the Fast
   tier, or full-context p90 441 ms for the Balanced tier.
3. **`audio_ctx` must be tuned per model — "smaller = faster" is false once it hurts
   quality.** On large-turbo, `audio_ctx=256` triggers decoder repetition
   (`S NOT S NOT …`) that runs the token loop away, making it both lower-quality *and*
   ~2× slower than `audio_ctx=512`. The decoder cost, not the encoder, dominates there.

This validates the P0 go/no-go: **whisper.cpp is a viable streaming STT backend**
on both CPU (Fast tier, base.en) and GPU (Fast tier even with the large-turbo
accuracy model). Next (P1): VAD-gated end-pointing for finalized segments,
token timestamps, and a C-ABI wrapper to link into the `ai_conference` client as
`EngineType::OfflineWhisperCpp`, A/B-benchmarked against `OfflineSherpa`.

## Caveats

- The bundled `models/for-tests-ggml-*.bin` are 0.5 MB CI fixtures — they exercise
  the code path but transcribe to empty text. Use a real downloaded model.
- GPU numbers above are from a `-DGGML_CUDA=ON` build (CUDA 13.3, RTX 2000 Ada).
  The default CPU-only build (`gpu: on` with no GPU backend) falls back to CPU.
- MT/TTS stages are stubs; their cost is added to the chunk budget once real
  engines land in P1/P2.
- Streaming re-decodes overlapping audio each step (sliding window). P1 replaces
  this with VAD end-pointing so finalized segments aren't re-decoded.
