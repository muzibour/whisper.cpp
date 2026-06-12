# CLAUDE.md — whisper.cpp (STS translation fork)

> This file mirrors the project's persistent memory into always-visible, committed
> context. It is the human-readable counterpart to the auto-memory under
> `C:\Users\emb-muzireh\.claude\projects\w--whisper-cpp\memory\`. Keep the two in sync
> when decisions change.

## What this fork is for

whisper.cpp here is the **STT third** of a real-time multilingual **speech-to-speech (STS)
translation** effort for video calls (Teams/Zoom-style), with voice-matched synthesis and
viseme lip-sync.

- **Latency goal:** `<200 ms` local compute / `<350 ms` mouth-to-ear same-region.
- **Plan:** `w:/whisper.cpp/docs/realtime-sts-translation-plan.md` (v2).

### The bigger picture — the existing app

A mature working app already exists (the **"ai_conference app"**) at:
`E:/Muzibour/Projects/VW/ADAS/AIML/DeepSeek/our_venture/Rhythm/Avarix/implementation/cognitive_multilingual_xr_platform/native/ai_conference`

It already provides:
- mediasoup SFU transport;
- `ITranslationEngine` + factory + hot-swap router (`client/translation_engine.hpp`);
- engines: OfflineSherpa (sherpa-onnx Whisper STT + VITS/Piper TTS), DirectSeamless
  (SeamlessM4T any→any S2ST sidecar), Azure/Gemini/OpenAI/ElevenLabs/Fish/Inworld;
- MediaPipe + `LipCompositor` (Delaunay video-warp) visemes; per-peer/per-listener language.

**whisper.cpp is NOT wired into that app yet** (only referenced in comments). Its job: add
`EngineType::OfflineWhisperCpp` — a streaming STT backend (option A: parallel to sherpa for
A/B benchmark) serving the **Fast (~150 ms)** tier.

## Locked decisions

1. Accept latency re-baseline.
2. Tier-1 voice **blend** default + Tier-2 **clone** GPU-opt-in.
3. Eval Marian vs NLLB for MT — **prefer Marian** (NLLB is CC-BY-NC, non-commercial).
4. **Sender-side per-language tracks (committed)** — migrate off the current receiver-side
   model in P3.
5. Transport = reuse mediasoup.

## Status

### P0 spike — DONE
Tool: `w:/whisper.cpp/examples/sts-translate/` (target `whisper-sts-translate`, registered in
`examples/CMakeLists.txt` non-SDL block; links only `common` + `whisper`).

`audio_ctx` is the dominant streaming-latency lever on CPU (`audio_ctx=0` runs the full 30 s
encoder context every chunk). base.en, CPU-only, 4 threads, AVX2, window 1500 ms / hop 500 ms:

| audio_ctx | STT p90 | verdict |
|-----------|---------|---------|
| 0 (full)  | 924 ms (RTF 1.78×) | ❌ |
| 512       | 236 ms | ❌ |
| 256       | 125 ms | ✅ best quality under 150 ms budget |
| 128       | 199 ms (p50 64 ms) | slight quality loss |

Batch baseline (whole 11 s file, one decode) ≈ 1015 ms (RTF 0.09×).

**Caveat:** `models/for-tests-ggml-*.bin` are 0.5 MB CI dummies → transcribe to empty text.
Download a real model: `sh ./models/download-ggml-model.sh base.en` (148 MB).

### GPU update (2026-06-12, RTX 2000 Ada 8 GB, CUDA 13.3, sm_89)
Built `-DGGML_CUDA=ON` (recipe below). Same clip/window/hop:
- base.en `audio_ctx=0` (full context, best quality) → p90 **48 ms** (vs 924 ms CPU). **On GPU
  `audio_ctx` is no longer a latency lever** for small models — run full context, ~100 ms left
  for MT/TTS. (`256`→25 ms, `512`→27 ms.)
- `large-v3-turbo-q5_0`: `512`→p90 **86 ms** (clean, Fast tier); `0`→441 ms (Balanced);
  `256`→160 ms — *worse AND slower*: too-small ctx triggers decoder repetition (`S NOT S NOT…`)
  that runs the token loop away.
- **Lesson:** tune `audio_ctx` per model; "smaller = faster" is false once quality degrades
  (decoder cost, not encoder, dominates there).

### P1 prototype — built (2026-06-12)
`examples/sts-translate/whisper_stream.{h,cpp}` — the streaming STT wrapper / C ABI that
`EngineType::OfflineWhisperCpp::accept_audio()` will drive.
- **C ABI:** `whisper_stream_init/free`, `push_i16/push_f32(frames,channels,src_rate)`,
  `poll_partial/poll_final`, `flush`, `last_decode_ms`.
- **Internals:** channel downmix + stateful linear resampler (e.g. 48 kHz→16 kHz) + rolling
  ring buffer + energy-RMS VAD endpointing.
- **Partials** = windowed decode at trimmed `audio_ctx` (low latency). **Finals** =
  full-utterance decode at `audio_ctx=0` (full context — critical: trimmed ctx on a
  multi-second utterance causes decoder-repetition runaway).
- **Driver:** `whisper-sts-translate --stream --src-rate 48000` synthesizes a stereo stream
  from the WAV to exercise resample+downmix+VAD (no file semantics). base.en/GPU partial
  p90 ≈ 50 ms, clean endpointed finals.

### Branch / build note (2026-06-12)
P1 work was committed on branch `arghh` (commit `20d22c9c`). Dev moved to
**`feature/whisper_adoptation`**, branched off a fresh upstream `master` (`df7638d8`). The
sts-translate tree (and this CLAUDE.md) were restored via `git checkout 20d22c9c -- <path>` and
the example re-registered with `add_subdirectory(sts-translate)` in `examples/CMakeLists.txt`
(after `vad-speech-segments`, non-SDL block). Builds & runs against the new master on both CPU
(`build/`) and CUDA (`build-cu13/`).

### Silero VAD swap — DONE (2026-06-12)
Replaced energy-RMS endpointing with the **native Silero VAD** now in upstream master:
`whisper_vad_init_from_file_with_params` + stateful `whisper_vad_detect_speech_no_reset`
over fixed 512-sample / 32 ms windows + `whisper_vad_reset_state` on each finalize. Energy
VAD kept as fallback (both unified into one cursor-based `apply_voiced` state machine).
- **New flags:** `--vad-model F` (NULL = energy fallback), `--vad-thold X` (prob 0.5 Silero /
  RMS 0.012 energy, auto-defaulted). C-ABI param: `whisper_stream_params.vad_model`.
- **Gotcha:** the Silero LSTM graph trips a CUDA backend assertion (`pre-allocated tensor …
  cannot run the operation`), so the VAD is forced **CPU-only even in the GPU build** — it's a
  0.88 MB model off the STT critical path, negligible.
- Installs a `whisper_log_set` filter (WARN/ERROR only) after model load to kill the
  4-INFO-lines-per-call VAD log spam.
- **Validated:** GPU STT + CPU Silero on jfk.wav → 3 clean endpointed finals, partial
  p90 **51.7 ms** (Fast PASS). Test model `models/for-tests-silero-v6.2.0-ggml.bin` (real,
  885 KB); `models/download-vad-model.sh` fetches silero-v5.1.2 / v6.2.0.

- **TODO P1 (remaining):** token timestamps; worker-thread push; A/B vs sherpa; wire into the
  app.

## Building with CUDA on this Windows box

Win11, VS2022 Community 17.14 / MSVC 19.44, RTX 2000 Ada (sm_89). Hard-won; several
non-obvious blockers.

**Toolchain:** CUDA 11.8 **rejects MSVC 19.44** (host_config check) and its VS integration
isn't in BuildCustomizations. Fix: install **CUDA 13.3** via
`winget install --id Nvidia.CUDA -e --disable-interactivity` (~2.3 GB, UAC elevates). 13.3
supports MSVC 19.44 and auto-drops VS integration into
`…/MSBuild/Microsoft/VC/v170/BuildCustomizations/CUDA 13.3.*`.

**Configure** (VS generator — MSBuild sets up its own MSVC env, no vcvars wrestling):
```bash
export CUDA_PATH="C:\\PROGRA~1\\NVIDIA~2\\CUDA\\v13.3"   # short path, no spaces
cmake -B build-cu13 -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DWHISPER_SDL2=OFF \
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3" \
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe"
```
Pin `CUDAToolkit_ROOT` / `CMAKE_CUDA_COMPILER` or `FindCUDAToolkit` silently grabs stale
**v11.8** includes/libs.

**Build** (two MSYS-bash gotchas):
```bash
export MSYS2_ARG_CONV_EXCL='*'   # else MSYS mangles /p:... → "MSB1008: Only one project"
cmake --build build-cu13 --target whisper-sts-translate --config Release -j -- \
  '-p:CudaToolkitDir=C:\PROGRA~1\NVIDIA~2\CUDA\v13.3\'   # -p: not /p:; short path; trailing backslash
```
MSBuild's CUDA targets need `CudaToolkitDir`; `CUDA_PATH` does NOT propagate through
`cmake --build`, so pass it as an explicit `-p:` property.

**Runtime:** CUDA 13 moved redistributable DLLs to `…/v13.3/bin/x64/` (cudart64_13,
cublas64_13, cublasLt64_13) — **not** `…/bin/`. Add `bin/x64` to PATH. Run from the exe's own
`bin/Release` dir so it finds sibling ggml/whisper DLLs:
```bash
cd build-cu13/bin/Release
export PATH="/c/PROGRA~1/NVIDIA~2/CUDA/v13.3/bin/x64:/c/PROGRA~1/NVIDIA~2/CUDA/v13.3/bin:$PATH"
./whisper-sts-translate.exe -m <model> -f <wav> -l en --audio-ctx 0
```
Success log: `ggml_cuda_init: found 1 CUDA devices … RTX 2000 Ada … compute capability 8.9`.

**Dead ends:** `vcvars64.bat` from a bare MSYS-spawned cmd silently no-ops (`vswhere.exe` not
on PATH); `call vcvars` truncates the rest of a .bat. The VS-generator path sidesteps both.
