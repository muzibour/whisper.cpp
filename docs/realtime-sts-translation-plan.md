# Real-Time Multilingual STS Translation — Engineering Plan (v2, reality-grounded)

> **v2 rewrite.** The v1 plan was written *before* discovering that a working
> conferencing application already exists at
> `…/cognitive_multilingual_xr_platform/native/ai_conference`. Most of what v1
> proposed to build from scratch (WebRTC transport, NMT stage, viseme/lip-sync,
> engine abstraction) **already exists there.** This version re-scopes the work
> to whisper.cpp's *actual* role inside that app, and records the stakeholder
> decisions taken in the planning session.

Source KB: `W:\knowledge_base\kokoro.cpp- More details.docx`.
Existing app: `E:\…\cognitive_multilingual_xr_platform\native\ai_conference` (the "ai_conference app").

---

## 0. What already exists (and therefore is NOT whisper.cpp's job)

The ai_conference app is a mature native C++ STS conferencing client. It already provides:

| Concern | Already implemented in ai_conference | Implication for this plan |
|---------|--------------------------------------|---------------------------|
| **Transport** | mediasoup SFU via `libmediasoupclient` (`mediasoup_wrapper`, `room_client`, `signaling_client`) | **Transport is solved.** v1's "pick libwebrtc/Teams/SFU" decision is moot. |
| **Engine abstraction** | `ITranslationEngine` + `TranslationEngineFactory` + hot-swappable `TranslationEngineRouter` (`translation_engine.hpp`) | New STT/STS work plugs in as a backend/engine behind this interface — no new app scaffolding. |
| **Engines shipped** | OfflineSherpa, **DirectSeamless** (Meta SeamlessM4T sidecar), StreamingSidecar (WS), Azure, Gemini, OpenAI, ElevenLabs, Fish, Inworld | Cascade *and* direct S2ST already work. |
| **NMT / any→any** | **SeamlessM4T** direct text-free S2ST sidecar (`seamless_sidecar/`), plus an external-MT seam (`translate_text()`), plus Whisper-translate (→en) | v1's "the NMT stage is missing!" is **false for this app** — direct any→any already exists. |
| **STT (offline)** | **sherpa-onnx Whisper** (ONNX), VAD = Silero, per-peer (`offline_sherpa_backend.cpp`) | This is the seam whisper.cpp replaces/competes with — see §2. |
| **TTS (offline)** | VITS / Piper via sherpa-onnx; cloud voices (ElevenLabs/Fish/Inworld) | Kokoro is *not* wired; VITS/Piper is the current offline voice. |
| **Viseme / lip-sync** | `VisemeFrame` (SAPI/Azure 0–21 ordinals), Azure real-time visemes, offline energy-envelope/text-heuristic, **`LipCompositor`** Delaunay video-warp on real frames + MediaPipe face landmarker | Lip-sync pipeline exists end-to-end. Whisper has no IPA; visemes come from TTS/Azure/heuristic, not whisper. |
| **Per-peer / per-listener** | Per-peer source language + per-listener single target; router caches per-peer state across engine swaps | Multi-party multi-language is already modelled. |
| **Quality/latency tiers** | `translation_quality: fast (~150 ms) | balanced | hq` in `config.yml` | Latency budget is already a first-class config knob. |
| **Capability routing** | Partial — quality tiers + offline/cloud engine selection at runtime | Server-fallback tiering is partly there; can be formalised later. |

**Bottom line:** This is *not* "build an STS app." It is **"add a whisper.cpp STT
backend to an existing STS app, and decide whether it should displace
sherpa-onnx-Whisper for the Fast tier."** Everything else in v1 is already done.

---

## 1. whisper.cpp's actual role

Today the offline cascade's STT third is **sherpa-onnx Whisper** (offline, non-streaming,
ONNX runtime). whisper.cpp can do better on the **Fast (~150 ms) tier** because it offers:

- **Streaming / partial decode** (emit hypotheses mid-utterance) — sherpa's offline Whisper
  decodes a whole VAD segment before any text appears.
- **GGML quantized models** (`large-v3-turbo-q5_0`) + **multi-backend** CUDA / Vulkan / Metal /
  CoreML / CPU-AVX, selectable per the capability tier.
- **Tight C ABI** already present in this repo ([include/whisper.h](../include/whisper.h)).

**Integration shape:** add a `whisper_cpp_backend` that satisfies the *same seam* as
`OfflineSpeechBackend` (`offline_sherpa_backend.hpp`) — i.e. `accept_audio()` in,
`OfflineUtterance` (source_text / translated_text / pcm16) out — OR expose it as a new
`EngineType::OfflineWhisperCpp` behind the existing factory/router. Either way:

- whisper.cpp owns **STT only** (+ its built-in `translate`→en when target == en).
- For any→any, it feeds the **existing** paths: the `translate_text()` MT seam, or hand the
  STT text to the SeamlessM4T sidecar's text input, or keep DirectSeamless for full text-free S2ST.
- We do **not** fork whisper.cpp internals — track upstream, add a thin streaming wrapper +
  C ABI, keep `sync : ggml` merges clean.

```
ai_conference RoomClient
  └─ TranslationEngineRouter (existing, hot-swap)
       ├─ EngineType::DirectSeamless     (existing)  ── any→any, text-free
       ├─ EngineType::OfflineSherpa      (existing)  ── sherpa Whisper + VITS
       └─ EngineType::OfflineWhisperCpp  (NEW)        ── whisper.cpp STT (streaming)
                                                          → MT seam / Seamless-text → TTS
```

---

## 2. The one real question: replace, or complement, sherpa-Whisper?

| Option | Pros | Cons |
|--------|------|------|
| **A — whisper.cpp as a new engine alongside sherpa** | Non-destructive; A/B benchmark in the live app; falls back to sherpa if it underperforms | Two STT stacks compiled in |
| **B — whisper.cpp replaces sherpa-Whisper inside `OfflineSpeechBackend`** | One STT stack; streaming partials everywhere | Bigger change to a working backend; risk to the shipped OfflineSherpa engine |

**Recommendation:** **A first** — ship `OfflineWhisperCpp` as a parallel engine, benchmark it
against `OfflineSherpa` in the real app on the Fast tier, then collapse to B only if it wins.
This is decided empirically in P0 (below), not on paper.

---

## 3. Locked decisions (planning session)

| # | Decision | Outcome | Note vs. existing app |
|---|----------|---------|------------------------|
| 1 | **Latency target** | **Accept re-baseline:** `<200 ms` local compute, `<350 ms` mouth-to-ear same-region | Matches the app's existing `fast (~150 ms)` tier ambition. |
| 2 | **Voice fidelity** | **Tier 1 (style/voice blend) default + Tier 2 true-clone as GPU-gated opt-in** | App currently uses fixed VITS/Piper + cloud voices; true clone (Kokoro style-latent / CosyVoice2 / XTTS) is future work. |
| 3 | **NMT engine** | **Evaluate both** NLLB-200 (CC-BY-NC ⚠ non-commercial) vs Marian/OPUS-MT in the spike | App *already* has SeamlessM4T (direct any→any) + an MT seam; the eval informs the **cascade** path only. Prefer permissively-licensed Marian for commercial use. |
| 4 | **Fan-out topology** | **Committed: sender-side per-language tracks.** Speaker computes STS once per distinct target language in the room and publishes one audio track (+ phoneme-timing data channel) per language; receivers subscribe to the track for their chosen language. | **Architectural migration (committed):** the app today is **receiver-side** (each listener runs STS on every incoming peer). P3 migrates the mediasoup produce/consume path to sender-side. Receiver-side kept only as a build-time fallback during the migration. |
| 5 | **Transport** | **Resolved — reuse the app's mediasoup stack.** No new transport. | Was an open question in v1; the existing `mediasoup_wrapper` settles it. |

---

## 4. Model selection

| Stage | Model | Format | Notes |
|-------|-------|--------|-------|
| STT (new) | **whisper large-v3-turbo `q5_0`** (Fast: `base`/`small` q5) | GGML | streaming partials, GPU backends |
| STT (incumbent) | sherpa-onnx Whisper | ONNX | current offline STT; benchmark target |
| Direct S2ST | **SeamlessM4T** (existing sidecar) | — | any→any, text-free; already wired |
| NMT (cascade) | Marian/OPUS-MT *(preferred)* or NLLB-200-distilled-600M | CT2/GGML | license: Marian permissive, NLLB **CC-BY-NC** |
| TTS | VITS/Piper *(incumbent)*; Kokoro-82M *(optional add)* | ONNX/GGML | Kokoro brings IPA-timed phonemes for better visemes |
| VAD | Silero (already in app) | ONNX | per-peer gating already implemented |

---

## 5. Phased delivery (re-scoped to the existing app)

| Phase | Goal | Deliverable | Exit criteria |
|-------|------|-------------|---------------|
| **P0 — whisper.cpp STT spike** ✅ | Prove streaming whisper.cpp can hit the Fast tier | `examples/sts-translate/` benchmark CLI (streaming sim + timed MT/TTS seams + batch baseline) | **DONE — go (CPU + GPU).** CPU-only base.en: `audio_ctx≈256` → STT p90 ≈ 125 ms (full quality); `audio_ctx=0` ≈ 924 ms — trimming `audio_ctx` is the CPU latency lever. **GPU (RTX 2000 Ada 8 GB, CUDA 13.3):** base.en at *full* context (`audio_ctx=0`, best quality) → p90 **48 ms**, so `audio_ctx` is no longer a latency lever on GPU; the **accuracy** model `large-v3-turbo-q5_0` is real-time-capable — p90 **86 ms** (`audio_ctx=512`, Fast tier) or 441 ms full-context (Balanced tier). Per-model `audio_ctx` tuning matters: too-small ctx causes decoder repetition that is both worse *and* slower. See `examples/sts-translate/README.md`. |
| **P1 — Streaming STT hardening** 🔨 | Low-latency partials in the app | ring buffer (100–150 ms hop, overlap), hardcoded lang from UI, VAD-gated, token timestamps, C ABI | **Prototype built** — `examples/sts-translate/whisper_stream.{h,cpp}`: C ABI (`push_i16/f32`, `poll_partial/final`, `flush`) with downmix + 48 kHz→16 kHz resample + ring buffer + energy-VAD endpointing (windowed partials at trimmed `audio_ctx`, full-context finals). `--stream` demo drives it from a synthetic 48 kHz stereo stream; partial p90 ≈ 50 ms on base.en/GPU, clean endpointed finals. **TODO:** Silero VAD (vs energy), token timestamps, worker-thread push, A/B vs `OfflineSherpa`, wire into the app engine. |
| **P2 — Cascade MT/TTS choice** | Lock the cascade any→any path | Benchmark Marian vs NLLB on the `translate_text()` seam; optional Kokoro TTS w/ phoneme callback for richer visemes | Source→target speech within budget; license-clean MT chosen |
| **P3 — Fan-out migration** | Sender-side per-language tracks (decision #4, committed) | Migrate mediasoup produce path: speaker computes STS once per distinct target lang, publishes per-language audio track + phoneme-timing data channel; receiver-side kept as build-time fallback during migration | 3-party call, each picks language, sender-side compute, visemes intact |
| **P4 — Voice fidelity Tier 2 (opt-in)** | True per-speaker clone on GPU tier | Kokoro style-latent / CosyVoice2 / XTTS behind capability gate | Opt-in clone passes MOS bar without breaking Fast tier for others |
| **P5 — Capability routing / server fallback** | Weak clients offload | Formalise the join-time benchmark → local vs server engine selection (same binaries server-side) | Weak client transparently offloads STS to server |

(Lip-sync, viseme mapping, MediaPipe, transport, multi-engine router — **already done**; touched
only where P3/P4 change the data they consume.)

---

## 6. Risks

| ID | Risk | Severity | Mitigation |
|----|------|----------|-----------|
| **R1** | Sender-side fan-out (decision #4, committed) is a **migration** from the app's current receiver-side model — non-trivial mediasoup produce/consume rework | **High** | Phased in P3; keep receiver-side as a build-time fallback until sender-side proves out; land it behind a feature flag so a 3-party call can be A/B-tested both ways. |
| R2 | whisper.cpp doesn't beat sherpa-Whisper enough to justify a second STT stack | Med | Option A keeps it non-destructive; decide on real P0 numbers |
| R3 | NLLB CC-BY-NC blocks commercial ship | Med | Prefer Marian/OPUS-MT; NLLB research-baseline only |
| R4 | True voice clone (Tier 2) breaks the latency budget | Med | GPU-tier opt-in only; default stays Tier 1 |
| R5 | `<200 ms` mouth-to-ear expectation resurfaces despite re-baseline | Low | Decision #1 logged; cite §3 |
| R6 | Upstream whisper.cpp/ggml churn | Low | New `examples/` + thin wrapper; don't fork core |

---

## 7. Immediate next step

**P0 spike** — add an `OfflineWhisperCpp` engine that:
1. Implements the `OfflineSpeechBackend` seam (or a sibling) using whisper.cpp streaming,
2. Registers in `TranslationEngineFactory` as `EngineType::OfflineWhisperCpp`,
3. Logs per-stage timings, and
4. A/B-benchmarks against `OfflineSherpa` on real call audio at the Fast tier.

Prototype lives in `examples/sts-translate/` in this repo first (no SDL), then links into the
ai_conference client.

---

## Appendix A — Source repositories & weights
- whisper.cpp: this repo · GGML weights: `huggingface.co/ggerganov/whisper.cpp` → `ggml-large-v3-turbo-q5_0.bin`
- ai_conference app: `…/native/ai_conference` (mediasoup client, MediaPipe, offline_sts_sdk, seamless_sidecar)
- SeamlessM4T sidecar: `seamless_sidecar/server.py`, `server_ws.py`
- sherpa-onnx (incumbent STT/TTS): `github.com/k2-fsa/sherpa-onnx`
- Kokoro (optional TTS): `github.com/hexgrad/kokoro` · NMT: Marian/OPUS-MT *(preferred)* or NLLB-200

## Appendix B — Licensing
Whisper (MIT), Kokoro-82M (Apache-2.0) commercial-safe. **NLLB-200 is CC-BY-NC — non-commercial;
use Marian/OPUS-MT for a commercial product.** Verify SeamlessM4T license terms for commercial use.
