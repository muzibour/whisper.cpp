#!/usr/bin/env bash
# Re-run the P0 streaming benchmark and print README-ready markdown table rows.
#
# Usage (from repo root, after a -DGGML_CUDA=ON build in build-cu13):
#   sh examples/sts-translate/bench-gpu.sh
#
# Override defaults via env:
#   BIN=...   path to whisper-sts-translate.exe
#   WAV=...   16 kHz mono wav (default samples/jfk.wav)
#   CUDA=...  CUDA toolkit dir (its bin/x64 holds cudart/cublas DLLs on CUDA >=13)
#   CTXS="0 256 512"   audio_ctx values to sweep
# Models swept: base.en (Fast) + large-v3-turbo-q5_0 (accuracy), if present.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="${BIN:-$ROOT/build-cu13/bin/Release/whisper-sts-translate.exe}"
WAV="${WAV:-$ROOT/samples/jfk.wav}"
CUDA="${CUDA:-/c/PROGRA~1/NVIDIA~2/CUDA/v13.3}"
CTXS="${CTXS:-0 256 512}"
STEP="${STEP:-500}"; LENGTH="${LENGTH:-1500}"

[ -x "$BIN" ] || { echo "ERROR: benchmark binary not found: $BIN" >&2
  echo "Build it first: cmake --build build-cu13 --target whisper-sts-translate --config Release" >&2; exit 1; }
[ -f "$WAV" ] || { echo "ERROR: wav not found: $WAV" >&2; exit 1; }

# CUDA runtime DLLs live in bin/x64 on CUDA >= 13; exe dir holds ggml/whisper DLLs.
export PATH="$CUDA/bin/x64:$CUDA/bin:$(dirname "$BIN"):$PATH"

sweep() { # $1 = model path, $2 = label
  local model="$1" label="$2"
  [ -f "$model" ] || { echo "  (skip $label — model not found: $model)"; return; }
  echo "| model | audio_ctx | STT p50 | STT p90 | STT max |"
  echo "|-------|----------:|--------:|--------:|--------:|"
  for ac in $CTXS; do
    line=$("$BIN" -m "$model" -f "$WAV" -l en --step "$STEP" --length "$LENGTH" --audio-ctx "$ac" 2>/dev/null \
           | tr -d '\r' | grep -E '^  STT ')
    p50=$(echo "$line" | sed -n 's/.*p50= *\([0-9.]*\).*/\1/p')
    p90=$(echo "$line" | sed -n 's/.*p90= *\([0-9.]*\).*/\1/p')
    max=$(echo "$line" | sed -n 's/.*max= *\([0-9.]*\).*/\1/p')
    printf "| %s | %s | %s | %s | %s |\n" "$label" "$ac" "${p50:-?}" "${p90:-?}" "${max:-?}"
  done
  echo
}

echo "## Benchmark: window=${LENGTH}ms hop=${STEP}ms  $(date +%Y-%m-%d)"
echo
sweep "$ROOT/models/ggml-base.en.bin"             "base.en"
sweep "$ROOT/models/ggml-large-v3-turbo-q5_0.bin" "large-v3-turbo-q5_0"
