#!/usr/bin/env bash
# Run the sts-translate spike examples (bash / MSYS / Linux / macOS).
#
#   sh examples/sts-translate/run-examples.sh           # list examples
#   sh examples/sts-translate/run-examples.sh all       # run them all
#   sh examples/sts-translate/run-examples.sh fast      # run one
#   sh examples/sts-translate/run-examples.sh legend    # just the abbreviations
#
# Env overrides: CUDA=<toolkit dir>  WAV=<input wav>  EXE=<exe path>  OUT=<output dir>
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WAV="${WAV:-$ROOT/samples/jfk.wav}"                     # INPUT audio (16 kHz mono WAV)
OUT="${OUT:-$ROOT/examples/sts-translate/out}"          # OUTPUT dir (one .txt per example)
BASE="$ROOT/models/ggml-base.en.bin"
LARGE="$ROOT/models/ggml-large-v3-turbo-q5_0.bin"
CUDA="${CUDA:-/c/PROGRA~1/NVIDIA~2/CUDA/v13.3}"
mkdir -p "$OUT"

# Locate the binary: prefer the CUDA build, fall back to the CPU build.
EXE="${EXE:-}"
if [ -z "$EXE" ]; then
  for c in "$ROOT/build-cu13/bin/Release/whisper-sts-translate.exe" \
           "$ROOT/build/bin/Release/whisper-sts-translate.exe" \
           "$ROOT/build/bin/whisper-sts-translate"; do
    [ -x "$c" ] && EXE="$c" && break
  done
fi
[ -n "$EXE" ] && [ -x "$EXE" ] || { echo "ERROR: benchmark binary not found. Build it first:" >&2
  echo "  cmake --build build-cu13 --target whisper-sts-translate --config Release" >&2; exit 1; }

# CUDA runtime DLLs live in bin/x64 on CUDA >= 13; also add the exe dir (MSYS needs it).
[ -d "$CUDA/bin/x64" ] && export PATH="$CUDA/bin/x64:$CUDA/bin:$PATH"
EXE_DIR="$(dirname "$EXE")"; EXE_NAME="$(basename "$EXE")"
export PATH="$EXE_DIR:$PATH"

GEOM="--step 500 --length 1500"

run() { # $1 label, $2 model, rest = extra args
  local label="$1" model="$2"; shift 2
  if [ ! -f "$model" ]; then printf '\n>> skip [%s] - model not found: %s\n' "$label" "$model"; return; fi
  local outfile="$OUT/$label.txt"
  printf '\n==================================================================\n'
  printf '>> [%s]  %s  %s\n' "$label" "$(basename "$model")" "$*"
  printf '==================================================================\n'
  ( cd "$EXE_DIR" && "./$EXE_NAME" -m "$model" -f "$WAV" $GEOM "$@" ) 2>&1 | tee "$outfile"
  printf '\nINPUT  : %s\nOUTPUT : %s\n' "$WAV" "$outfile"
}

ex_full()       { run full       "$BASE"  -l en --audio-ctx 0;            }   # best quality, no trimming
ex_fast()       { run fast       "$BASE"  -l en --audio-ctx 256;          }   # CPU Fast-tier sweet spot
ex_accuracy()   { run accuracy   "$LARGE" -l en --audio-ctx 512;          }   # large-v3-turbo, clean
ex_aggressive() { run aggressive "$BASE"  -l en --audio-ctx 32;           }   # too small -> degraded
ex_sts() {                                                                       # STT->MT->TTS pipeline; renders sts.wav via SAPI
  local wav="$OUT/sts.wav"
  command -v cygpath >/dev/null 2>&1 && wav="$(cygpath -w "$wav")"              # SAPI needs a native Windows path
  run sts "$BASE" -l en -tl es --audio-ctx 0 --tts-out "$wav"
}
ex_translate()  { run translate  "$BASE"  -l en --translate --audio-ctx 0;}   # whisper X->en task
ex_batch()      { run batch      "$BASE"  -l en --batch;                  }   # one-shot baseline, no streaming
ex_stream()     { run stream     "$BASE"  -l en --stream --src-rate 48000 --step 200 --audio-ctx 0; } # P1 live-stream wrapper

legend() {
  cat <<EOF

------------------------------------------------------------------
Abbreviations
  STT        Speech-to-Text  - whisper transcription stage (real)
  MT         Machine Translation - source->target text (P0 stub, passthrough)
  TTS        Text-to-Speech  - target speech synthesis (P0 stub, no-op)
  RTF        Real-Time Factor = compute time / audio duration (<1 = faster than real time)
  p50 / p90  50th / 90th percentile per-chunk latency, in milliseconds
  ac         audio_ctx = encoder context length in positions (1500 = full 30 s; ~20 ms each)
  Fast tier  the app's ~150 ms per-chunk latency budget
  en / es    ISO language codes (English / Spanish)
Flags
  -m  GGML model file          -f  input WAV (16 kHz mono)
  -l  source language          -tl target language (MT seam)
  --length window size (ms)    --step hop between windows (ms)
  --audio-ctx N  trim encoder context   --translate  whisper X->English
  --batch one-shot decode (no streaming)   --no-gpu  force CPU
Paths
  INPUT : $WAV
  OUTPUT: $OUT/<example>.txt
EOF
}

list() {
  cat <<'EOF'
sts-translate examples (pass a name, "all", or "legend"):
  full         base.en   audio_ctx=0    best quality, no trimming
  fast         base.en   audio_ctx=256  CPU Fast-tier sweet spot
  accuracy     large-v3-turbo  ac=512   accuracy model, clean & under budget
  aggressive   base.en   audio_ctx=32   deliberately too small -> degraded text
  sts          base.en   en->es         full STT->MT->TTS pipeline (MT/TTS are P0 stubs)
  translate    base.en   --translate    whisper X->English task
  batch        base.en   --batch        one-shot baseline (no streaming)
  stream       base.en   --stream       P1: live 48 kHz stream -> resample+VAD -> partials+finals
EOF
}

case "${1:-list}" in
  list|"") list ;;
  legend)  legend ;;
  all) for e in full fast accuracy aggressive sts translate batch stream; do ex_$e; done; legend ;;
  full|fast|accuracy|aggressive|sts|translate|batch|stream) ex_"$1"; legend ;;
  *) echo "unknown example: $1"; echo; list; exit 1 ;;
esac
