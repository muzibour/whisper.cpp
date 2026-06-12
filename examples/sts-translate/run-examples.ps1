<#
  Run the sts-translate spike examples (PowerShell / pwsh).

    pwsh examples/sts-translate/run-examples.ps1            # list examples
    pwsh examples/sts-translate/run-examples.ps1 all        # run them all
    pwsh examples/sts-translate/run-examples.ps1 fast       # run one
    pwsh examples/sts-translate/run-examples.ps1 legend     # just the abbreviations

  Params: -Cuda <toolkit dir>  -Wav <input wav>  -Exe <exe path>  -Out <output dir>
#>
param(
    [string]$What = "list",
    [string]$Cuda = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3",
    [string]$Wav,
    [string]$Exe,
    [string]$Out
)
$ErrorActionPreference = "Stop"

$Root  = (Resolve-Path "$PSScriptRoot\..\..").Path
if (-not $Wav) { $Wav = Join-Path $Root "samples\jfk.wav" }          # INPUT audio (16 kHz mono WAV)
if (-not $Out) { $Out = Join-Path $Root "examples\sts-translate\out" } # OUTPUT dir (one .txt per example)
$Base  = Join-Path $Root "models\ggml-base.en.bin"
$Large = Join-Path $Root "models\ggml-large-v3-turbo-q5_0.bin"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

# Locate the binary: prefer the CUDA build, fall back to the CPU build.
if (-not $Exe) {
    foreach ($c in @(
        "$Root\build-cu13\bin\Release\whisper-sts-translate.exe",
        "$Root\build\bin\Release\whisper-sts-translate.exe")) {
        if (Test-Path $c) { $Exe = $c; break }
    }
}
if (-not $Exe -or -not (Test-Path $Exe)) {
    Write-Error "benchmark binary not found. Build it first: cmake --build build-cu13 --target whisper-sts-translate --config Release"
}

# CUDA runtime DLLs live in bin\x64 on CUDA >= 13. The exe finds its sibling DLLs itself.
if (Test-Path "$Cuda\bin\x64") { $env:Path = "$Cuda\bin\x64;$Cuda\bin;$env:Path" }

# whisper logs to stderr; under "Stop" PowerShell would treat that as a terminating
# error. Setup validation above is done, so relax to "Continue" before running the exe.
$ErrorActionPreference = "Continue"

$Geom = @("--step","500","--length","1500")

function Run-Example {
    param([string]$Label, [string]$Model, [string[]]$Extra)
    if (-not (Test-Path $Model)) { Write-Host "`n>> skip [$Label] - model not found: $Model"; return }
    $OutFile = Join-Path $Out "$Label.txt"
    Write-Host "`n=================================================================="
    Write-Host ">> [$Label]  $(Split-Path $Model -Leaf)  $($Extra -join ' ')"
    Write-Host "=================================================================="
    & $Exe -m $Model -f $Wav @Geom @Extra *>&1 | Tee-Object -FilePath $OutFile
    Write-Host "`nINPUT  : $Wav"
    Write-Host "OUTPUT : $OutFile"
}

$Examples = [ordered]@{
    full       = { Run-Example full       $Base  @("-l","en","--audio-ctx","0") }
    fast       = { Run-Example fast       $Base  @("-l","en","--audio-ctx","256") }
    accuracy   = { Run-Example accuracy   $Large @("-l","en","--audio-ctx","512") }
    aggressive = { Run-Example aggressive $Base  @("-l","en","--audio-ctx","32") }
    sts        = { Run-Example sts        $Base  @("-l","en","-tl","es","--audio-ctx","0","--tts-out","$Out\sts.wav") }
    translate  = { Run-Example translate  $Base  @("-l","en","--translate","--audio-ctx","0") }
    batch      = { Run-Example batch      $Base  @("-l","en","--batch") }
    stream     = { Run-Example stream     $Base  @("-l","en","--stream","--src-rate","48000","--step","200","--audio-ctx","0") }
}

function Show-Legend {
    Write-Host ""
    Write-Host "------------------------------------------------------------------"
    Write-Host "Abbreviations"
    Write-Host "  STT        Speech-to-Text  - whisper transcription stage (real)"
    Write-Host "  MT         Machine Translation - source->target text (P0 stub, passthrough)"
    Write-Host "  TTS        Text-to-Speech  - target speech synthesis (P0 stub, no-op)"
    Write-Host "  RTF        Real-Time Factor = compute time / audio duration (<1 = faster than real time)"
    Write-Host "  p50 / p90  50th / 90th percentile per-chunk latency, in milliseconds"
    Write-Host "  ac         audio_ctx = encoder context length in positions (1500 = full 30 s; ~20 ms each)"
    Write-Host "  Fast tier  the app's ~150 ms per-chunk latency budget"
    Write-Host "  en / es    ISO language codes (English / Spanish)"
    Write-Host "Flags"
    Write-Host "  -m  GGML model file          -f  input WAV (16 kHz mono)"
    Write-Host "  -l  source language          -tl target language (MT seam)"
    Write-Host "  --length window size (ms)    --step hop between windows (ms)"
    Write-Host "  --audio-ctx N  trim encoder context   --translate  whisper X->English"
    Write-Host "  --batch one-shot decode (no streaming)   --no-gpu  force CPU"
    Write-Host "Paths"
    Write-Host "  INPUT : $Wav"
    Write-Host "  OUTPUT: $Out\<example>.txt"
}

function Show-List {
    Write-Host "sts-translate examples (pass a name, 'all', or 'legend'):"
    Write-Host "  full         base.en   audio_ctx=0    best quality, no trimming"
    Write-Host "  fast         base.en   audio_ctx=256  CPU Fast-tier sweet spot"
    Write-Host "  accuracy     large-v3-turbo  ac=512   accuracy model, clean & under budget"
    Write-Host "  aggressive   base.en   audio_ctx=32   deliberately too small -> degraded text"
    Write-Host "  sts          base.en   en->es         full STT->MT->TTS pipeline (MT/TTS are P0 stubs)"
    Write-Host "  translate    base.en   --translate    whisper X->English task"
    Write-Host "  batch        base.en   --batch        one-shot baseline (no streaming)"
    Write-Host "  stream       base.en   --stream       P1: live 48 kHz stream -> resample+VAD -> partials+finals"
}

switch ($What) {
    "list"   { Show-List }
    "legend" { Show-Legend }
    "all"    { foreach ($k in $Examples.Keys) { & $Examples[$k] }; Show-Legend }
    default  {
        if ($Examples.Contains($What)) { & $Examples[$What]; Show-Legend }
        else { Write-Host "unknown example: $What`n"; Show-List; exit 1 }
    }
}
