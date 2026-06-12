@echo off
REM Run the sts-translate spike examples (Windows cmd.exe).
REM
REM   examples\sts-translate\run-examples.bat           list examples
REM   examples\sts-translate\run-examples.bat all       run them all
REM   examples\sts-translate\run-examples.bat fast      run one
REM   examples\sts-translate\run-examples.bat legend    just the abbreviations
REM
REM Override the CUDA dir by setting CUDA before calling.
setlocal EnableExtensions

for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
if not defined CUDA set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "WAV=%ROOT%\samples\jfk.wav"
set "OUT=%ROOT%\examples\sts-translate\out"
set "BASE=%ROOT%\models\ggml-base.en.bin"
set "LARGE=%ROOT%\models\ggml-large-v3-turbo-q5_0.bin"
if not exist "%OUT%" mkdir "%OUT%"

REM Locate the binary: prefer the CUDA build, fall back to the CPU build.
set "EXE=%ROOT%\build-cu13\bin\Release\whisper-sts-translate.exe"
if not exist "%EXE%" set "EXE=%ROOT%\build\bin\Release\whisper-sts-translate.exe"
if not exist "%EXE%" (
    echo ERROR: benchmark binary not found. Build it first:
    echo   cmake --build build-cu13 --target whisper-sts-translate --config Release
    exit /b 1
)

REM CUDA runtime DLLs live in bin\x64 on CUDA ^>= 13; the exe finds its sibling DLLs itself.
if exist "%CUDA%\bin\x64" set "PATH=%CUDA%\bin\x64;%CUDA%\bin;%PATH%"

set "GEOM=--step 500 --length 1500"

set "WHAT=%~1"
if "%WHAT%"=="" set "WHAT=list"

if /i "%WHAT%"=="list"       goto :list
if /i "%WHAT%"=="legend"     ( call :legend & goto :end )
if /i "%WHAT%"=="all"        goto :all
if /i "%WHAT%"=="full"       ( call :ex_full       & call :legend & goto :end )
if /i "%WHAT%"=="fast"       ( call :ex_fast       & call :legend & goto :end )
if /i "%WHAT%"=="accuracy"   ( call :ex_accuracy   & call :legend & goto :end )
if /i "%WHAT%"=="aggressive" ( call :ex_aggressive & call :legend & goto :end )
if /i "%WHAT%"=="sts"        ( call :ex_sts        & call :legend & goto :end )
if /i "%WHAT%"=="translate"  ( call :ex_translate  & call :legend & goto :end )
if /i "%WHAT%"=="batch"      ( call :ex_batch      & call :legend & goto :end )
if /i "%WHAT%"=="stream"     ( call :ex_stream     & call :legend & goto :end )
echo unknown example: %WHAT%
echo.
goto :list

:all
call :ex_full
call :ex_fast
call :ex_accuracy
call :ex_aggressive
call :ex_sts
call :ex_translate
call :ex_batch
call :ex_stream
call :legend
goto :end

REM --- :run  label  model  "extra args"  "description" -----------------------
:run
set "RLBL=%~1"
set "RMODEL=%~2"
set "REXTRA=%~3"
set "RDESC=%~4"
if not exist "%RMODEL%" ( echo. & echo ^>^> skip [%RLBL%] - model not found: %RMODEL% & goto :eof )
echo.
echo ==================================================================
echo ^>^> [%RLBL%]  %RDESC%
echo ==================================================================
set "ROUT=%OUT%\%RLBL%.txt"
"%EXE%" -m "%RMODEL%" -f "%WAV%" %GEOM% %REXTRA% > "%ROUT%" 2>&1
type "%ROUT%"
echo.
echo INPUT  : %WAV%
echo OUTPUT : %ROUT%
goto :eof

:ex_full
call :run "full"       "%BASE%"  "-l en --audio-ctx 0"             "audio_ctx=0 best quality"
goto :eof
:ex_fast
call :run "fast"       "%BASE%"  "-l en --audio-ctx 256"           "audio_ctx=256 CPU sweet spot"
goto :eof
:ex_accuracy
call :run "accuracy"   "%LARGE%" "-l en --audio-ctx 512"           "large-v3-turbo audio_ctx=512 clean"
goto :eof
:ex_aggressive
call :run "aggressive" "%BASE%"  "-l en --audio-ctx 32"            "audio_ctx=32 too small -^> degraded"
goto :eof
:ex_sts
call :run "sts"        "%BASE%"  "-l en -tl es --audio-ctx 0 --tts-out %OUT%\sts.wav" "en-^>es pipeline; renders sts.wav"
goto :eof
:ex_translate
call :run "translate"  "%BASE%"  "-l en --translate --audio-ctx 0" "whisper X-^>en task"
goto :eof
:ex_batch
call :run "batch"      "%BASE%"  "-l en --batch"                   "one-shot baseline (no streaming)"
goto :eof
:ex_stream
call :run "stream"     "%BASE%"  "-l en --stream --src-rate 48000 --step 200 --audio-ctx 0" "P1 live-stream wrapper"
goto :eof

:legend
echo.
echo ------------------------------------------------------------------
echo Abbreviations
echo   STT        Speech-to-Text  - whisper transcription stage (real)
echo   MT         Machine Translation - source-^>target text (P0 stub, passthrough)
echo   TTS        Text-to-Speech  - target speech synthesis (P0 stub, no-op)
echo   RTF        Real-Time Factor = compute time / audio duration (^<1 = faster than real time)
echo   p50 / p90  50th / 90th percentile per-chunk latency, in milliseconds
echo   ac         audio_ctx = encoder context length in positions (1500 = full 30 s; ~20 ms each)
echo   Fast tier  the app's ~150 ms per-chunk latency budget
echo   en / es    ISO language codes (English / Spanish)
echo Flags
echo   -m  GGML model file          -f  input WAV (16 kHz mono)
echo   -l  source language          -tl target language (MT seam)
echo   --length window size (ms)    --step hop between windows (ms)
echo   --audio-ctx N  trim encoder context   --translate  whisper X-^>English
echo   --batch one-shot decode (no streaming)   --no-gpu  force CPU
echo Paths
echo   INPUT : %WAV%
echo   OUTPUT: %OUT%\^<example^>.txt
goto :eof

:list
echo sts-translate examples (pass a name, "all", or "legend"):
echo   full         base.en   audio_ctx=0    best quality, no trimming
echo   fast         base.en   audio_ctx=256  CPU Fast-tier sweet spot
echo   accuracy     large-v3-turbo  ac=512   accuracy model, clean ^& under budget
echo   aggressive   base.en   audio_ctx=32   deliberately too small -^> degraded text
echo   sts          base.en   en-^>es         full STT-^>MT-^>TTS pipeline (MT/TTS are P0 stubs)
echo   translate    base.en   --translate    whisper X-^>English task
echo   batch        base.en   --batch        one-shot baseline (no streaming)
echo   stream       base.en   --stream       P1: live 48 kHz stream -^> resample+VAD -^> partials+finals
goto :end

:end
endlocal
