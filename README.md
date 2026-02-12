# dictate-min

Minimal push-to-talk dictation clone (Linux/macOS).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
dictate-min
```

List input devices:

```bash
dictate-min --list-input-devices
```

## Controls

- Hold configured PTT key (default `Right Ctrl`) to record.
- Release to transcribe and paste into the active app.
- `Ctrl+C` to quit.

## Environment variables

### General
- `DICTATE_MODE`: `ptt` or `loopback` (default `ptt`)
- `DICTATE_INPUT_DEVICE`: numeric sounddevice input index (optional)
- `DICTATE_INPUT_DEVICE_NAME`: case-insensitive input-name substring match (used when `DICTATE_INPUT_DEVICE` is unset)
- `DICTATE_SAMPLE_RATE`: input sample rate hint (default `16000`, auto-falls back if unsupported)
- `DICTATE_PASTE`: `1` or `0` (default `1` in `ptt`, `0` in `loopback`)
- `DICTATE_PASTE_MODE`: `clipboard`, `type`, `primary` (default `type` on Linux, `clipboard` otherwise)
  - invalid values fall back to `clipboard`
- `DICTATE_PASTE_PRIMARY_CLICK`: for `primary` mode on Linux, trigger middle-click paste after setting PRIMARY selection (default `1`)
- `DICTATE_PASTE_PRESERVE`: preserve and restore prior clipboard/selection text around paste in `clipboard`/`primary` modes (default `1`)
- `DICTATE_PASTE_RESTORE_DELAY_MS`: delay before restoring clipboard after paste to avoid race with target app (default `80`)
- `DICTATE_DEBUG`: `1` for verbose runtime diagnostics (default `0`)
- `DICTATE_DEBUG_KEYS`: `1` to log every key press/release and whether it matches PTT (default `0`)
- `DICTATE_FILE_LOG`: `1` to append runtime/model/output events to `YYYYMMDD.log` (default `1`)

### Push-to-talk mode
- `DICTATE_PTT_KEY`: `cmd_r`, `super_r`, `cmd_l`, `super_l`, `super`, `win`, `shift_r`, `shift_l`, `ctrl_l`, `ctrl_r`, `alt_l`, `alt_r` (default `ctrl_r`)
- `DICTATE_PTT_AUTO_RESUME_MEDIA`: on PTT release, run `playerctl play` to recover paused playback (Linux, default `1`)

### Loopback mode
- `DICTATE_LOOPBACK_CHUNK_S`: chunk size in seconds (default `4`)
- `DICTATE_LOOPBACK_HINT`: fallback name hint for non-pulse auto-pick (default `loopback pcm`)
- `DICTATE_PULSE_SOURCE`: force pulse source name, e.g. `jamesdsp_sink.monitor`
- `DICTATE_MIN_CHUNK_RMS`: skip near-silent chunks below threshold (default `0.0008`)

### Cleanup model (Ollama)
- `DICTATE_CLEANUP`: enable cleanup pass (`1`/`0`, default `1`)
- `DICTATE_OLLAMA_URL`: cleanup endpoint (default `http://localhost:11434/api/chat`)
- `DICTATE_OLLAMA_MODEL`: cleanup model name (default auto-pick first local model)

### Whisper-specific

#### STT model + decoding
- `DICTATE_STT_MODEL`: faster-whisper model, e.g. `tiny`, `small`, `medium.en` (default `medium.en`)
- `DICTATE_STT_DEVICE`: `cpu`, `auto`, `cuda` (default `auto`)
- `DICTATE_STT_COMPUTE`: e.g. `int8`, `float16` (default auto: `float16` for `auto/cuda`, `int8` for `cpu`)
- `DICTATE_STT_CONDITION_PREV`: maps to `condition_on_previous_text` (default `0`)
- `DICTATE_STT_BEAM_SIZE`: beam size (default `5`)
- `DICTATE_STT_NO_SPEECH_THRESHOLD`: no-speech threshold (default `0.6`)
- `DICTATE_STT_LOGPROB_THRESHOLD`: log-prob threshold (default `-1.0`)
- `DICTATE_STT_COMPRESSION_RATIO_THRESHOLD`: compression-ratio threshold (default `2.4`)
- `DICTATE_INPUT_LANGUAGE`: `auto` or language code (default `auto`)

#### Context + overlap
- `DICTATE_CONTEXT`: enable text context carryover (`1`/`0`, default `1`)
- `DICTATE_CONTEXT_CHARS`: max retained text context chars (default `600`)
- `DICTATE_CONTEXT_RESET_EVERY`: reset context every N emitted chunks (`0` disables, default `0`)
- `DICTATE_AUDIO_CONTEXT_S`: prepended previous-audio seconds per chunk (default `1.6`)
- `DICTATE_AUDIO_CONTEXT_PAD_S`: overlap pad used for timestamp clipping (default `0.12`)
- `DICTATE_TRIM_CHUNK_PERIOD`: trim trailing `.`/`...` from chunk output (default `1`)

#### Loop/failure protection
- `DICTATE_LOOP_GUARD`: enable pathological-loop detection + context reset (default `1`)
- `DICTATE_LOOP_GUARD_REPEAT_RATIO`: repetition trigger ratio (default `0.55`)
- `DICTATE_LOOP_GUARD_PUNCT_RATIO`: punctuation-density trigger ratio (default `0.35`)
- `DICTATE_LOOP_GUARD_SHORT_RUN`: repeated short-token run trigger length (default `4`)
- `DICTATE_LOOP_GUARD_SHORT_LEN`: max token length considered “short” for run detection (default `3`)

## Notes

- STT is `faster-whisper` (works on Linux; no MLX required).
- Cleanup uses Ollama chat API.
- Your Ollama build does not expose `/api/transcribe`, so Ollama is used for cleanup only in this version.
- If CUDA init fails, set `DICTATE_STT_DEVICE=cpu` explicitly for stable fallback.

## Linux speaker-output test (no hotkey)

Use loopback mode to transcribe whatever is playing on a loopback/monitor input device:

```bash
source .venv/bin/activate
DICTATE_MODE=loopback dictate-min
```

If auto-pick chooses the wrong source, pin it:

```bash
DICTATE_MODE=loopback DICTATE_INPUT_DEVICE=25 dictate-min
```

Or steer auto-pick by name:

```bash
DICTATE_MODE=loopback DICTATE_LOOPBACK_HINT="loopback pcm" dictate-min
```

On PipeWire/Pulse systems, loopback mode first tries:
- `PULSE_SOURCE=<Active Sink>.monitor` (derived from current sink inputs)
- fallback: `<Default Sink>.monitor`
- with the `pulse` input device

This makes capture follow real playback routing (e.g. JamesDSP) instead of your default input source.

Overlap handling:
- Uses `word_timestamps=True` and drops words that fall inside the prepended audio-context window.
- Adds cross-chunk prefix/suffix dedup on emitted words.

Where transcription goes:
- Always printed to stdout as plain text chunks (space-separated, no `TRANSCRIPT:` prefix)
- Also pasted only when `DICTATE_PASTE=1`
- Daily file log in current working directory: `YYYYMMDD.log`

## Examples

### 1) Basic PTT (type into focused app)
```bash
dictate-min
```

### 2) Loopback transcription only (no paste), debug on
```bash
DICTATE_MODE=loopback DICTATE_DEBUG=1 DICTATE_CLEANUP=0 dictate-min
```

### 3) Force capture from JamesDSP monitor
```bash
DICTATE_MODE=loopback DICTATE_PULSE_SOURCE=jamesdsp_sink.monitor dictate-min
```

### 3b) Select webcam mic by name
```bash
DICTATE_MODE=ptt DICTATE_INPUT_DEVICE_NAME=HTI-UC320 dictate-min
```

### 4) Fast/low-latency test setup
```bash
DICTATE_MODE=loopback \
DICTATE_STT_MODEL=tiny \
DICTATE_LOOPBACK_CHUNK_S=3 \
DICTATE_STT_BEAM_SIZE=1 \
DICTATE_CLEANUP=0 \
dictate-min
```

### 5) Higher quality setup
```bash
DICTATE_MODE=loopback \
DICTATE_STT_MODEL=medium.en \
DICTATE_STT_BEAM_SIZE=5 \
DICTATE_INPUT_LANGUAGE=en \
dictate-min
```

### 6) Aggressive loop protection
```bash
DICTATE_MODE=loopback \
DICTATE_LOOP_GUARD=1 \
DICTATE_LOOP_GUARD_REPEAT_RATIO=0.40 \
DICTATE_LOOP_GUARD_PUNCT_RATIO=0.20 \
DICTATE_LOOP_GUARD_SHORT_RUN=3 \
DICTATE_CONTEXT_RESET_EVERY=6 \
dictate-min
```

### 7) Disable chunk-final period trimming
```bash
DICTATE_TRIM_CHUNK_PERIOD=0 dictate-min
```

### 8) Avoid clipboard, type directly
```bash
DICTATE_PASTE_MODE=type dictate-min
```

### 9) Linux PRIMARY selection + middle-click paste
```bash
DICTATE_PASTE_MODE=primary DICTATE_PASTE_PRIMARY_CLICK=1 dictate-min
```

### 10) Disable clipboard preservation
```bash
DICTATE_PASTE_PRESERVE=0 dictate-min
```

Clipboard preservation note:
- Preservation is best-effort for text clipboard content. Non-text clipboard payloads (e.g. images/custom mime types) may not round-trip through text clipboard APIs.
