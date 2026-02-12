from __future__ import annotations

import contextlib
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pyperclip
import requests
import sounddevice as sd
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button as MouseButton
from pynput.mouse import Controller as MouseController
from scipy.signal import resample_poly
from scipy.io.wavfile import write as wav_write


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class Config:
    sample_rate: int = env_int("DICTATE_SAMPLE_RATE", 16_000)
    channels: int = 1
    block_ms: int = 30
    device_id: int | None = None
    input_device_name: str = os.environ.get("DICTATE_INPUT_DEVICE_NAME", "")
    stt_model: str = os.environ.get("DICTATE_STT_MODEL", "medium.en")
    stt_device: str = os.environ.get("DICTATE_STT_DEVICE", "auto")
    stt_compute_type: str = os.environ.get("DICTATE_STT_COMPUTE", "")
    stt_condition_on_previous_text: bool = env_bool("DICTATE_STT_CONDITION_PREV", False)
    stt_beam_size: int = env_int("DICTATE_STT_BEAM_SIZE", 5)
    stt_no_speech_threshold: float = env_float("DICTATE_STT_NO_SPEECH_THRESHOLD", 0.6)
    stt_log_prob_threshold: float = env_float("DICTATE_STT_LOGPROB_THRESHOLD", -1.0)
    stt_compression_ratio_threshold: float = env_float("DICTATE_STT_COMPRESSION_RATIO_THRESHOLD", 2.4)
    input_language: str = os.environ.get("DICTATE_INPUT_LANGUAGE", "auto")
    cleanup: bool = env_bool("DICTATE_CLEANUP", True)
    ollama_url: str = os.environ.get("DICTATE_OLLAMA_URL", "http://localhost:11434/api/chat")
    ollama_model: str = os.environ.get("DICTATE_OLLAMA_MODEL", "")
    mode: str = os.environ.get("DICTATE_MODE", "ptt").lower()
    loopback_chunk_s: int = env_int("DICTATE_LOOPBACK_CHUNK_S", 4)
    loopback_hint: str = os.environ.get("DICTATE_LOOPBACK_HINT", "loopback pcm").lower()
    paste: bool = env_bool("DICTATE_PASTE", True)
    paste_mode: str = os.environ.get(
        "DICTATE_PASTE_MODE", "type" if platform.system() == "Linux" else "clipboard"
    ).lower()
    paste_primary_click: bool = env_bool("DICTATE_PASTE_PRIMARY_CLICK", True)
    paste_preserve: bool = env_bool("DICTATE_PASTE_PRESERVE", True)
    paste_restore_delay_ms: int = env_int("DICTATE_PASTE_RESTORE_DELAY_MS", 80)
    debug: bool = env_bool("DICTATE_DEBUG", False)
    debug_keys: bool = env_bool("DICTATE_DEBUG_KEYS", False)
    min_chunk_rms: float = env_float("DICTATE_MIN_CHUNK_RMS", 0.0008)
    context_enabled: bool = env_bool("DICTATE_CONTEXT", True)
    context_chars: int = env_int("DICTATE_CONTEXT_CHARS", 600)
    context_reset_every: int = env_int("DICTATE_CONTEXT_RESET_EVERY", 0)
    audio_context_s: float = env_float("DICTATE_AUDIO_CONTEXT_S", 1.6)
    audio_context_pad_s: float = env_float("DICTATE_AUDIO_CONTEXT_PAD_S", 0.12)
    trim_chunk_terminal_period: bool = env_bool("DICTATE_TRIM_CHUNK_PERIOD", True)
    ptt_auto_resume_media: bool = env_bool("DICTATE_PTT_AUTO_RESUME_MEDIA", True)
    loop_guard_enabled: bool = env_bool("DICTATE_LOOP_GUARD", True)
    loop_guard_max_repeat_ratio: float = env_float("DICTATE_LOOP_GUARD_REPEAT_RATIO", 0.55)
    loop_guard_max_punct_ratio: float = env_float("DICTATE_LOOP_GUARD_PUNCT_RATIO", 0.35)
    loop_guard_short_run: int = env_int("DICTATE_LOOP_GUARD_SHORT_RUN", 4)
    loop_guard_short_len: int = env_int("DICTATE_LOOP_GUARD_SHORT_LEN", 3)
    file_log_enabled: bool = env_bool("DICTATE_FILE_LOG", True)
    ptt_key: str = os.environ.get("DICTATE_PTT_KEY", "ctrl_r")

    @property
    def block_size(self) -> int:
        return int(self.sample_rate * (self.block_ms / 1000.0))


class Recorder:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.stream: sd.InputStream | None = None
        self.chunks: list[np.ndarray] = []
        self.recording = False
        self.lock = threading.Lock()

    def start(self) -> None:
        with self.lock:
            if self.recording:
                return
            self.chunks.clear()
            self.recording = True

        try:
            self.stream = sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                dtype="float32",
                blocksize=self.cfg.block_size,
                device=self.cfg.device_id,
                callback=self._on_audio,
            )
        except sd.PortAudioError as e:
            # Loopback/monitor sources often only support native rates (e.g. 48000).
            msg = str(e).lower()
            if "invalid sample rate" not in msg:
                raise
            info = sd.query_devices(self.cfg.device_id)
            native = int(float(info.get("default_samplerate", self.cfg.sample_rate)))
            print(f"audio: sample_rate {self.cfg.sample_rate} unsupported on device {self.cfg.device_id}; using {native}")
            self.cfg.sample_rate = native
            self.stream = sd.InputStream(
                samplerate=self.cfg.sample_rate,
                channels=self.cfg.channels,
                dtype="float32",
                blocksize=self.cfg.block_size,
                device=self.cfg.device_id,
                callback=self._on_audio,
            )
        self.stream.start()

    def stop(self) -> np.ndarray:
        with self.lock:
            if not self.recording:
                return np.zeros((0,), dtype=np.int16)
            self.recording = False

        if self.stream is not None:
            with contextlib.suppress(Exception):
                self.stream.stop()
            with contextlib.suppress(Exception):
                self.stream.close()
            self.stream = None

        if not self.chunks:
            return np.zeros((0,), dtype=np.int16)

        audio = np.concatenate(self.chunks).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767.0).astype(np.int16)

    def _on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status:
            return
        with self.lock:
            if not self.recording:
                return
            self.chunks.append(indata[:, 0].copy())

    def take_chunk(self) -> np.ndarray:
        with self.lock:
            if not self.chunks:
                return np.zeros((0,), dtype=np.int16)
            audio = np.concatenate(self.chunks).astype(np.float32)
            self.chunks.clear()
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767.0).astype(np.int16)


class Pipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._kb = KeyboardController()
        self._has_output = False
        self._stt_model = self._build_stt()
        self._paste_mod = keyboard.Key.cmd if platform.system() == "Darwin" else keyboard.Key.ctrl
        self._mouse = MouseController()
        self._context_text = ""
        self._prev_audio_tail: np.ndarray | None = None
        self._last_emitted_words: list[str] = []
        if self.cfg.cleanup and not self.cfg.ollama_model:
            self.cfg.ollama_model = self._default_ollama_model()

    def reset_context(self) -> None:
        self._context_text = ""
        self._prev_audio_tail = None
        self._last_emitted_words = []

    def _build_stt(self):
        from faster_whisper import WhisperModel

        device = self.cfg.stt_device
        compute_type = self.cfg.stt_compute_type
        started = time.time()
        print(
            f"loading stt model '{self.cfg.stt_model}' "
            f"(device={device}, compute={compute_type})...",
            flush=True,
        )
        try:
            model = WhisperModel(self.cfg.stt_model, device=device, compute_type=compute_type)
            print(f"stt model ready in {time.time() - started:.1f}s (device={device})", flush=True)
            return model
        except RuntimeError as e:
            msg = str(e).lower()
            if device == "auto" and compute_type == "float16" and ("float16" in msg or "compute type" in msg):
                print("stt init: float16 unsupported on auto-selected backend, retrying int8")
                retry_started = time.time()
                model = WhisperModel(self.cfg.stt_model, device="auto", compute_type="int8")
                print(
                    f"stt model ready in {time.time() - retry_started:.1f}s (device=auto, compute=int8)",
                    flush=True,
                )
                return model
            if "cublas" in msg or "cuda" in msg:
                print("stt init: cuda unavailable, falling back to cpu")
                fallback_started = time.time()
                model = WhisperModel(self.cfg.stt_model, device="cpu", compute_type="int8")
                print(
                    f"stt model ready in {time.time() - fallback_started:.1f}s (device=cpu fallback)",
                    flush=True,
                )
                return model
            raise

    def _default_ollama_model(self) -> str:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            if models:
                return str(models[0].get("name", ""))
        except Exception:
            pass
        return ""

    @staticmethod
    def _is_cuda_runtime_failure(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return ("libcublas" in msg) or ("cuda" in msg)

    def _switch_stt_to_cpu(self) -> None:
        from faster_whisper import WhisperModel

        started = time.time()
        print("stt runtime: cuda failure detected, switching to cpu/int8", flush=True)
        self.cfg.stt_device = "cpu"
        self.cfg.stt_compute_type = "int8"
        self._stt_model = WhisperModel(self.cfg.stt_model, device="cpu", compute_type="int8")
        print(f"stt model ready in {time.time() - started:.1f}s (device=cpu runtime fallback)", flush=True)

    def transcribe(self, audio_i16: np.ndarray) -> str:
        if audio_i16.size == 0:
            return ""

        stt_rate = 16_000
        audio_f32 = audio_i16.astype(np.float32) / 32767.0
        if self.cfg.sample_rate != stt_rate:
            audio_f32 = resample_poly(audio_f32, stt_rate, self.cfg.sample_rate).astype(np.float32)

        context_samples = int(max(0.0, self.cfg.audio_context_s) * stt_rate)
        context_audio = (
            self._prev_audio_tail[-context_samples:]
            if (self._prev_audio_tail is not None and context_samples > 0)
            else np.zeros((0,), dtype=np.float32)
        )
        prepended_s = len(context_audio) / stt_rate
        combined_f32 = np.concatenate([context_audio, audio_f32], dtype=np.float32)
        combined_i16 = np.clip(combined_f32 * 32767.0, -32768, 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name

        try:
            wav_write(wav_path, stt_rate, combined_i16)
            language = None if self.cfg.input_language == "auto" else self.cfg.input_language
            initial_prompt = self._context_text if (self.cfg.context_enabled and self._context_text) else None
            retried_on_cpu = False
            while True:
                try:
                    segments, _info = self._stt_model.transcribe(
                        wav_path,
                        language=language,
                        vad_filter=True,
                        beam_size=max(1, self.cfg.stt_beam_size),
                        no_speech_threshold=self.cfg.stt_no_speech_threshold,
                        log_prob_threshold=self.cfg.stt_log_prob_threshold,
                        compression_ratio_threshold=self.cfg.stt_compression_ratio_threshold,
                        condition_on_previous_text=self.cfg.stt_condition_on_previous_text,
                        initial_prompt=initial_prompt,
                        word_timestamps=True,
                    )
                    # Consume the generator here so runtime CUDA failures are caught in this block.
                    segments = list(segments)
                    break
                except RuntimeError as e:
                    if retried_on_cpu or not self._is_cuda_runtime_failure(e):
                        raise
                    self._switch_stt_to_cpu()
                    retried_on_cpu = True
            cutoff = max(0.0, prepended_s - self.cfg.audio_context_pad_s)
            kept_words: list[str] = []
            for s in segments:
                words = getattr(s, "words", None) or []
                if words:
                    for w in words:
                        if getattr(w, "end", 0.0) >= cutoff:
                            token = str(getattr(w, "word", "")).strip()
                            if token:
                                kept_words.append(token)
                elif getattr(s, "end", 0.0) >= cutoff:
                    seg_text = s.text.strip()
                    if seg_text:
                        kept_words.extend(seg_text.split())
            text = " ".join(kept_words).strip()
            text = self._dedup(text)
            text = self._trim_prefix_overlap(text)
            text = self._trim_first_word_overlap(text)
            if text and self.cfg.context_enabled:
                self._context_text = (self._context_text + " " + text).strip()[-self.cfg.context_chars :]
            if text:
                self._last_emitted_words = (self._last_emitted_words + text.split())[-80:]
            self._prev_audio_tail = audio_f32[-context_samples:] if context_samples > 0 else None
            return text
        finally:
            with contextlib.suppress(OSError):
                os.remove(wav_path)

    def cleanup(self, text: str) -> str:
        if not text or not self.cfg.cleanup or not self.cfg.ollama_model:
            return text
        if self._looks_clean(text):
            return text

        payload = {
            "model": self.cfg.ollama_model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a dictation post-processor. "
                        "Fix punctuation and capitalization only. "
                        "Do not add ideas. Output only corrected text."
                    ),
                },
                {"role": "user", "content": text},
            ],
        }

        try:
            resp = requests.post(self.cfg.ollama_url, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            return self._postprocess(str(content).strip()) or text
        except Exception:
            return text

    def output(self, text: str) -> None:
        if not text or not self.cfg.paste:
            return
        if self._has_output:
            text = " " + text
        mode = self.cfg.paste_mode
        target = self._active_window_desc()
        self._debug_log(f"paste target={target} mode={mode}")
        if mode == "type":
            self._kb.type(text)
        elif mode == "primary":
            with self._preserve_clipboards(primary=True, clipboard=True):
                if not self._copy_primary_selection(text):
                    pyperclip.copy(text)
                    self._paste_with_shortcut()
                elif platform.system() == "Linux" and self.cfg.paste_primary_click:
                    with contextlib.suppress(Exception):
                        self._mouse.click(MouseButton.middle, 1)
        else:
            with self._preserve_clipboards(primary=False, clipboard=True):
                pyperclip.copy(text)
                self._paste_with_shortcut()
        self._has_output = True

    def _debug_log(self, msg: str) -> None:
        if self.cfg.debug:
            print(f"[debug {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def _paste_with_shortcut(self) -> None:
        time.sleep(0.04)
        self._kb.press(self._paste_mod)
        self._kb.press("v")
        self._kb.release("v")
        self._kb.release(self._paste_mod)

    def _copy_primary_selection(self, text: str) -> bool:
        if platform.system() != "Linux":
            return False
        candidates: list[list[str]] = []
        if shutil.which("wl-copy"):
            candidates.append(["wl-copy", "--primary", "--type", "text/plain"])
        if shutil.which("xclip"):
            candidates.append(["xclip", "-selection", "primary", "-in"])
        if shutil.which("xsel"):
            candidates.append(["xsel", "--primary", "--input"])
        for cmd in candidates:
            try:
                subprocess.run(
                    cmd,
                    input=text,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1,
                    check=True,
                )
                return True
            except Exception:
                continue
        return False

    @contextlib.contextmanager
    def _preserve_clipboards(self, primary: bool, clipboard: bool):
        saved_primary: str | None = None
        saved_clipboard: str | None = None
        if self.cfg.paste_preserve:
            if primary:
                saved_primary = self._read_primary_selection()
            if clipboard:
                saved_clipboard = self._read_clipboard_text()
        try:
            yield
        finally:
            if self.cfg.paste_preserve:
                if self.cfg.paste_restore_delay_ms > 0:
                    time.sleep(max(0, self.cfg.paste_restore_delay_ms) / 1000.0)
                if primary and saved_primary is not None:
                    self._copy_primary_selection(saved_primary)
                if clipboard and saved_clipboard is not None:
                    self._write_clipboard_text(saved_clipboard)

    @staticmethod
    def _read_clipboard_text() -> str | None:
        try:
            return pyperclip.paste()
        except Exception:
            return None

    @staticmethod
    def _write_clipboard_text(text: str) -> bool:
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            return False

    @staticmethod
    def _read_primary_selection() -> str | None:
        if platform.system() != "Linux":
            return None
        candidates: list[list[str]] = []
        if shutil.which("wl-paste"):
            candidates.append(["wl-paste", "--primary", "--no-newline"])
        if shutil.which("xclip"):
            candidates.append(["xclip", "-selection", "primary", "-out"])
        if shutil.which("xsel"):
            candidates.append(["xsel", "--primary", "--output"])
        for cmd in candidates:
            try:
                out = subprocess.check_output(cmd, text=True, timeout=1)
                return out
            except Exception:
                continue
        return None

    @staticmethod
    def _active_window_desc() -> str:
        if platform.system() != "Linux":
            return "unknown"
        if shutil.which("xdotool"):
            try:
                title = subprocess.check_output(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    text=True,
                    timeout=1,
                ).strip()
                if title:
                    return title
            except Exception:
                pass
        return "unknown"

    @staticmethod
    def _dedup(text: str) -> str:
        words = text.split()
        n = len(words)
        if n < 4:
            return text
        half = n // 2
        if " ".join(words[:half]).lower() == " ".join(words[half : half * 2]).lower():
            return " ".join(words[:half])
        return text

    @staticmethod
    def _looks_clean(text: str) -> bool:
        words = text.split()
        if not words or len(words) > 8:
            return False
        if not (text[0].isupper() or text[0].isdigit() or text[0] in '"('):
            return False
        if len(words) >= 4 and text[-1] not in ".!?,;:":
            return False
        return True

    @staticmethod
    def _postprocess(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        for token in ("<|end|>", "<|endoftext|>", "</s>"):
            text = text.replace(token, "")
        for p in (
            "Sure, here's the corrected text:",
            "Here is the corrected text:",
            "Corrected text:",
            "Output:",
            "Result:",
        ):
            if text.lower().startswith(p.lower()):
                text = text[len(p) :].strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            text = text[1:-1]
        return text.strip()

    def _trim_prefix_overlap(self, text: str) -> str:
        if not text:
            return text
        new_words = text.split()
        old_words = self._last_emitted_words
        if not new_words or not old_words:
            return text
        max_k = min(len(new_words), len(old_words), 24)
        overlap = 0
        for k in range(max_k, 0, -1):
            if [w.lower() for w in new_words[:k]] == [w.lower() for w in old_words[-k:]]:
                overlap = k
                break
        if overlap > 0:
            new_words = new_words[overlap:]
        return " ".join(new_words).strip()

    def _trim_first_word_overlap(self, text: str) -> str:
        if not text or not self._last_emitted_words:
            return text
        new_words = text.split()
        if not new_words:
            return text
        norm_prev = [self._normalize_word(w) for w in self._last_emitted_words]
        norm_new = [self._normalize_word(w) for w in new_words]
        max_k = min(3, len(norm_prev), len(norm_new))
        trim_n = 0
        for k in range(max_k, 0, -1):
            if all(norm_prev[-k + i] and norm_new[i] and norm_prev[-k + i] == norm_new[i] for i in range(k)):
                trim_n = k
                break
        if trim_n > 0:
            new_words = new_words[trim_n:]
        return " ".join(new_words).strip()

    @staticmethod
    def _normalize_word(word: str) -> str:
        return re.sub(r"^[^\w]+|[^\w]+$", "", word.lower())


def resolve_ptt_key(name: str) -> keyboard.Key:
    mapping = {
        "super": keyboard.Key.cmd,
        "super_r": keyboard.Key.cmd_r,
        "super_l": keyboard.Key.cmd_l,
        "win": keyboard.Key.cmd,
        "shift_l": keyboard.Key.shift_l,
        "shift_r": keyboard.Key.shift_r,
        "ctrl_l": keyboard.Key.ctrl_l,
        "ctrl_r": keyboard.Key.ctrl_r,
        "cmd_r": keyboard.Key.cmd_r,
        "cmd_l": keyboard.Key.cmd_l,
        "alt_l": keyboard.Key.alt_l,
        "alt_r": keyboard.Key.alt_r,
    }
    return mapping.get(name, keyboard.Key.ctrl_l)


def _list_input_devices() -> list[tuple[int, str]]:
    devices: list[tuple[int, str]] = []
    for idx, dev in enumerate(sd.query_devices()):
        max_in = int(dev.get("max_input_channels", 0))
        if max_in > 0:
            devices.append((idx, str(dev.get("name", ""))))
    return devices


def _print_input_devices() -> None:
    default_input = sd.default.device[0]
    for idx, name in _list_input_devices():
        marker = " (default)" if idx == default_input else ""
        print(f"{idx:3d} {name}{marker}")


def _pick_loopback_device(hint: str) -> tuple[int | None, str]:
    devices = _list_input_devices()
    if not devices:
        return None, "none"

    hint_l = hint.strip().lower()
    bluetooth_terms = ("bluetooth", "wh-1000", "handsfree", "headset", "a2dp")

    def score(name: str) -> int:
        n = name.lower()
        s = 0
        if "loopback (loopback pcm)" in n:
            s += 100
        if "monitor" in n:
            s += 80
        if "loopback" in n:
            s += 60
        if "pulse" in n or "pipewire" in n:
            s += 20
        if hint_l and hint_l in n:
            s += 50
        if any(t in n for t in bluetooth_terms):
            s -= 120
        return s

    best_idx, best_name = max(devices, key=lambda d: score(d[1]))
    return best_idx, best_name


def _find_input_device_by_name(substr: str) -> int | None:
    needle = substr.strip().lower()
    if not needle:
        return None
    for idx, name in _list_input_devices():
        if needle in name.lower():
            return idx
    return None


def _default_sink_monitor_source() -> str | None:
    try:
        out = subprocess.check_output(["pactl", "info"], text=True, timeout=2)
    except Exception:
        return None
    for line in out.splitlines():
        if line.startswith("Default Sink:"):
            sink = line.split(":", 1)[1].strip()
            if sink:
                return f"{sink}.monitor"
    return None


def _active_sink_monitor_source() -> str | None:
    try:
        sink_inputs = subprocess.check_output(
            ["pactl", "list", "short", "sink-inputs"], text=True, timeout=2
        )
        sinks = subprocess.check_output(["pactl", "list", "short", "sinks"], text=True, timeout=2)
    except Exception:
        return None

    sink_id_to_name: dict[str, str] = {}
    for line in sinks.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            sink_id_to_name[parts[0].strip()] = parts[1].strip()

    last_sink_id: str | None = None
    for line in sink_inputs.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            last_sink_id = parts[1].strip()

    if not last_sink_id:
        return None
    sink_name = sink_id_to_name.get(last_sink_id)
    if not sink_name:
        return None
    return f"{sink_name}.monitor"


def _resume_media_playback() -> None:
    # Best-effort resume for Linux desktops where PTT/mic profile switches can pause playback.
    with contextlib.suppress(Exception):
        subprocess.run(
            ["playerctl", "play"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
            check=False,
        )


def main() -> int:
    if "--list-input-devices" in sys.argv:
        _print_input_devices()
        return 0

    cfg = Config()
    if cfg.paste_mode not in {"clipboard", "type", "primary"}:
        cfg.paste_mode = "clipboard"
    if not cfg.stt_compute_type:
        cfg.stt_compute_type = "float16" if cfg.stt_device in {"auto", "cuda"} else "int8"
    file_log_lock = threading.Lock()
    file_log_path = f"{datetime.now().strftime('%Y%m%d')}.log"

    def flog(event: str, message: str) -> None:
        if not cfg.file_log_enabled:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe = message.replace("\n", " ").replace("\r", " ")
        line = f"{ts} [{event}] {safe}\n"
        with file_log_lock:
            with open(file_log_path, "a", encoding="utf-8") as f:
                f.write(line)

    def dlog(msg: str) -> None:
        if cfg.debug:
            print(f"[debug {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    ptt_indicator_lock = threading.Lock()
    ptt_indicator_visible = False

    def show_ptt_indicator() -> None:
        nonlocal ptt_indicator_visible
        if not cfg.debug or cfg.mode != "ptt":
            return
        with ptt_indicator_lock:
            if ptt_indicator_visible:
                return
            sys.stderr.write("*")
            sys.stderr.flush()
            ptt_indicator_visible = True

    def clear_ptt_indicator() -> None:
        nonlocal ptt_indicator_visible
        if not cfg.debug or cfg.mode != "ptt":
            return
        with ptt_indicator_lock:
            if not ptt_indicator_visible:
                return
            # Erase the single-char live PTT marker.
            sys.stderr.write("\b \b")
            sys.stderr.flush()
            ptt_indicator_visible = False
    if (dev := os.environ.get("DICTATE_INPUT_DEVICE")) is not None:
        try:
            cfg.device_id = int(dev)
        except ValueError:
            pass
    elif cfg.input_device_name:
        matched = _find_input_device_by_name(cfg.input_device_name)
        if matched is not None:
            cfg.device_id = matched
            print(f"input device selected by name: {cfg.device_id} ({cfg.input_device_name})", flush=True)
        else:
            print(f"input device name not found: {cfg.input_device_name!r}", flush=True)

    if cfg.mode not in {"ptt", "loopback"}:
        print(f"invalid DICTATE_MODE={cfg.mode!r}, expected 'ptt' or 'loopback'")
        return 2

    if cfg.mode == "loopback" and "DICTATE_PASTE" not in os.environ:
        cfg.paste = False
    if cfg.mode == "loopback" and cfg.device_id is None:
        pulse_source = (
            os.environ.get("DICTATE_PULSE_SOURCE")
            or _active_sink_monitor_source()
            or _default_sink_monitor_source()
        )
        pulse_idx = _find_input_device_by_name("pulse")
        if pulse_source and pulse_idx is not None:
            os.environ["PULSE_SOURCE"] = pulse_source
            cfg.device_id = pulse_idx
            print(
                f"loopback input auto-selected via pulse monitor: "
                f"source={pulse_source} device={cfg.device_id}"
            , flush=True)
            dlog(f"PULSE_SOURCE={os.environ.get('PULSE_SOURCE')}")
            flog("SOURCE", f"auto pulse monitor source={pulse_source} device={cfg.device_id}")
        else:
            auto_id, auto_name = _pick_loopback_device(cfg.loopback_hint)
            cfg.device_id = auto_id
            print(f"loopback input auto-selected: {cfg.device_id} ({auto_name})", flush=True)
            flog("SOURCE", f"auto fallback source_name={auto_name} device={cfg.device_id}")

    recorder = Recorder(cfg)
    pipeline = Pipeline(cfg)
    ptt_key = resolve_ptt_key(cfg.ptt_key)
    work_q: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    print(
        f"platform={platform.system().lower()} mode={cfg.mode} "
        f"ptt={cfg.ptt_key} stt={cfg.stt_model} cleanup_model={cfg.ollama_model or 'disabled'} "
        f"paste={cfg.paste} paste_mode={cfg.paste_mode}"
    , flush=True)
    dlog(f"sample_rate={cfg.sample_rate} stt_device={cfg.stt_device} stt_compute={cfg.stt_compute_type}")
    dlog(
        "stt_decode="
        f"beam={cfg.stt_beam_size} "
        f"no_speech={cfg.stt_no_speech_threshold} "
        f"logprob={cfg.stt_log_prob_threshold} "
        f"compression={cfg.stt_compression_ratio_threshold}"
    )
    dlog(f"stt_condition_on_previous_text={cfg.stt_condition_on_previous_text}")
    if cfg.stt_condition_on_previous_text:
        dlog("warning: condition_on_previous_text=True can increase repetition loops")
    dlog(f"context_enabled={cfg.context_enabled} context_chars={cfg.context_chars}")
    dlog(f"context_reset_every={cfg.context_reset_every}")
    dlog(f"audio_context_s={cfg.audio_context_s} audio_context_pad_s={cfg.audio_context_pad_s}")
    dlog(f"trim_chunk_terminal_period={cfg.trim_chunk_terminal_period}")
    dlog(f"ptt_auto_resume_media={cfg.ptt_auto_resume_media}")
    dlog(
        "loop_guard="
        f"{cfg.loop_guard_enabled} "
        f"repeat_ratio={cfg.loop_guard_max_repeat_ratio} "
        f"punct_ratio={cfg.loop_guard_max_punct_ratio} "
        f"short_run={cfg.loop_guard_short_run} "
        f"short_len={cfg.loop_guard_short_len}"
    )
    dlog(
        "transcription destination: stdout"
        + (f" + paste({cfg.paste_mode})" if cfg.paste else " only")
    )
    dlog(f"resolved_ptt_key={ptt_key!r}")
    dlog(f"debug_keys={cfg.debug_keys}")
    flog(
        "START",
        " ".join(
            [
                f"mode={cfg.mode}",
                f"device={cfg.device_id}",
                f"stt_model={cfg.stt_model}",
                f"cleanup_model={cfg.ollama_model or 'disabled'}",
                f"paste={cfg.paste}",
            ]
        ),
    )

    def worker() -> None:
        chunk_no = 0
        bad_streak = 0
        ok_since_reset = 0
        while not stop_event.is_set():
            try:
                audio = work_q.get(timeout=0.5)
            except queue.Empty:
                continue
            chunk_no += 1
            if audio.size == 0:
                dlog(f"chunk#{chunk_no}: empty")
                continue
            if len(audio) < int(cfg.sample_rate * 0.2):
                dlog(f"chunk#{chunk_no}: too short ({len(audio)/cfg.sample_rate:.3f}s)")
                continue

            rms = float(np.sqrt(np.mean((audio.astype(np.float32) / 32767.0) ** 2) + 1e-12))
            dlog(f"chunk#{chunk_no}: {len(audio)/cfg.sample_rate:.2f}s rms={rms:.6f}")
            flog("CHUNK", f"n={chunk_no} sec={len(audio)/cfg.sample_rate:.2f} rms={rms:.6f}")
            if rms < cfg.min_chunk_rms:
                dlog(f"chunk#{chunk_no}: below rms threshold {cfg.min_chunk_rms:.6f}, skipped")
                flog("CHUNK_SKIP", f"n={chunk_no} reason=low_rms threshold={cfg.min_chunk_rms:.6f}")
                continue

            t0 = time.time()
            flog("MODEL_CALL", f"n={chunk_no} model=stt beam={cfg.stt_beam_size} cond_prev={cfg.stt_condition_on_previous_text}")
            try:
                text = pipeline.transcribe(audio)
            except Exception as e:
                dlog(f"chunk#{chunk_no}: transcribe error: {e}")
                flog("CHUNK_SKIP", f"n={chunk_no} reason=transcribe_error err={e!r}")
                continue
            dlog(f"chunk#{chunk_no}: transcribe_ms={(time.time()-t0)*1000:.0f}")
            flog("MODEL_RESULT", f"n={chunk_no} model=stt ms={(time.time()-t0)*1000:.0f} text={text!r}")
            if not text:
                dlog(f"chunk#{chunk_no}: no speech text")
                flog("CHUNK_SKIP", f"n={chunk_no} reason=no_text")
                continue
            if cfg.trim_chunk_terminal_period:
                trimmed = _trim_terminal_period_for_chunk(text)
                if trimmed != text:
                    dlog(f"chunk#{chunk_no}: trimmed terminal period")
                    text = trimmed
            if cfg.loop_guard_enabled and _is_pathological_loop_text(
                text,
                cfg.loop_guard_max_repeat_ratio,
                cfg.loop_guard_max_punct_ratio,
                cfg.loop_guard_short_run,
                cfg.loop_guard_short_len,
            ):
                bad_streak += 1
                dlog(f"chunk#{chunk_no}: pathological repetition detected (streak={bad_streak}), dropped")
                flog("CHUNK_SKIP", f"n={chunk_no} reason=loop_detected streak={bad_streak} text={text!r}")
                pipeline.reset_context()
                ok_since_reset = 0
                continue
            bad_streak = 0
            dlog(f"chunk#{chunk_no}: raw_text={text!r}")
            t1 = time.time()
            flog("MODEL_CALL", f"n={chunk_no} model=cleanup enabled={cfg.cleanup and bool(cfg.ollama_model)}")
            final = pipeline.cleanup(text)
            dlog(f"chunk#{chunk_no}: cleanup_ms={(time.time()-t1)*1000:.0f}")
            flog("MODEL_RESULT", f"n={chunk_no} model=cleanup ms={(time.time()-t1)*1000:.0f} text={final!r}")
            if final:
                pipeline.output(final)
                clear_ptt_indicator()
                sys.stdout.write(final + " ")
                sys.stdout.flush()
                flog("OUTPUT", f"n={chunk_no} text={final!r}")
                ok_since_reset += 1
                if cfg.context_reset_every > 0 and ok_since_reset >= cfg.context_reset_every:
                    dlog(f"chunk#{chunk_no}: periodic context reset after {ok_since_reset} chunks")
                    flog("RESET", f"n={chunk_no} reason=periodic count={ok_since_reset}")
                    pipeline.reset_context()
                    ok_since_reset = 0
            else:
                dlog(f"chunk#{chunk_no}: empty after cleanup")
                flog("CHUNK_SKIP", f"n={chunk_no} reason=empty_after_cleanup")

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    held = False

    def on_press(k: keyboard.Key | keyboard.KeyCode | None) -> None:
        nonlocal held
        if cfg.debug and cfg.debug_keys:
            dlog(f"key press: {k!r} matches_ptt={k == ptt_key}")
        if k == ptt_key and not held:
            held = True
            try:
                recorder.start()
                show_ptt_indicator()
                dlog("ptt pressed: recording started")
            except Exception as e:
                held = False
                print(f"record start failed: {e}")

    def on_release(k: keyboard.Key | keyboard.KeyCode | None) -> None:
        nonlocal held
        if cfg.debug and cfg.debug_keys:
            dlog(f"key release: {k!r} matches_ptt={k == ptt_key}")
        if k == ptt_key and held:
            held = False
            audio = recorder.stop()
            work_q.put(audio)
            dlog("ptt released: queued audio chunk")
            if cfg.mode == "ptt" and cfg.ptt_auto_resume_media and platform.system() == "Linux":
                _resume_media_playback()
                dlog("ptt release: requested media resume via playerctl")

    if cfg.mode == "ptt":
        print(
            f"dictate-min running. Hold PTT key '{cfg.ptt_key}' to record; release to transcribe."
        )
        print("Press Ctrl+C to quit.")

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            listener.stop()
            recorder.stop()
            clear_ptt_indicator()
    else:
        print(
            f"dictate-min loopback mode. Capturing device={cfg.device_id if cfg.device_id is not None else 'default'} "
            f"in {cfg.loopback_chunk_s}s chunks."
        , flush=True)
        print("Press Ctrl+C to quit.", flush=True)
        dlog("loopback capture stream opened once; chunking without closing stream")
        try:
            recorder.start()
            while True:
                time.sleep(max(1, cfg.loopback_chunk_s))
                audio = recorder.take_chunk()
                work_q.put(audio)
                dlog(f"queued chunk samples={audio.size}")
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            recorder.stop()

    return 0


def _trim_terminal_period_for_chunk(text: str) -> str:
    s = text.rstrip()
    if not s:
        return s
    if len(s.split()) < 3:
        return s
    if s.endswith("..."):
        return s[:-3].rstrip()
    if s.endswith("."):
        return s[:-1].rstrip()
    return s


def _is_pathological_loop_text(
    text: str,
    max_repeat_ratio: float,
    max_punct_ratio: float,
    short_run: int,
    short_len: int,
) -> bool:
    s = text.strip()
    if not s:
        return False
    words = s.split()
    if len(words) >= 8:
        normalized = [w.lower().strip(".,!?;:()[]{}\"'") for w in words if w.strip()]
        if normalized:
            unique_ratio = len(set(normalized)) / max(1, len(normalized))
            repeat_ratio = 1.0 - unique_ratio
            if repeat_ratio >= max_repeat_ratio:
                return True

    punct = sum(1 for ch in s if ch in "。.,!?;:")
    if punct / max(1, len(s)) >= max_punct_ratio:
        return True

    if "。 i 。 i" in s.lower() or "i 。 i 。" in s.lower():
        return True

    # Catch classic stutters like "nd nd nd nd" / "i i i i"
    toks = [re.sub(r"^[^\w]+|[^\w]+$", "", w.lower()) for w in words]
    run = 1
    for i in range(1, len(toks)):
        if toks[i] and toks[i] == toks[i - 1] and len(toks[i]) <= max(1, short_len):
            run += 1
            if run >= max(2, short_run):
                return True
        else:
            run = 1
    return False


if __name__ == "__main__":
    raise SystemExit(main())
