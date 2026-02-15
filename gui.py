#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from urllib.parse import urlparse

import requests


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: str  # bool | str | int | float | enum | combo
    default: str
    section: str
    choices: tuple[str, ...] = ()
    width: int = 38


FIELD_SPECS: list[FieldSpec] = [
    FieldSpec("DICTATE_MODE", "Mode", "enum", "ptt", "General", ("ptt", "loopback")),
    FieldSpec("DICTATE_PTT_KEY", "PTT Key", "enum", "ctrl_r", "General", ("cmd_r", "super_r", "cmd_l", "super_l", "super", "win", "shift_r", "shift_l", "ctrl_l", "ctrl_r", "alt_l", "alt_r")),
    FieldSpec("DICTATE_INPUT_DEVICE", "Input Device ID", "combo", "", "General"),
    FieldSpec("DICTATE_INPUT_DEVICE_NAME", "Input Device Name", "combo", "", "General"),
    FieldSpec("DICTATE_INPUT_LANGUAGE", "Input Language", "str", "auto", "General"),
    FieldSpec("DICTATE_SAMPLE_RATE", "Sample Rate", "int", "16000", "General"),
    FieldSpec("DICTATE_DEBUG", "Debug", "bool", "0", "General"),
    FieldSpec("DICTATE_DEBUG_KEYS", "Debug Keys", "bool", "0", "General"),
    FieldSpec("DICTATE_FILE_LOG", "File Log", "bool", "1", "General"),

    FieldSpec("DICTATE_PASTE", "Paste Output", "bool", "1", "Output"),
    FieldSpec("DICTATE_PASTE_ALIGN_FOCUS", "Align Paste to PTT Focus", "bool", "1", "Output"),
    FieldSpec("DICTATE_PASTE_MODE", "Paste Mode", "enum", "type", "Output", ("clipboard", "type", "primary")),
    FieldSpec("DICTATE_PASTE_PRIMARY_CLICK", "Primary Click Paste", "bool", "1", "Output"),
    FieldSpec("DICTATE_PASTE_PRESERVE", "Preserve Clipboard", "bool", "1", "Output"),
    FieldSpec("DICTATE_PASTE_RESTORE_DELAY_MS", "Restore Delay (ms)", "int", "80", "Output"),

    FieldSpec("DICTATE_PTT_AUTO_PAUSE_MEDIA", "Auto Pause Media", "bool", "1", "PTT Media"),
    FieldSpec("DICTATE_PTT_DUCK_MEDIA", "Duck Media", "bool", "0", "PTT Media"),
    FieldSpec("DICTATE_PTT_DUCK_SCOPE", "Duck Scope", "enum", "default", "PTT Media", ("default", "all")),
    FieldSpec("DICTATE_PTT_DUCK_MEDIA_PERCENT", "Duck Percent", "int", "30", "PTT Media"),
    FieldSpec("DICTATE_PTT_AUTO_SUBMIT", "Auto Submit", "bool", "0", "PTT Media"),

    FieldSpec("DICTATE_LOOPBACK_CHUNK_S", "Loopback Chunk Seconds", "int", "4", "Loopback"),
    FieldSpec("DICTATE_LOOPBACK_HINT", "Loopback Hint", "str", "loopback pcm", "Loopback"),
    FieldSpec("DICTATE_PULSE_SOURCE", "Pulse Source", "str", "", "Loopback"),
    FieldSpec("DICTATE_MIN_CHUNK_RMS", "Min Chunk RMS", "float", "0.0008", "Loopback"),

    FieldSpec(
        "DICTATE_STT_MODEL",
        "STT Model",
        "combo",
        "medium.en",
        "Whisper",
        ("tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v2", "large-v3", "large"),
    ),
    FieldSpec("DICTATE_STT_DEVICE", "STT Device", "enum", "auto", "Whisper", ("cpu", "auto", "cuda")),
    FieldSpec("DICTATE_STT_COMPUTE", "STT Compute", "str", "", "Whisper"),
    FieldSpec("DICTATE_STT_CONDITION_PREV", "Condition on Previous", "bool", "0", "Whisper"),
    FieldSpec("DICTATE_STT_BEAM_SIZE", "Beam Size", "int", "5", "Whisper"),
    FieldSpec("DICTATE_STT_NO_SPEECH_THRESHOLD", "No-Speech Threshold", "float", "0.6", "Whisper"),
    FieldSpec("DICTATE_STT_LOGPROB_THRESHOLD", "Logprob Threshold", "float", "-1.0", "Whisper"),
    FieldSpec("DICTATE_STT_COMPRESSION_RATIO_THRESHOLD", "Compression Ratio Threshold", "float", "2.4", "Whisper"),
    FieldSpec("DICTATE_STT_TAIL_PAD_S", "Tail Pad Seconds", "float", "0.08", "Whisper"),

    FieldSpec("DICTATE_CONTEXT", "Context Enabled", "bool", "1", "Context"),
    FieldSpec("DICTATE_CONTEXT_CHARS", "Context Chars", "int", "600", "Context"),
    FieldSpec("DICTATE_CONTEXT_RESET_EVERY", "Context Reset Every", "int", "1", "Context"),
    FieldSpec("DICTATE_AUDIO_CONTEXT_S", "Audio Context Seconds", "float", "1.6", "Context"),
    FieldSpec("DICTATE_AUDIO_CONTEXT_PAD_S", "Audio Context Pad", "float", "0.12", "Context"),
    FieldSpec("DICTATE_TRIM_CHUNK_PERIOD", "Trim Chunk Period", "bool", "1", "Context"),
    FieldSpec("DICTATE_LOOP_GUARD", "Loop Guard", "bool", "1", "Context"),
    FieldSpec("DICTATE_LOOP_GUARD_REPEAT_RATIO", "Loop Repeat Ratio", "float", "0.55", "Context"),
    FieldSpec("DICTATE_LOOP_GUARD_PUNCT_RATIO", "Loop Punct Ratio", "float", "0.35", "Context"),
    FieldSpec("DICTATE_LOOP_GUARD_SHORT_RUN", "Loop Short Run", "int", "4", "Context"),
    FieldSpec("DICTATE_LOOP_GUARD_SHORT_LEN", "Loop Short Len", "int", "3", "Context"),
    FieldSpec("DICTATE_LOOP_GUARD_ALLOW", "Loop Guard Allow", "str", "ipv4,mac,semver,dotted_numeric", "Context"),

    FieldSpec("DICTATE_CLEANUP", "Cleanup Enabled", "bool", "1", "Cleanup"),
    FieldSpec("DICTATE_CLEANUP_BACKEND", "Cleanup Backend", "enum", "ollama", "Cleanup", ("ollama", "generic_v1", "api_v1", "lm_api_v1")),
    FieldSpec("DICTATE_CLEANUP_URL", "Cleanup URL", "str", "http://localhost:11434/api/chat", "Cleanup", width=52),
    FieldSpec("DICTATE_CLEANUP_MODEL", "Cleanup Model", "combo", "", "Cleanup", width=52),
    FieldSpec("DICTATE_CLEANUP_API_TOKEN", "Cleanup API Token", "str", "", "Cleanup", width=52),
    FieldSpec("LM_API_TOKEN", "LM API Token", "str", "", "Cleanup", width=52),
    FieldSpec("DICTATE_CLEANUP_PROMPT", "Cleanup Prompt", "str", "You are a dictation post-processor. Fix punctuation and capitalization only. Do not add ideas. Output only corrected text.", "Cleanup", width=70),
    FieldSpec("DICTATE_CLEANUP_PROMPTS", "Cleanup Prompt Rules", "str", "", "Cleanup", width=70),
    FieldSpec("DICTATE_CLEANUP_REASONING", "Cleanup Reasoning", "str", "off", "Cleanup"),
    FieldSpec("DICTATE_CLEANUP_TEMPERATURE", "Cleanup Temperature", "float", "0.2", "Cleanup"),
    FieldSpec("DICTATE_CLEANUP_HISTORY_SIZE", "Cleanup History Size", "int", "12", "Cleanup"),
]

RESTART_REQUIRED_ENV: set[str] = {
    "DICTATE_MODE",
    "DICTATE_INPUT_DEVICE",
    "DICTATE_INPUT_DEVICE_NAME",
    "DICTATE_SAMPLE_RATE",
    "DICTATE_PTT_KEY",
    "DICTATE_STT_MODEL",
    "DICTATE_STT_DEVICE",
    "DICTATE_STT_COMPUTE",
    "DICTATE_STT_CONDITION_PREV",
    "DICTATE_STT_BEAM_SIZE",
    "DICTATE_STT_NO_SPEECH_THRESHOLD",
    "DICTATE_STT_LOGPROB_THRESHOLD",
    "DICTATE_STT_COMPRESSION_RATIO_THRESHOLD",
    "DICTATE_STT_TAIL_PAD_S",
    "DICTATE_INPUT_LANGUAGE",
    "DICTATE_LOOPBACK_CHUNK_S",
    "DICTATE_LOOPBACK_HINT",
    "DICTATE_PULSE_SOURCE",
    "DICTATE_MIN_CHUNK_RMS",
    "DICTATE_CONTEXT",
    "DICTATE_CONTEXT_CHARS",
    "DICTATE_AUDIO_CONTEXT_S",
    "DICTATE_AUDIO_CONTEXT_PAD_S",
    "DICTATE_LOOP_GUARD_REPEAT_RATIO",
    "DICTATE_LOOP_GUARD_PUNCT_RATIO",
    "DICTATE_LOOP_GUARD_SHORT_RUN",
    "DICTATE_LOOP_GUARD_SHORT_LEN",
    "DICTATE_PASTE",
    "DICTATE_PASTE_MODE",
    "DICTATE_PASTE_PRIMARY_CLICK",
    "DICTATE_PASTE_PRESERVE",
    "DICTATE_PASTE_RESTORE_DELAY_MS",
}

BUILTIN_PRESETS: dict[str, dict[str, str]] = {
    "Default": {},
    "Terminal Dictation": {
        "DICTATE_MODE": "ptt",
        "DICTATE_PASTE": "1",
        "DICTATE_PASTE_MODE": "type",
        "DICTATE_PTT_DUCK_MEDIA": "1",
        "DICTATE_PTT_DUCK_SCOPE": "default",
        "DICTATE_CLEANUP_BACKEND": "generic_v1",
        "DICTATE_CLEANUP_REASONING": "off",
        "DICTATE_CLEANUP_TEMPERATURE": "0.2",
        "DICTATE_CLEANUP_PROMPTS": "/terminal/Output only shell-safe plain text commands. No markdown.",
    },
    "Low Latency": {
        "DICTATE_STT_MODEL": "tiny",
        "DICTATE_STT_BEAM_SIZE": "1",
        "DICTATE_CLEANUP": "0",
        "DICTATE_CONTEXT_RESET_EVERY": "1",
    },
}
PRESETS: dict[str, dict[str, str]] = {name: dict(values) for name, values in BUILTIN_PRESETS.items()}

ENV_TOOLTIPS: dict[str, str] = {
    "DICTATE_MODE": "Run mode: `ptt` records on push-to-talk key, `loopback` continuously captures output loopback audio.",
    "DICTATE_PTT_KEY": "Hotkey used for push-to-talk recording.",
    "DICTATE_INPUT_DEVICE": "Numeric audio input device index. Overrides name-based matching when set.",
    "DICTATE_INPUT_DEVICE_NAME": "Case-insensitive input device name substring match when no numeric device index is set.",
    "DICTATE_INPUT_LANGUAGE": "Whisper language code, or `auto` to detect automatically.",
    "DICTATE_SAMPLE_RATE": "Input sample rate hint in Hz. Falls back automatically if unsupported.",
    "DICTATE_DEBUG": "Enable verbose runtime diagnostics in logs.",
    "DICTATE_DEBUG_KEYS": "Log key press/release events and push-to-talk matching decisions.",
    "DICTATE_FILE_LOG": "Append runtime events to `YYYYMMDD.log` files.",
    "DICTATE_PASTE": "Emit transcribed text into the active app using configured paste mode.",
    "DICTATE_PASTE_ALIGN_FOCUS": "When enabled, capture focused window on PTT press; if focus changed by output time, paste into captured window, then restore focus.",
    "DICTATE_PASTE_MODE": "Output transport: `clipboard`, `type`, or `primary`.",
    "DICTATE_PASTE_PRIMARY_CLICK": "In Linux `primary` mode, trigger middle-click paste after setting PRIMARY selection.",
    "DICTATE_PASTE_PRESERVE": "Preserve/restore previous clipboard or PRIMARY content around paste.",
    "DICTATE_PASTE_RESTORE_DELAY_MS": "Delay before clipboard restore to avoid target-app paste races.",
    "DICTATE_PTT_AUTO_PAUSE_MEDIA": "Pause media playback when PTT starts (Linux).",
    "DICTATE_PTT_DUCK_MEDIA": "Lower output sink volume while PTT is held (Linux).",
    "DICTATE_PTT_DUCK_SCOPE": "Ducking target scope: `default` sink or `all` non-monitor sinks.",
    "DICTATE_PTT_DUCK_MEDIA_PERCENT": "Target media volume percent while ducking is active.",
    "DICTATE_PTT_AUTO_SUBMIT": "Press Enter after emitting each PTT chunk. Hold Shift on release to suppress per chunk.",
    "DICTATE_LOOPBACK_CHUNK_S": "Chunk size in seconds for loopback capture mode.",
    "DICTATE_LOOPBACK_HINT": "Name hint used when auto-selecting non-pulse loopback devices.",
    "DICTATE_PULSE_SOURCE": "Force a specific PulseAudio/PipeWire source name, e.g. `sink.monitor`.",
    "DICTATE_MIN_CHUNK_RMS": "Skip chunks below this RMS threshold as near-silent.",
    "DICTATE_STT_MODEL": "faster-whisper model name, e.g. `tiny`, `small`, `medium.en`, `large-v3`.",
    "DICTATE_STT_DEVICE": "Whisper device: `cpu`, `auto`, or `cuda`.",
    "DICTATE_STT_COMPUTE": "Whisper compute type, e.g. `int8` or `float16`. Empty uses automatic selection.",
    "DICTATE_STT_CONDITION_PREV": "Enable Whisper `condition_on_previous_text` behavior.",
    "DICTATE_STT_BEAM_SIZE": "Whisper beam-search width.",
    "DICTATE_STT_NO_SPEECH_THRESHOLD": "No-speech probability threshold for dropping segments.",
    "DICTATE_STT_LOGPROB_THRESHOLD": "Log-probability threshold for filtering low-confidence segments.",
    "DICTATE_STT_COMPRESSION_RATIO_THRESHOLD": "Compression-ratio threshold used to detect degenerate decoding output.",
    "DICTATE_STT_TAIL_PAD_S": "Trailing silence appended before decode to reduce end-of-chunk hallucinations.",
    "DICTATE_CONTEXT": "Enable text context carryover between emitted chunks.",
    "DICTATE_CONTEXT_CHARS": "Maximum retained context characters.",
    "DICTATE_CONTEXT_RESET_EVERY": "Reset context every N emitted chunks (`0` disables periodic reset).",
    "DICTATE_AUDIO_CONTEXT_S": "Audio overlap seconds prepended from previous chunk.",
    "DICTATE_AUDIO_CONTEXT_PAD_S": "Overlap pad used for timestamp clipping around the context boundary.",
    "DICTATE_TRIM_CHUNK_PERIOD": "Trim trailing `.` or `...` from chunk output.",
    "DICTATE_LOOP_GUARD": "Enable pathological loop detection and context reset safeguards.",
    "DICTATE_LOOP_GUARD_REPEAT_RATIO": "Loop guard repetition trigger ratio.",
    "DICTATE_LOOP_GUARD_PUNCT_RATIO": "Loop guard punctuation-density trigger ratio.",
    "DICTATE_LOOP_GUARD_SHORT_RUN": "Loop guard repeated short-token run trigger length.",
    "DICTATE_LOOP_GUARD_SHORT_LEN": "Maximum token length counted as `short` in loop-run detection.",
    "DICTATE_LOOP_GUARD_ALLOW": "Comma-separated (recommended; space/semicolon also accepted) allowlist that bypasses loop guard for exact matches. Supported: ipv4, ipv6, mac, semver, dotted_numeric.",
    "DICTATE_CLEANUP": "Enable cleanup pass over raw transcription output.",
    "DICTATE_CLEANUP_BACKEND": "Cleanup backend: `ollama` or OpenAI-compatible `generic_v1` aliases.",
    "DICTATE_CLEANUP_URL": "Cleanup chat endpoint URL.",
    "DICTATE_CLEANUP_MODEL": "Cleanup model name. Can be auto-discovered for supported backends.",
    "DICTATE_CLEANUP_API_TOKEN": "Bearer token used for cleanup backend authorization.",
    "LM_API_TOKEN": "Primary API token alias for cleanup backend authorization.",
    "DICTATE_CLEANUP_PROMPT": "Default cleanup system prompt.",
    "DICTATE_CLEANUP_PROMPTS": "Window-title prompt override rules (`/match/prompt`) with optional default fallback.",
    "DICTATE_CLEANUP_REASONING": "Reasoning mode option for generic_v1 payloads.",
    "DICTATE_CLEANUP_TEMPERATURE": "Sampling temperature for generic_v1 cleanup requests.",
    "DICTATE_CLEANUP_HISTORY_SIZE": "Number of prior cleanup exchanges kept for prompt template placeholders.",
}


def parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        data[k.strip()] = v.strip()
    return data


def write_env_file(path: Path, values: dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in sorted(values.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)

        self.inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 350) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event: tk.Event[tk.Widget]) -> None:
        self._schedule_show()

    def _on_leave(self, _event: tk.Event[tk.Widget]) -> None:
        self._cancel()
        self._hide()

    def _schedule_show(self) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self._tip,
            text=self.text,
            justify="left",
            background="#111827",
            foreground="#f8fafc",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
            wraplength=560,
        )
        label.pack()

    def _hide(self) -> None:
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class DictateGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Dictate Config")
        self.geometry("1180x820")
        self.minsize(980, 700)

        self._proc: subprocess.Popen[str] | None = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._vars: dict[str, tk.Variable] = {}
        self._widgets: dict[str, tk.Widget] = {}
        self._preset_box: ttk.Combobox | None = None
        self._spec_by_name = {s.name: s for s in FIELD_SPECS}
        self._env_path = tk.StringVar(value=str(Path.cwd() / ".env"))
        self._presets_path = Path.cwd() / ".dictate-presets.json"
        self._runtime_overrides_path = Path.cwd() / ".dictate-runtime.env"
        self._preset = tk.StringVar(value="Default")
        self._auto_restart = tk.BooleanVar(value=True)
        self._suspend_auto_apply = True
        self._pending_changed_keys: set[str] = set()
        self._pending_apply_after_id: str | None = None
        self._input_devices: list[tuple[str, str]] = []
        self._tooltips: list[Tooltip] = []

        self._load_saved_presets()
        self._setup_style()
        self._build_ui()
        self._reset_to_defaults()
        self._suspend_auto_apply = False
        self._refresh_input_devices()
        self.after(80, self._drain_log)

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        base_font = ("Segoe UI", 10)
        self.option_add("*Font", base_font)
        style.configure("TFrame", background="#f5f7fb")
        style.configure("TLabelframe", background="#f5f7fb")
        style.configure("TLabelframe.Label", font=("Segoe UI Semibold", 10), foreground="#1b2230")
        style.configure("TLabel", background="#f5f7fb", foreground="#1b2230")
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 14), foreground="#0f172a")
        style.configure("Sub.TLabel", foreground="#475569")
        style.configure("Warn.TLabel", foreground="#b45309")
        style.configure("TButton", padding=(10, 6))
        style.configure("Accent.TButton", padding=(10, 6), foreground="#ffffff", background="#0b5fff")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x")
        ttk.Label(top, text="Dictate Desktop Config", style="Header.TLabel").pack(anchor="w")
        ttk.Label(top, text="Edit environment settings, save to .env, and launch dictate-min.", style="Sub.TLabel").pack(anchor="w", pady=(2, 10))

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(0, 10))
        ttk.Label(controls, text="Preset").grid(row=0, column=0, sticky="w", padx=(0, 8))
        preset_box = ttk.Combobox(controls, textvariable=self._preset, values=list(PRESETS), state="normal", width=28)
        preset_box.grid(row=0, column=1, sticky="w")
        self._preset_box = preset_box
        ttk.Button(controls, text="Apply Preset", command=self._apply_preset).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(controls, text="Save Preset", command=self._save_preset).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(controls, text="Reset Defaults", command=self._reset_to_defaults).grid(row=0, column=4, padx=(8, 0))

        ttk.Label(controls, text="Env File").grid(row=0, column=5, sticky="w", padx=(16, 8))
        ttk.Entry(controls, textvariable=self._env_path, width=70).grid(row=0, column=6, sticky="ew")
        ttk.Button(controls, text="Browse", command=self._browse_env).grid(row=0, column=7, padx=(8, 0))
        ttk.Button(controls, text="Load", command=self._load_env).grid(row=0, column=8, padx=(8, 0))
        ttk.Button(controls, text="Save", command=self._save_env).grid(row=0, column=9, padx=(8, 0))

        ttk.Button(controls, text="Start dictate-min", style="Accent.TButton", command=self._start).grid(row=1, column=0, pady=(8, 0), padx=(0, 8))
        ttk.Button(controls, text="Apply Live", command=self._apply_live).grid(row=1, column=1, pady=(8, 0), padx=(0, 8))
        ttk.Button(controls, text="Stop", command=self._stop).grid(row=1, column=2, pady=(8, 0), padx=(0, 8))
        ttk.Button(controls, text="Refresh Models", command=self._refresh_model_choices).grid(row=1, column=3, pady=(8, 0), padx=(0, 8))
        ttk.Checkbutton(controls, text="Auto Restart", variable=self._auto_restart).grid(row=1, column=4, pady=(8, 0), padx=(4, 8), sticky="w")
        ttk.Button(controls, text="Refresh Input Devices", command=self._refresh_input_devices).grid(row=1, column=5, pady=(8, 0), padx=(0, 8))
        ttk.Button(controls, text="Copy Start Command", command=self._copy_start_command).grid(row=1, column=6, pady=(8, 0), padx=(0, 8))

        controls.columnconfigure(6, weight=1)

        body = ttk.Panedwindow(root, orient="vertical")
        body.pack(fill="both", expand=True)

        notebook = ttk.Notebook(body)
        body.add(notebook, weight=5)

        sections = sorted({s.section for s in FIELD_SPECS})
        for section in sections:
            tab = ScrollableFrame(notebook)
            notebook.add(tab, text=section)
            self._render_section(tab.inner, section)
        docs_tab = ttk.Frame(notebook)
        notebook.add(docs_tab, text="Documentation")
        self._render_documentation(docs_tab)

        log_frame = ttk.Labelframe(body, text="Runtime Log", padding=8)
        body.add(log_frame, weight=2)
        self._log = tk.Text(log_frame, height=12, wrap="word", bg="#0b1220", fg="#d1d9e6", insertbackground="#d1d9e6")
        self._log.tag_configure("log_error", foreground="#ef4444")
        self._log.tag_configure("log_warn", foreground="#f59e0b")
        self._log.pack(fill="both", expand=True)

    def _render_section(self, parent: ttk.Frame, section: str) -> None:
        row = 0
        for spec in [s for s in FIELD_SPECS if s.section == section]:
            label = ttk.Label(parent, text=spec.label)
            label.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=6)
            if spec.kind == "bool":
                var = tk.BooleanVar(value=(spec.default == "1"))
                w = ttk.Checkbutton(parent, variable=var)
                w.grid(row=row, column=1, sticky="w", pady=6)
            elif spec.kind == "enum":
                var = tk.StringVar(value=spec.default)
                w = ttk.Combobox(parent, textvariable=var, values=list(spec.choices), state="readonly", width=max(20, spec.width))
                w.grid(row=row, column=1, sticky="ew", pady=6)
            elif spec.kind == "combo":
                var = tk.StringVar(value=spec.default)
                w = ttk.Combobox(parent, textvariable=var, values=list(spec.choices), state="normal", width=max(20, spec.width))
                w.grid(row=row, column=1, sticky="ew", pady=6)
            else:
                var = tk.StringVar(value=spec.default)
                show = "*" if "TOKEN" in spec.name else ""
                w = ttk.Entry(parent, textvariable=var, width=spec.width, show=show)
                w.grid(row=row, column=1, sticky="ew", pady=6)
            env_label = ttk.Label(parent, text=spec.name, style="Sub.TLabel")
            env_label.grid(row=row, column=2, sticky="w", padx=(10, 0), pady=6)
            if spec.name in RESTART_REQUIRED_ENV:
                ttk.Label(parent, text="restart", style="Warn.TLabel").grid(row=row, column=3, sticky="w", padx=(8, 0), pady=6)
            self._vars[spec.name] = var
            self._widgets[spec.name] = w
            tip = self._tooltip_text(spec.name)
            self._add_tooltip(label, tip)
            self._add_tooltip(w, tip)
            self._add_tooltip(env_label, tip)
            var.trace_add("write", lambda *_args, key=spec.name: self._on_var_changed(key))
            row += 1
        parent.columnconfigure(1, weight=1)

    def _tooltip_text(self, name: str) -> str:
        details = ENV_TOOLTIPS.get(name, "No description available in README.")
        if name in RESTART_REQUIRED_ENV:
            return f"{name}\n{details}\nChange requires restart of dictate-min."
        return f"{name}\n{details}"

    def _add_tooltip(self, widget: tk.Widget, text: str) -> None:
        self._tooltips.append(Tooltip(widget, text))

    def _render_documentation(self, parent: ttk.Frame) -> None:
        frame = ttk.Frame(parent, padding=8)
        frame.pack(fill="both", expand=True)
        text = tk.Text(
            frame,
            wrap="word",
            bg="#ffffff",
            fg="#0f172a",
            insertbackground="#0f172a",
            relief="flat",
            padx=10,
            pady=10,
        )
        yscroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=yscroll.set)
        text.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        text.tag_configure("h1", font=("Segoe UI Semibold", 15), spacing1=10, spacing3=4)
        text.tag_configure("h2", font=("Segoe UI Semibold", 13), spacing1=8, spacing3=3)
        text.tag_configure("h3", font=("Segoe UI Semibold", 11), spacing1=6, spacing3=2)
        text.tag_configure("code", font=("Consolas", 10), background="#f1f5f9")
        text.tag_configure("inline_code", font=("Consolas", 10), background="#f8fafc")
        text.tag_configure("bullet", lmargin1=14, lmargin2=30)
        readme = Path.cwd() / "README.md"
        if not readme.exists():
            text.insert("end", "README.md not found.\n")
            text.configure(state="disabled")
            return
        lines = readme.read_text(encoding="utf-8").splitlines()
        in_code_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                text.insert("end", "\n")
                continue
            if in_code_block:
                text.insert("end", line + "\n", ("code",))
                continue
            heading = re.match(r"^(#{1,3})\s+(.*)$", line)
            if heading:
                level = len(heading.group(1))
                content = heading.group(2).strip()
                text.insert("end", content + "\n", (f"h{level}",))
                continue
            bullet = re.match(r"^\s*-\s+(.*)$", line)
            if bullet:
                text.insert("end", "- ", ("bullet",))
                self._insert_markdown_inline(text, bullet.group(1), ("bullet",))
                text.insert("end", "\n", ("bullet",))
                continue
            self._insert_markdown_inline(text, line)
            text.insert("end", "\n")
        text.configure(state="disabled")

    @staticmethod
    def _insert_markdown_inline(widget: tk.Text, line: str, tags: tuple[str, ...] = ()) -> None:
        i = 0
        for match in re.finditer(r"`([^`]+)`", line):
            start, end = match.span()
            if start > i:
                widget.insert("end", line[i:start], tags)
            widget.insert("end", match.group(1), (*tags, "inline_code"))
            i = end
        if i < len(line):
            widget.insert("end", line[i:], tags)

    def _browse_env(self) -> None:
        p = filedialog.asksaveasfilename(title="Select .env", defaultextension=".env", initialfile=".env")
        if p:
            self._env_path.set(p)

    def _apply_preset(self) -> None:
        self._suspend_auto_apply = True
        self._reset_to_defaults()
        preset = PRESETS.get(self._preset.get(), {})
        for key, value in preset.items():
            self._set_var(key, value)
        self._refresh_model_choices()
        self._suspend_auto_apply = False
        self._schedule_auto_apply(set(preset.keys()))

    def _save_preset(self) -> None:
        name = self._preset.get().strip()
        if not name:
            messagebox.showwarning("Preset", "Enter a preset name first.")
            return
        if name == "Default":
            messagebox.showwarning("Preset", "The 'Default' preset name is reserved.")
            return
        if name in PRESETS and not messagebox.askyesno("Preset", f"Overwrite preset '{name}'?"):
            return
        preset_values: dict[str, str] = {}
        for spec in FIELD_SPECS:
            current = self._var_as_string(spec.name)
            current_norm = self._normalize_for_compare(spec, current)
            default_norm = self._normalize_for_compare(spec, spec.default)
            if current_norm != default_norm:
                preset_values[spec.name] = current
        PRESETS[name] = preset_values
        self._refresh_preset_choices()
        if not self._write_saved_presets():
            return
        self._append_log(f"Saved preset '{name}' with {len(preset_values)} override(s)")

    def _reset_to_defaults(self) -> None:
        was_suspended = self._suspend_auto_apply
        self._suspend_auto_apply = True
        for spec in FIELD_SPECS:
            self._set_var(spec.name, spec.default)
        self._refresh_model_choices()
        self._suspend_auto_apply = was_suspended

    def _load_env(self) -> None:
        path = Path(self._env_path.get())
        data = parse_env_file(path)
        self._suspend_auto_apply = True
        for key, value in data.items():
            if key in self._vars:
                self._set_var(key, value)
        self._refresh_model_choices()
        self._suspend_auto_apply = False
        self._schedule_auto_apply(set(data.keys()))
        self._append_log(f"Loaded {len(data)} entries from {path}")

    def _save_env(self) -> None:
        path = Path(self._env_path.get())
        values = self._env_values(include_empty=False)
        write_env_file(path, values)
        self._append_log(f"Saved {len(values)} entries to {path}")

    def _env_values(self, include_empty: bool = False) -> dict[str, str]:
        out: dict[str, str] = {}
        for spec in FIELD_SPECS:
            v = self._var_as_string(spec.name)
            if v or include_empty:
                out[spec.name] = v
        return out

    def _set_var(self, name: str, value: str) -> None:
        var = self._vars[name]
        if isinstance(var, tk.BooleanVar):
            var.set(str(value).strip().lower() in {"1", "true", "yes", "on"})
        else:
            var.set(str(value))

    def _refresh_preset_choices(self) -> None:
        if self._preset_box is not None:
            self._preset_box["values"] = list(PRESETS)

    def _load_saved_presets(self) -> None:
        if not self._presets_path.exists():
            return
        try:
            data = json.loads(self._presets_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        for raw_name, raw_values in data.items():
            name = str(raw_name).strip()
            if not name or name == "Default":
                continue
            if not isinstance(raw_values, dict):
                continue
            clean: dict[str, str] = {}
            for key, value in raw_values.items():
                if key in self._spec_by_name:
                    clean[key] = str(value)
            PRESETS[name] = clean

    def _write_saved_presets(self) -> bool:
        to_save: dict[str, dict[str, str]] = {}
        for name, values in PRESETS.items():
            if name == "Default":
                continue
            if name in BUILTIN_PRESETS and values == BUILTIN_PRESETS[name]:
                continue
            to_save[name] = dict(values)
        try:
            self._presets_path.write_text(
                json.dumps(to_save, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            messagebox.showerror("Preset", f"Failed to write presets: {e!r}")
            return False
        return True

    def _var_as_string(self, name: str) -> str:
        spec = self._spec_by_name[name]
        var = self._vars[name]
        if spec.kind == "bool":
            return "1" if bool(var.get()) else "0"
        return str(var.get()).strip()

    @staticmethod
    def _normalize_for_compare(spec: FieldSpec, value: str) -> str:
        s = value.strip()
        if spec.kind == "bool":
            return "1" if s.lower() in {"1", "true", "yes", "on"} else "0"
        if spec.kind == "int":
            try:
                return str(int(s))
            except Exception:
                return s
        if spec.kind == "float":
            try:
                return format(float(s), ".12g")
            except Exception:
                return s
        return s

    def _build_start_command(self) -> str:
        env_parts: list[str] = []
        for spec in FIELD_SPECS:
            current = self._var_as_string(spec.name)
            current_norm = self._normalize_for_compare(spec, current)
            default_norm = self._normalize_for_compare(spec, spec.default)
            if current_norm != default_norm:
                env_parts.append(f"{spec.name}={shlex.quote(current)}")
        return (" ".join(env_parts) + " " if env_parts else "") + "dictate-min"

    def _copy_start_command(self) -> None:
        cmd = self._build_start_command()
        copied = False
        if shutil.which("wl-copy"):
            try:
                subprocess.run(["wl-copy"], input=cmd, text=True, check=True)
                copied = True
            except Exception:
                copied = False
        if not copied and shutil.which("xclip"):
            try:
                subprocess.run(["xclip", "-selection", "clipboard"], input=cmd, text=True, check=True)
                copied = True
            except Exception:
                copied = False
        if not copied and shutil.which("xsel"):
            try:
                subprocess.run(["xsel", "--clipboard", "--input"], input=cmd, text=True, check=True)
                copied = True
            except Exception:
                copied = False
        if not copied and shutil.which("pbcopy"):
            try:
                subprocess.run(["pbcopy"], input=cmd, text=True, check=True)
                copied = True
            except Exception:
                copied = False
        if not copied:
            self.clipboard_clear()
            self.clipboard_append(cmd)
            self.update_idletasks()
        self._append_log(f"Copied start command: {cmd}")

    def _start(self) -> None:
        if self._proc and self._proc.poll() is None:
            messagebox.showinfo("dictate-min", "Process is already running.")
            return
        env = os.environ.copy()
        runtime_values = self._env_values(include_empty=False)
        write_env_file(self._runtime_overrides_path, runtime_values)
        env.update(runtime_values)
        env["DICTATE_RUNTIME_ENV_FILE"] = str(self._runtime_overrides_path)
        try:
            self._proc = subprocess.Popen(
                ["dictate-min"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            messagebox.showerror("dictate-min", "Command 'dictate-min' not found. Activate your venv and install package with pip -e .")
            return
        self._append_log(f"Started dictate-min (live overrides: {self._runtime_overrides_path})")
        threading.Thread(target=self._pump_process_output, daemon=True).start()

    def _apply_live(self, quiet: bool = False) -> bool:
        if not self._proc or self._proc.poll() is not None:
            if not quiet:
                self._append_log("No running process. Start dictate-min first.")
            return False
        values = self._env_values(include_empty=False)
        write_env_file(self._runtime_overrides_path, values)
        if not quiet:
            self._append_log(
                f"Applied live update to {self._runtime_overrides_path} "
                "(some settings still require restart)"
            )
        return True

    def _stop(self) -> None:
        if not self._proc or self._proc.poll() is not None:
            self._append_log("No running process to stop.")
            return
        self._proc.terminate()
        self._append_log("Sent terminate signal.")

    def _restart_process(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
        self._proc = None
        self._start()

    def _pump_process_output(self) -> None:
        assert self._proc is not None
        proc = self._proc
        if proc.stdout is not None:
            for line in proc.stdout:
                self._log_queue.put(line.rstrip("\n"))
        rc = proc.wait()
        self._log_queue.put(f"[process exited rc={rc}]")

    def _drain_log(self) -> None:
        while True:
            try:
                line = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)
        self.after(80, self._drain_log)

    def _append_log(self, line: str) -> None:
        s = line.strip().lower()
        info_tokens = (
            "prompt disabled target=",
            "using default cleanup prompt",
        )
        error_tokens = (
            "error:",
            "record start failed",
            "portaudioerror",
            "traceback",
            "exception",
            "http error status=",
            " failed in 'src/hostapi/alsa",
            "model refresh failed",
            "input device refresh failed",
        )
        warn_tokens = (
            "warning",
            "restart required",
        )
        if any(tok in s for tok in info_tokens):
            self._log.insert("end", line + "\n")
        elif any(tok in s for tok in error_tokens):
            self._log.insert("end", line + "\n", ("log_error",))
        elif any(tok in s for tok in warn_tokens):
            self._log.insert("end", line + "\n", ("log_warn",))
        else:
            self._log.insert("end", line + "\n")
        self._log.see("end")

    def _on_var_changed(self, key: str) -> None:
        if self._suspend_auto_apply:
            return
        self._schedule_auto_apply({key})

    def _schedule_auto_apply(self, keys: set[str]) -> None:
        self._pending_changed_keys.update(keys)
        if self._pending_apply_after_id is not None:
            self.after_cancel(self._pending_apply_after_id)
        self._pending_apply_after_id = self.after(450, self._flush_auto_apply)

    def _flush_auto_apply(self) -> None:
        self._pending_apply_after_id = None
        changed = set(self._pending_changed_keys)
        self._pending_changed_keys.clear()
        if not changed:
            return
        if not self._proc or self._proc.poll() is not None:
            return
        needs_restart = bool(changed & RESTART_REQUIRED_ENV)
        if needs_restart and self._auto_restart.get():
            self._append_log(
                f"Auto apply: restart required for {sorted(changed & RESTART_REQUIRED_ENV)}; restarting process."
            )
            self._restart_process()
            return
        applied = self._apply_live(quiet=True)
        if applied:
            if needs_restart:
                self._append_log(
                    f"Auto apply: updated live, but restart required for {sorted(changed & RESTART_REQUIRED_ENV)}"
                )
            else:
                self._append_log(f"Auto apply: applied {sorted(changed)}")

    @staticmethod
    def _normalize_cleanup_url(backend: str, url: str) -> str:
        out = (url or "").strip()
        if backend in {"generic_v1", "api_v1", "lm_api_v1"}:
            parsed = urlparse(out)
            if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
                out = out.rstrip("/") + "/v1/chat"
        return out

    def _refresh_model_choices(self) -> None:
        # STT models: keep built-in list.
        stt_w = self._widgets.get("DICTATE_STT_MODEL")
        stt_spec = self._spec_by_name["DICTATE_STT_MODEL"]
        if isinstance(stt_w, ttk.Combobox):
            stt_w["values"] = list(stt_spec.choices)

        cleanup_w = self._widgets.get("DICTATE_CLEANUP_MODEL")
        if not isinstance(cleanup_w, ttk.Combobox):
            return
        backend = self._var_as_string("DICTATE_CLEANUP_BACKEND").strip().lower()
        url = self._normalize_cleanup_url(backend, self._var_as_string("DICTATE_CLEANUP_URL"))
        token = self._var_as_string("LM_API_TOKEN") or self._var_as_string("DICTATE_CLEANUP_API_TOKEN")
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        models: list[str] = []
        try:
            if backend == "ollama":
                tags_url = url.rstrip("/")
                if tags_url.endswith("/api/chat"):
                    tags_url = tags_url[: -len("/api/chat")] + "/api/tags"
                else:
                    tags_url = tags_url + "/api/tags"
                resp = requests.get(tags_url, headers=headers, timeout=3)
                resp.raise_for_status()
                data = resp.json()
                models = [str(m.get("name", "")).strip() for m in data.get("models", []) if isinstance(m, dict)]
            else:
                base = url.rstrip("/")
                candidates: list[str] = []
                if base.endswith("/api/v1/chat"):
                    root = base[: -len("/api/v1/chat")]
                    candidates = [root + "/api/v1/models", root + "/v1/models"]
                elif base.endswith("/v1/chat"):
                    root = base[: -len("/v1/chat")]
                    candidates = [root + "/v1/models", root + "/api/v1/models"]
                else:
                    candidates = [base + "/api/v1/models", base + "/v1/models"]
                for endpoint in candidates:
                    try:
                        resp = requests.get(endpoint, headers=headers, timeout=3)
                        resp.raise_for_status()
                        data = resp.json()
                    except Exception:
                        continue
                    if isinstance(data, dict):
                        if isinstance(data.get("data"), list):
                            for item in data["data"]:
                                if isinstance(item, dict):
                                    mid = str(item.get("id", "")).strip()
                                    if mid:
                                        models.append(mid)
                        if isinstance(data.get("models"), list):
                            for item in data["models"]:
                                if isinstance(item, dict):
                                    mtype = str(item.get("type", "")).strip().lower()
                                    key = str(item.get("key", "")).strip()
                                    if key and (not mtype or mtype == "llm"):
                                        models.append(key)
                    if models:
                        break
        except Exception as e:
            self._append_log(f"Model refresh failed: {e!r}")

        uniq = sorted({m for m in models if m})
        cleanup_w["values"] = uniq
        self._append_log(f"Loaded {len(uniq)} cleanup model(s) for backend={backend}")

    def _refresh_input_devices(self) -> None:
        id_widget = self._widgets.get("DICTATE_INPUT_DEVICE")
        name_widget = self._widgets.get("DICTATE_INPUT_DEVICE_NAME")
        if not isinstance(id_widget, ttk.Combobox) or not isinstance(name_widget, ttk.Combobox):
            return
        try:
            proc = subprocess.run(
                ["dictate-min", "--list-input-devices"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=6,
            )
        except Exception as e:
            self._append_log(f"Input device refresh failed: {e!r}")
            return
        devices: list[tuple[str, str]] = []
        for line in proc.stdout.splitlines():
            s = line.rstrip()
            if not s.strip():
                continue
            parts = s.strip().split(maxsplit=1)
            if not parts or not parts[0].isdigit():
                continue
            idx = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            devices.append((idx, name))
        self._input_devices = devices
        id_widget["values"] = [idx for idx, _ in devices]
        name_widget["values"] = [name for _, name in devices]
        self._append_log(f"Loaded {len(devices)} input device(s)")


def main() -> int:
    app = DictateGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
