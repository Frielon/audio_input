"""Settings: built-in defaults, an optional ``config.toml`` and command-line overrides.

Every field of :class:`Settings` can be set in ``config.toml`` at the project
root (see ``config.example.toml``); a few common ones also have command-line
flags (see ``python -m audio_input --help``).
"""
from __future__ import annotations

import logging
import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("AUDIO_INPUT_CONFIG", PROJECT_ROOT / "config.toml"))
DATA_DIR = Path(os.environ.get("AUDIO_INPUT_DATA_DIR", PROJECT_ROOT / "data"))

CHOICES = {
    "insert": ("paste", "type", "none"),
    "insert_text": ("transcript", "corrected"),
    "whisper_backend": ("auto", "mlx", "faster"),
    "ipa": ("all", "errors", "off"),
}


@dataclass
class Settings:
    # push-to-talk
    hotkey: str = "left_option"       # see hotkey.KEYS: left_option, right_option, right_command, fn, f13, ...
    start_delay: float = 0.25         # the key must be held alone this long before the microphone opens, so
                                      # that the key keeps working in shortcuts (Option+←, Option-click …)
    min_seconds: float = 0.5          # shorter holds are taken as an accidental tap
    max_seconds: float = 120.0        # recording stops by itself after this long
    input_device: Optional[str] = None  # microphone name (or part of it) or index; None = system default

    # putting the text where the cursor is
    insert: str = "paste"             # paste (clipboard + Cmd+V, clipboard restored) | type (key events) | none
    insert_text: str = "transcript"   # what you said, or the grammar-corrected version
    restore_clipboard: bool = True
    smart_spacing: bool = True        # add a space when the cursor is right after a word
    keep_fillers: bool = False        # False: "um", "uh" … are left out of the inserted text (still shown)

    # speech recognition
    whisper_backend: str = "auto"     # mlx (Apple GPU) | faster (faster-whisper, CPU) | auto
    mlx_model: str = "mlx-community/whisper-large-v3-turbo"
    whisper_model: str = "small.en"   # faster-whisper model
    beam_size: int = 5                # faster-whisper only
    vocabulary: list[str] = field(default_factory=list)  # names and terms to help recognition
    # Whisper copies the style of its prompt: a spoken, imperfect one keeps your grammar and filler words
    # as they were said (instead of "fixing" them), so the feedback sees what you really said.
    whisper_prompt: str = "Um, so, uh, yesterday I go to the lab and my friend say the code are broken, so we fix it."

    # feedback
    phonemes: bool = True             # sounds (IPA), stress and the coach (the bundled Parrot Shadowing analysis)
    # the coach's areas you work on: listed first, with the teacher's tips (th, final, sz, r, uh, stress,
    # rhythm, pause, focus)
    focus: list[str] = field(default_factory=lambda: ["final", "th", "r", "sz", "stress"])
    model_voice: bool = True          # a macOS voice reads the corrected sentence: ▶ model / ⇄ compare, its stress
    tts_voice: str = "Samantha"       # any English voice of `say -v '?'`
    grammar: bool = True
    grammar_model: str = "grammarly/coedit-large"
    grammar_device: str = "auto"      # auto (Apple GPU when available) | mps | cpu

    # overlay
    overlay: bool = True
    overlay_seconds: float = 0.0      # 0: the overlay stays on screen (close it with ✕); else hide after this long
    idle_opacity: float = 0.35        # opacity while the mouse is not over it (1 = never fades)
    attention_seconds: float = 4.0    # a new result stays fully visible this long (0 = only the mouse decides)
    overlay_width: int = 960          # maximal width in points
    font_size: int = 28
    ipa: str = "all"                  # IPA under all words, only under words with sound errors, or off
    click_through: bool = False       # True: the overlay ignores the mouse (cannot be dragged)

    # storage
    recent: int = 10                  # recordings kept (data/recent) to replay and compare in the overlay
    history: bool = True              # data/history.jsonl: every utterance and its feedback
    keep_audio: bool = False          # also keep the recordings under data/recordings

    def validate(self) -> "Settings":
        for name, allowed in CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {', '.join(allowed)} (got {getattr(self, name)!r})")
        if self.min_seconds < 0 or self.max_seconds <= self.min_seconds:
            raise ValueError("need 0 <= min_seconds < max_seconds")
        if not 0.05 <= self.idle_opacity <= 1:
            raise ValueError("idle_opacity must be between 0.05 and 1")
        if not 0 <= self.start_delay <= 2:
            raise ValueError("start_delay must be between 0 and 2 seconds")
        from .feedback import CATEGORIES

        unknown = [c for c in self.focus if c not in CATEGORIES]
        if unknown:
            raise ValueError(f"unknown focus area(s) {unknown}; use {', '.join(CATEGORIES)}")
        return self


def _coerce(name: str, value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("1", "true", "yes", "on", "0", "false", "no", "off"):
            return value.lower() in ("1", "true", "yes", "on")
        raise ValueError(f"{name} must be true or false")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return [str(v) for v in value]
    if name == "input_device" and value is not None:
        return str(value)
    return value


def save_hotkey(name: str, path: Optional[Path] = None) -> Path:
    """Set ``hotkey`` in config.toml (created if needed), keeping everything else in it."""
    path = Path(path) if path else CONFIG_PATH
    line = f'hotkey = "{name}"'
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?m)^\s*hotkey\s*=.*$", text):
            text = re.sub(r"(?m)^\s*hotkey\s*=.*$", line, text, count=1)
        else:
            text = line + "\n" + text
    else:
        text = "# audio_input settings; all of them are listed in config.example.toml\n" + line + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def load(path: Optional[Path] = None, overrides: Optional[dict] = None) -> Settings:
    """Defaults, then the TOML file (if it exists), then ``overrides`` (None values are skipped)."""
    settings = Settings()
    known = {f.name: f for f in fields(Settings)}
    path = Path(path) if path else CONFIG_PATH
    values: dict = {}
    if path.exists():
        with open(path, "rb") as fh:
            values.update(tomllib.load(fh))
    elif path != CONFIG_PATH:
        raise FileNotFoundError(path)
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key, value in values.items():
        if key not in known:
            log.warning("Unknown setting %r in %s (ignored)", key, path)
            continue
        setattr(settings, key, _coerce(key, value, getattr(settings, key)))
    return settings.validate()
