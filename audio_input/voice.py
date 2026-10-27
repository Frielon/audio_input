"""The model voice: a macOS voice (``say``) reads the sentence, as for Parrot Shadowing's practice texts.

It reads the corrected sentence (without filler words), so you hear both the
grammar and the pronunciation to aim for.  Readings are cached on disk.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .history import write_wav
from .recent import read_wav

log = logging.getLogger(__name__)

SR = 16000
MAX_CACHE = 400
_VOICE_RE = re.compile(r"^(?P<name>.+?)\s{2,}(?P<lang>[a-z]{2}_[A-Z]{2})\s+#")


def available() -> bool:
    return shutil.which("say") is not None


@lru_cache(maxsize=1)
def english_voices() -> tuple[str, ...]:
    if not available():
        return ()
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # pragma: no cover - system dependent
        return ()
    return tuple(m.group("name").strip() for m in map(_VOICE_RE.match, out.splitlines())
                 if m and m.group("lang").startswith("en_"))


def trim_silence(audio: np.ndarray, margin: float = 0.05, floor_db: float = 45.0) -> np.ndarray:
    if not len(audio):
        return audio
    peak = float(np.abs(audio).max())
    if peak <= 0:
        return audio[:0]
    loud = np.where(np.abs(audio) > peak * 10 ** (-floor_db / 20))[0]
    m = int(margin * SR)
    return audio[max(0, loud[0] - m):min(len(audio), loud[-1] + m)]


class Voice:
    def __init__(self, cache_dir: Path, voice: Optional[str] = "Samantha"):
        self.cache_dir = Path(cache_dir)
        self.voice = voice
        self._lock = threading.Lock()

    @property
    def name(self) -> Optional[str]:
        voices = english_voices()
        if self.voice in voices:
            return self.voice
        return "Samantha" if "Samantha" in voices else (voices[0] if voices else None)

    def speak(self, text: str) -> np.ndarray:
        """``text`` read by the voice: 16 kHz mono float32 without the silence around it."""
        text = re.sub(r"\s+", " ", text).strip()
        voice = self.name
        key = hashlib.sha1(f"{voice}\n{text}".encode()).hexdigest()[:20]
        path = self.cache_dir / f"{key}.wav"
        if path.exists():
            return read_wav(path)
        from faster_whisper import decode_audio

        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "text.txt"), os.path.join(tmp, "speech.aiff")
            with open(src, "w", encoding="utf-8") as fh:
                fh.write(text)  # from a file, so the text can never be taken for an option
            subprocess.run(["say", "-o", out, "-f", src] + (["-v", voice] if voice else []),
                           check=True, capture_output=True, timeout=60)
            audio = trim_silence(np.asarray(decode_audio(out, sampling_rate=SR), dtype=np.float32))
        with self._lock:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            write_wav(path, audio)
            files = sorted(self.cache_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
            for old in files[:-MAX_CACHE]:
                old.unlink(missing_ok=True)
        return audio
