"""The last recordings (10 by default), kept so you can replay them, compare and look at the feedback again.

Every utterance is stored in ``data/recent`` as ``<id>.wav`` (the recording the
feedback refers to), ``<id>.json`` (the overlay's data) and, once synthesized,
``<id>.model.wav`` (the model voice reading the sentence).  Older ones are
deleted as new ones come in, and the list survives a restart.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from .history import write_wav

log = logging.getLogger(__name__)

SR = 16000


def read_wav(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


class Recent:
    def __init__(self, folder: Path, keep: int = 10):
        self.folder = Path(folder)
        self.keep = keep
        self._lock = threading.Lock()
        self._audio: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def ids(self) -> list[str]:
        """Oldest first."""
        if not self.folder.exists():
            return []
        return sorted(p.stem for p in self.folder.glob("*.json"))

    def position(self, rid: str) -> tuple[int, int]:
        """``(1-based index, total)`` of a recording (the newest is ``total``)."""
        ids = self.ids()
        return (ids.index(rid) + 1 if rid in ids else 0), len(ids)

    def neighbour(self, rid: str, step: int) -> Optional[str]:
        ids = self.ids()
        if rid not in ids:
            return ids[-1] if ids else None
        k = ids.index(rid) + step
        return ids[k] if 0 <= k < len(ids) else None

    def add(self, payload: dict, audio: np.ndarray) -> Optional[str]:
        if self.keep <= 0:
            return None
        with self._lock:
            self.folder.mkdir(parents=True, exist_ok=True)
            rid = time.strftime("%Y%m%d-%H%M%S")
            n = 1
            while (self.folder / f"{rid}.json").exists():
                rid = f"{time.strftime('%Y%m%d-%H%M%S')}-{n}"
                n += 1
            write_wav(self.folder / f"{rid}.wav", audio)
            self._write(rid, payload)
            self._audio[rid] = np.asarray(audio, dtype=np.float32)
            self._trim_cache()
            for old in self.ids()[:-self.keep]:
                for path in self.folder.glob(f"{old}.*"):
                    path.unlink(missing_ok=True)
                self._audio.pop(old, None)
        return rid

    def _write(self, rid: str, payload: dict) -> None:
        tmp = self.folder / f".{rid}.json.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.folder / f"{rid}.json")

    def save(self, rid: str, payload: dict) -> None:
        with self._lock:
            if (self.folder / f"{rid}.json").exists():
                self._write(rid, payload)

    def load(self, rid: str) -> Optional[dict]:
        try:
            return json.loads((self.folder / f"{rid}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _trim_cache(self) -> None:
        while len(self._audio) > 4:
            self._audio.popitem(last=False)

    def _cached(self, key: str, path: Path) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._audio:
                self._audio.move_to_end(key)
                return self._audio[key]
        if not path.exists():
            return None
        audio = read_wav(path)
        with self._lock:
            self._audio[key] = audio
            self._trim_cache()
        return audio

    def audio(self, rid: str) -> Optional[np.ndarray]:
        return self._cached(rid, self.folder / f"{rid}.wav")

    def model_audio(self, rid: str) -> Optional[np.ndarray]:
        return self._cached(f"{rid}.model", self.folder / f"{rid}.model.wav")

    def save_model_audio(self, rid: str, audio: np.ndarray) -> None:
        with self._lock:
            if (self.folder / f"{rid}.json").exists():
                write_wav(self.folder / f"{rid}.model.wav", audio)
                self._audio[f"{rid}.model"] = np.asarray(audio, dtype=np.float32)
                self._trim_cache()
