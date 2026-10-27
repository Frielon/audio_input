"""Playback of recordings and the model voice (one sound at a time, on the default output device)."""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

SR = 16000
PAD_BEFORE, PAD_AFTER = 0.06, 0.1  # around a single word: its edges are soft


def segment(audio: np.ndarray, start: float, end: float, sr: int = SR) -> np.ndarray:
    a = max(0, int((start - PAD_BEFORE) * sr))
    b = min(len(audio), int((end + PAD_AFTER) * sr))
    return audio[a:b] if b > a else audio[:0]


def join(parts: Sequence[np.ndarray], gap: float = 0.45, sr: int = SR) -> np.ndarray:
    """Sounds one after the other with a short silence in between ("compare": the model, then you)."""
    silence = np.zeros(int(gap * sr), dtype=np.float32)
    out: list[np.ndarray] = []
    for k, p in enumerate(p for p in parts if p is not None and len(p)):
        if k:
            out.append(silence)
        out.append(np.asarray(p, dtype=np.float32))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def sound_for(act: str, words: Sequence[dict], me: Optional[np.ndarray],
              model: Callable[[], Optional[np.ndarray]],
              speak: Optional[Callable[[str], np.ndarray]] = None) -> Optional[np.ndarray]:
    """The sound a click in the overlay asks for.

    ``play:me`` your recording, ``play:model`` the model voice, ``play:compare``
    the model then you, ``w:<i>`` you saying word i, ``m:<i>`` the model saying
    it (cut from its reading, or spoken on its own when the model reads another
    word there), ``t:<start>:<end>`` you at the spot of a coach remark.
    ``model`` is called only when the model voice is needed.
    """
    if act == "play:me":
        return me
    if act == "play:model":
        return model()
    if act == "play:compare":
        return join([model(), me])
    kind, _, rest = act.partition(":")
    if kind == "w":
        w = words[int(rest)]
        return segment(me, *w["t"]) if me is not None and w.get("t") else None
    if kind == "m":
        w = words[int(rest)]
        reading = model() if w.get("mt") else None
        if reading is not None:
            return segment(reading, *w["mt"])
        text = w["w"].strip(".,!?;:\"'()“”‘’")
        return speak(text) if speak and text else None
    if kind == "t" and me is not None:
        a, b = rest.split(":")
        return segment(me, float(a), float(b))
    return None


class Player:
    def __init__(self):
        self._lock = threading.Lock()
        self.error: Optional[str] = None

    def play(self, audio: Optional[np.ndarray], sr: int = SR) -> bool:
        if audio is None or not len(audio):
            return False
        import sounddevice as sd

        with self._lock:
            try:
                sd.stop()
                peak = float(np.abs(audio).max())
                gain = min(4.0, 0.7 / peak) if peak > 0 else 1.0  # quiet recordings are played louder
                sd.play(np.asarray(audio, dtype=np.float32) * gain, sr)
                return True
            except Exception as exc:  # no output device, …
                self.error = str(exc)
                log.warning("playback failed: %s", exc)
                return False

    def stop(self) -> None:
        import sounddevice as sd

        with self._lock:
            try:
                sd.stop()
            except Exception:
                pass
