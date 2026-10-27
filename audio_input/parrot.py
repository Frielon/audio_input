"""The pronunciation analysis of Parrot Shadowing, used as a library.

Parrot Shadowing's backend modules are bundled in :mod:`audio_input.parrot_backend`
(``scripts/sync_parrot.sh`` refreshes them): the phoneme recognizer
(wav2vec2 -> espeak IPA), the alignment of heard and expected sounds, the
stress/rhythm measurements and the teacher's checklist (``coach``).

Dictation has no target sentence, so the transcript stands in for it: the
recognizer's sounds are compared with the expected sounds of the words
Whisper understood, i.e. with what you meant to say.
"""
from __future__ import annotations

import logging
import threading
from types import ModuleType
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# The speaker pauses nowhere: punctuation in a transcript comes from Whisper, which puts commas
# where *you* paused, so "no pause at the comma" checks would be circular.  Pauses that split a
# phrase ("the ‖ root") are still reported.
_DICTATION_REFERENCE = {"pauses_after": []}


class Pronunciation:
    """Sounds (IPA), stress, rhythm and the coach's checks for one utterance."""

    def __init__(self):
        self.backend: Optional[ModuleType] = None
        self.error: Optional[str] = None
        self.loaded = False
        self._lock = threading.Lock()

    def _import(self) -> ModuleType:
        if self.backend is not None:
            return self.backend
        from . import parrot_backend
        from .parrot_backend import analysis, english, phoneme_engine, scoring  # noqa: F401

        if not phoneme_engine.available():
            raise RuntimeError(f"phoneme stack unavailable: {phoneme_engine.unavailable_reason()}")
        self.backend = parrot_backend
        return self.backend

    def available(self) -> bool:
        if self.backend is not None:
            return True
        if self.error:
            return False
        try:
            self._import()
            return True
        except Exception as exc:
            self.error = str(exc)
            log.warning("Pronunciation feedback unavailable: %s", exc)
            return False

    def load(self) -> None:
        """Load the phoneme model and espeak, and run one inference (the first one is slow)."""
        with self._lock:
            if self.loaded:
                return
            backend = self._import()
            backend.phoneme_engine.get_phonemizer()
            backend.phoneme_engine.get_model()
            backend.phoneme_engine.recognize(np.zeros(16000, dtype=np.float32))
            self.loaded = True

    def display_words(self, text: str) -> list[str]:
        return self._import().scoring.display_words(text)

    def analyze(self, text: str, audio: np.ndarray, words: Sequence) -> dict:
        """``words`` are the recognised words with ``text``, ``start``, ``end`` and ``probability``.

        Returns parrot's ``{"phonemes", "prosody", "coach"}`` for the display
        words of ``text`` (see ``backend.analysis.analyze_take``), plus the
        ``syllables`` of every word (phone index ranges, as parrot draws them).
        """
        backend = self._import()
        scored = backend.scoring.score_attempt(text, list(words))
        spans = [(w.start, w.end) if w.start is not None and w.end is not None else None for w in scored.words]
        out = backend.analysis.analyze_take(text, np.asarray(audio, dtype=np.float32), spans,
                                            dict(_DICTATION_REFERENCE))
        out["syllables"] = [backend.english.syllabify(w["expected"]) for w in out["phonemes"]["words"]]
        return out

    def analyze_reference(self, text: str, audio: np.ndarray) -> dict:
        """What a model reading of ``text`` does: word times, pauses, rhythm, pitch, and the stress of each word.

        The same analysis parrot runs on the speaker of a video or the voice of a
        practice text; without word times, the phones are cut into words by
        alignment, which is exact enough for a synthetic voice.
        """
        backend = self._import()
        return backend.analysis.analyze_reference(text, np.asarray(audio, dtype=np.float32), None)
