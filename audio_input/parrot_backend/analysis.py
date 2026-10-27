"""One recording -> sound, stress, rhythm, pause and coach feedback.

Runs the optional phoneme stack (see :mod:`backend.phoneme_engine`): the
recording is transcribed into phones once, aligned with the sentence once, and
the alignment and the acoustic measurements are shared by every analysis.
The same steps, without the coach, describe the *speaker's* recording of the
sentence, which the learner's take is compared with.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import acoustics, coach, english, phoneme_engine, phoneme_scoring, prosody
from .phoneme_scoring import Span
from .scoring import display_words


def _measure(text: str, audio: np.ndarray, spans: Optional[Sequence[Span]]):
    words = display_words(text)
    expected, stress = phoneme_engine.phonemize_sentence_stress(words)
    heard = phoneme_engine.recognize(audio)
    al = phoneme_scoring.align_sentence(words, expected, heard, spans,
                                        forms=[english.alternative_forms(w) for w in words])
    ac = acoustics.analyze(audio)
    ph = phoneme_scoring.score_phonemes(words, expected, heard, spans=spans, stress=stress,
                                        alignment=al, acoustics=ac)
    pr = prosody.analyze(words, al, ac, stress)
    return words, expected, stress, al, ph, pr


def analyze_take(text: str, audio: np.ndarray, spans: Optional[Sequence[Span]] = None,
                 reference: Optional[dict] = None) -> dict:
    """Feedback on the learner's recording; ``reference`` is :func:`analyze_reference` of the speaker."""
    words, expected, stress, al, ph, pr = _measure(text, audio, spans)
    return {"phonemes": ph, "prosody": pr,
            "coach": coach.coach(words, al, ph, pr, reference, dictionary=expected)}


def analyze_reference(text: str, audio: np.ndarray, spans: Optional[Sequence[Span]] = None) -> dict:
    """What the speaker does in the sentence: pauses, rhythm, pitch, word timing and stress pattern."""
    _, _, _, _, ph, pr = _measure(text, audio, spans)
    return {
        "pauses_after": sorted({p["after"] for p in pr["pauses"]}),
        "function_ratio": pr["function_ratio"],
        "pitch_range": pr["pitch_range"],
        "speech": pr["speech"],
        "contour": pr["contour"],
        "words": [{k: w[k] for k in ("start", "end", "pause_after") if k in w} for w in pr["words"]],
        "stress": {str(wi): w["stress"]["prominence"] for wi, w in enumerate(ph["words"])
                   if w.get("stress") and w["stress"].get("prominence")},
    }
