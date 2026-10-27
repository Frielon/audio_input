"""Rhythm, pauses and focus words of one recording.

* **Word timing** comes from the phone alignment (more precise than Whisper's
  word timestamps) and is cut at the pauses found in the loudness envelope.
* **Rhythm**: English gives stressed syllables of content words much more time
  than function words.  The *function-word ratio* is the median time per
  syllable of the one-syllable function words divided by that of the content
  words: about 0.65-0.7 for native speech and synthetic voices, about 1.1 when
  every word is said on its own (measured on 100 native and 54 synthetic
  sentences).  Function words said with their full, strong vowel in the middle
  of a phrase (``and`` as ``ænd`` instead of ``ən``) are listed.
* **Pauses**: silences of 180 ms or more, attached to the word before them.
* **Pitch range** in semitones: a flat voice cannot make focus words stand out.
  (Which word carries the focus is not judged: on synthetic speech with a
  known emphasis no measure tried found it reliably, so the app marks the
  likely focus word and shows the pitch line instead.)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import english
from .acoustics import Acoustics
from .phoneme_engine import is_vowel
from .phoneme_scoring import SentenceAlignment, edit_cost

FORM_MARGIN = 0.5  # how much better one form has to fit to call it weak / strong


def form_kind(word: str, heard: Sequence[str]) -> Optional[str]:
    """``weak`` / ``strong`` when the heard phones clearly fit one form of a function word."""
    forms = english.weak_strong(word)
    if not forms or not heard:
        return None
    weak = min(edit_cost(f, heard) for f in forms[0])
    strong = min(edit_cost(f, heard) for f in forms[1])
    if weak + FORM_MARGIN <= strong:
        return "weak"
    if strong + FORM_MARGIN <= weak:
        return "strong"
    return None


def analyze(words: Sequence[str], al: SentenceAlignment, ac: Acoustics,
            stress: Optional[Sequence[Sequence[int]]] = None) -> dict:
    n = len(words)
    pauses = ac.pauses()
    content = [not english.is_function_word(w, _stressed(stress, i)) for i, w in enumerate(words)]
    heard_texts = [h.text for h in al.heard]

    spans: list[Optional[tuple[float, float]]] = []
    for i in range(n):
        sp = al.word_time(i)
        if sp is not None:
            start, end = sp
            for ps, _ in pauses:  # a word does not run into the silence after it
                if start < ps < end:
                    end = ps
                    break
            sp = (start, max(end, start + 0.04))
        spans.append(sp)

    pause_after: list[Optional[float]] = [None] * n
    out_pauses = []
    for ps, pe in pauses:
        before = [i for i in range(n) if spans[i] and spans[i][0] < ps]
        after = [i for i in range(n) if spans[i] and spans[i][0] >= pe - 0.02]
        if not before or not after:
            continue  # silence before the first or after the last word
        i = before[-1]
        if after[0] != i + 1 and any(spans[k] is None for k in range(i + 1, after[0])):
            i = after[0] - 1  # unheard words in between: put the pause before the next heard word
        pause_after[i] = round((pause_after[i] or 0.0) + (pe - ps), 2)
        out_pauses.append({"after": i, "start": ps, "end": pe})

    out_words = []
    for i, w in enumerate(words):
        wa = al.words[i]
        syl = max(1, sum(1 for p in wa.expected if is_vowel(p)))
        entry: dict = {"content": content[i], "syllables": syl}
        if english.weak_strong(w) is not None and wa.hi > wa.lo:
            kind = form_kind(w, heard_texts[wa.lo:wa.hi])
            if kind:
                entry["form"] = kind
        sp = spans[i]
        if sp is not None:
            start, end = sp
            entry.update({"start": round(start, 3), "end": round(end, 3), "dps": round((end - start) / syl, 3)})
        if pause_after[i]:
            entry["pause_after"] = pause_after[i]
        out_words.append(entry)

    fn = [e["dps"] for i, e in enumerate(out_words) if "dps" in e and not content[i] and e["syllables"] == 1]
    cw = [e["dps"] for i, e in enumerate(out_words) if "dps" in e and content[i]]
    ratio = round(float(np.median(fn) / np.median(cw)), 2) if len(fn) >= 2 and len(cw) >= 2 else None
    timed = [s for s in spans if s]
    bounds = (timed[0][0], timed[-1][1]) if timed else (0.0, ac.duration)
    return {
        "words": out_words,
        "pauses": out_pauses,
        "function_ratio": ratio,
        "pitch_range": ac.pitch_range(),
        "speech": [round(bounds[0], 3), round(bounds[1], 3)],
        "contour": ac.contour(max(0.0, bounds[0] - 0.15), min(ac.duration, bounds[1] + 0.15)),
    }


def _stressed(stress: Optional[Sequence[Sequence[int]]], i: int) -> bool:
    return bool(stress and i < len(stress) and 1 in stress[i])
