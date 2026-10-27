"""Lexical stress check: which syllable of a multi-syllable word was most prominent?

The phoneme recognizer cannot label stress (it is prosody, not a segment), so the
stressed syllable is measured from the recording itself.  Every syllable gets a
region around its vowel (from the recognizer's phone timing); inside it the
loudness peak and the *nucleus* (the frames within 6 dB of the peak) are found
and these cues are measured:

* loudness: peak level, energy of the nucleus, level of the 1-4 kHz band and
  the spectral tilt (stressed vowels gain most in the upper band);
* duration of the nucleus (shortened for a word-final syllable before a pause,
  which is lengthened in any case);
* pitch: median and maximum F0 of the nucleus;
* vowel quality: a reduced vowel (ə, ɪ-like) is rarely stressed.

The cues are z-scored within the word and combined with weights fitted on native
speech (a YouTube speaker and three TTS voices, 640 words; cross-validated
agreement with the dictionary stress: 82% in fast connected speech, 91% for
words said on their own).  A verdict is only given when the winning syllable is
ahead by a clear margin; a "wrong stress" verdict needs a larger one, and an
even larger one when the syllable heard as stressed carries secondary stress.
On native speech that keeps false "wrong stress" verdicts at 2-4% for words of
two or three syllables.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .acoustics import ENV_HOP, Acoustics
from .phoneme_engine import Phone, is_vowel

REDUCED_VOWELS = frozenset({"ə", "ɐ", "ᵻ", "ɚ", "i", "əl", "n̩", "l̩"})
WEIGHTS = {"peak": 0.245, "hi": 0.273, "energy": 0.271, "nuc": 0.324, "f0": 0.214, "f0max": 0.197,
           "tilt": 0.306, "full": 0.98, "last_before_pause": -0.107, "last": -0.51}
OK_MARGIN = 0.5           # expected syllable on top by at least this much: stress ✓
WRONG_MARGIN = 1.5        # another syllable on top by at least this much: stress ✗
WRONG_MARGIN_LONG = 2.0   # ... for words of four or more syllables
WRONG_MARGIN_SECONDARY = 3.0  # ... when that syllable has secondary stress (ˌ)
FINAL_LENGTHENING = 0.65  # a syllable before a pause is longer anyway


def _z(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    sd = arr.std()
    return (arr - arr.mean()) / sd if sd > 1e-9 else np.zeros_like(arr)


def syllable_cues(ac: Acoustics, region: tuple[float, float]) -> Optional[dict]:
    """Prominence cues of the syllable whose vowel lies in ``region``."""
    m = ac.env_mask(*region)
    if not m.any():
        return None
    db, hi, tilt, t = ac.db[m], ac.hi_db[m], ac.tilt[m], ac.t[m]
    k = int(np.argmax(db))
    peak = float(db[k])
    lo = hi_i = k
    while lo > 0 and db[lo - 1] >= peak - 6:
        lo -= 1
    while hi_i < len(db) - 1 and db[hi_i + 1] >= peak - 6:
        hi_i += 1
    f0 = ac.f0_in(t[lo] - 0.01, t[hi_i] + 0.01)
    return {
        "peak": peak,
        "hi": float(hi.max()),
        "tilt": float(tilt[lo:hi_i + 1].mean()),
        "energy": float(10 * math.log10(np.sum(10 ** (db[lo:hi_i + 1] / 10)) * ENV_HOP + 1e-12)),
        "nuc": (hi_i - lo + 1) * ENV_HOP,
        "f0": float(np.median(f0)) if len(f0) else None,
        "f0max": float(np.max(f0)) if len(f0) else None,
    }


def prominence_scores(cues: Sequence[dict], vowels: Sequence[str], pause_after: bool) -> np.ndarray:
    """Weighted sum of the within-word z-scores of every cue (higher = more prominent)."""
    n = len(cues)
    last = np.array([1.0 if k == n - 1 else 0.0 for k in range(n)])
    nuc = [max(c["nuc"], 0.01) * (FINAL_LENGTHENING if (k == n - 1 and pause_after) else 1.0)
           for k, c in enumerate(cues)]
    feats = {
        "peak": _z([c["peak"] for c in cues]),
        "hi": _z([c["hi"] for c in cues]),
        "energy": _z([c["energy"] for c in cues]),
        "tilt": _z([c["tilt"] for c in cues]),
        "nuc": _z(np.log(nuc)),
        "full": np.array([0.0 if v in REDUCED_VOWELS else 1.0 for v in vowels]),
        "last": last,
        "last_before_pause": last * (1.0 if pause_after else 0.0),
    }
    for key in ("f0", "f0max"):
        known = [c[key] for c in cues if c[key]]
        if len(known) >= 2:
            ref = float(np.median(known))
            feats[key] = _z([12 * math.log2(c[key] / ref) if c[key] else 0.0 for c in cues])
        else:
            feats[key] = np.zeros(n)
    return sum(w * feats[k] for k, w in WEIGHTS.items())


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def analyze_word(expected: Sequence[str], stress: Sequence[int], heard_map: dict[int, int],
                 heard: Sequence[Phone], acoustics: Acoustics, word_time: Optional[tuple[float, float]] = None,
                 pause_after: bool = False) -> Optional[dict]:
    """Stress verdict for one word.

    ``heard_map`` maps an expected-phone index to the index (in ``heard``) of the
    sound the learner produced for it; ``word_time`` is the word's span in the
    recording.  Returns None for words without an expected primary stress or
    with fewer than two syllables.  ``prominence`` holds the share of
    prominence of every syllable that was heard (for drawing bars).
    """
    vowels = [i for i, p in enumerate(expected) if is_vowel(p)]
    if len(vowels) < 2:
        return None
    # In sentence context espeak demotes some words to secondary stress (ˌ); that
    # syllable is still the one the learner should make prominent.
    primary = [k for k, i in enumerate(vowels) if i < len(stress) and stress[i] == 1] or \
              [k for k, i in enumerate(vowels) if i < len(stress) and stress[i] == 2]
    if not primary:
        return None
    k_exp = primary[0]
    secondary = [k for k, i in enumerate(vowels) if i < len(stress) and stress[i] == 2 and k != k_exp]
    base = {"syllables": len(vowels), "expected": k_exp, "secondary": secondary}
    present = [(k, heard_map[i]) for k, i in enumerate(vowels) if i in heard_map]
    if k_exp not in {k for k, _ in present}:
        return {**base, "detected": None, "match": None, "note": "the stressed vowel was not heard"}
    if len(present) < 2:
        return {**base, "detected": None, "match": None, "note": "too few syllables heard"}

    onsets = [heard[h].start for _, h in present]
    w_start = (word_time[0] if word_time else onsets[0] - 0.05) - 0.03
    w_end = word_time[1] if word_time else heard[present[-1][1]].end + 0.25
    bounds = [w_start] + [(onsets[j] + onsets[j + 1]) / 2 for j in range(len(onsets) - 1)] + [max(w_end, onsets[-1] + 0.06)]
    cues = [syllable_cues(acoustics, (max(bounds[j], onsets[j] - 0.04), bounds[j + 1])) for j in range(len(onsets))]
    if any(c is None for c in cues):
        return {**base, "detected": None, "match": None, "note": "unclear"}
    scores = prominence_scores(cues, [heard[h].text for _, h in present], pause_after)
    order = np.argsort(-scores, kind="stable")
    margin = float(scores[order[0]] - scores[order[1]])
    share = _softmax(scores)
    prominence: list[Optional[float]] = [None] * len(vowels)
    for j, (k, _) in enumerate(present):
        prominence[k] = round(float(share[j]), 2)
    detected = present[int(order[0])][0]
    base["prominence"] = prominence
    base["margin"] = round(margin, 2)
    if detected == k_exp:
        ok = margin >= OK_MARGIN
        return {**base, "detected": detected if ok else None, "match": True if ok else None,
                "note": None if ok else "unclear"}
    need = WRONG_MARGIN_SECONDARY if detected in secondary else \
        WRONG_MARGIN_LONG if len(vowels) >= 4 else WRONG_MARGIN
    if margin >= need:
        return {**base, "detected": detected, "match": False, "note": None}
    return {**base, "detected": None, "match": None, "note": "unclear"}
