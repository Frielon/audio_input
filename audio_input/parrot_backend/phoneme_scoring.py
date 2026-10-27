"""Compare the phones heard in a recording with the expected phones of the sentence.

The heard phones are a single time-ordered sequence.  Scoring cuts that
sequence into slots: one slot per word plus a "gap" slot before, between and
after the words for surplus sounds.  The initial cuts come from Whisper's word
time spans when they are available (so a wrong word such as ``cook`` for
``write`` is compared with its own sounds and the sounds of extra words are not
blamed on the last word), or from a global edit-distance alignment otherwise.
Each boundary is then refined by moving a few phones left or right whenever
that lowers the total edit cost, which repairs the ~100 ms imprecision of
Whisper's word timestamps.

Function words may be compared with one of several accepted pronunciations
(``and`` as ``æ n d`` or the reduced ``ə n``): reducing them is what fluent
speakers do, so it must not count as an error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .phoneme_engine import Phone, is_vowel

GAP_COST = 0.5  # a surplus sound between words costs less than an insertion inside a word

# Sounds that are free variants of each other in (American) English: a flapped t/d in
# "water", a glottal stop for t in "button", schwa and barred-i variants of reduced vowels,
# and two spellings the recognizer uses for the same diphthong / r-coloured vowel.
_ALLOPHONES = {frozenset(p) for p in (("t", "ɾ"), ("d", "ɾ"), ("t", "ʔ"), ("ə", "ɐ"), ("ɪ", "ᵻ"), ("ə", "ᵻ"),
                                      ("ɔɪ", "oɪ"), ("ɔːɹ", "oːɹ"))}


def same_sound(a: str, b: str) -> bool:
    return a == b or frozenset((a, b)) in _ALLOPHONES


def _sub_cost(a: str, b: str) -> float:
    if same_sound(a, b):
        return 0.0
    # Substituting within the same class (vowel/consonant) is the likelier confusion,
    # so the aligner prefers it when several alignments have the same edit count.
    return 0.9 if is_vowel(a) == is_vowel(b) else 1.0


def _dp(expected: Sequence[str], heard: Sequence[str]) -> list[list[float]]:
    n, m = len(expected), len(heard)
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = float(i)
    for j in range(1, m + 1):
        cost[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + _sub_cost(expected[i - 1], heard[j - 1]),
                cost[i - 1][j] + 1.0,
                cost[i][j - 1] + 1.0,
            )
    return cost


def edit_cost(expected: Sequence[str], heard: Sequence[str]) -> float:
    return _dp(expected, heard)[len(expected)][len(heard)]


def align(expected: Sequence[str], heard: Sequence[str]) -> list[tuple[str, Optional[int], Optional[int]]]:
    """Levenshtein alignment; returns ops ``(kind, expected_index, heard_index)``.

    ``kind`` is ``ok``, ``sub``, ``del`` (expected phone not heard) or ``ins``
    (heard phone with no expected counterpart).  On ties, exact matches win,
    then insertions, so that surplus sounds at the end of a recording are
    reported as extras rather than paired with the last expected phone.
    """
    cost = _dp(expected, heard)
    eps = 1e-9
    ops: list[tuple[str, Optional[int], Optional[int]]] = []
    i, j = len(expected), len(heard)
    while i > 0 or j > 0:
        diag_ok = i > 0 and j > 0 and abs(cost[i][j] - (cost[i - 1][j - 1] + _sub_cost(expected[i - 1], heard[j - 1]))) < eps
        if diag_ok and same_sound(expected[i - 1], heard[j - 1]):
            ops.append(("ok", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif j > 0 and abs(cost[i][j] - (cost[i][j - 1] + 1.0)) < eps:
            ops.append(("ins", None, j - 1))
            j -= 1
        elif i > 0 and abs(cost[i][j] - (cost[i - 1][j] + 1.0)) < eps:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("sub", i - 1, j - 1))
            i, j = i - 1, j - 1
    ops.reverse()
    return ops


@dataclass
class PhoneResult:
    phone: str
    status: str  # ok | sub | del
    heard: Optional[str] = None
    prob: Optional[float] = None

    def to_dict(self) -> dict:
        d: dict = {"p": self.phone, "status": self.status}
        if self.heard is not None:
            d["heard"] = self.heard
        if self.prob is not None:
            d["prob"] = round(self.prob, 2)
        return d


Span = Optional[tuple[float, float]]

# Slot numbering over the phone sequence: gap 0, word 0, gap 1, word 1, …, word n-1, gap n.
# Word i is slot 2i+1; the gap before word i is slot 2i; the gap after the last word is 2n.


def _slots_by_time(heard: Sequence[Phone], spans: Sequence[Span], tolerance: float) -> Optional[list[int]]:
    valid = [(i, sp[0], sp[1]) for i, sp in enumerate(spans) if sp]
    if any(valid[k][1] > valid[k + 1][1] + 1e-6 for k in range(len(valid) - 1)):
        return None  # Whisper paired words out of order; the sequence method is safer
    slots: list[int] = []
    for ph in heard:
        t = (ph.start + ph.end) / 2
        best, best_d = None, None
        for i, s, e in valid:
            d = 0.0 if s <= t <= e else min(abs(t - s), abs(t - e))
            if d <= tolerance and (best_d is None or d < best_d):
                best, best_d = i, d
        if best is not None:
            slots.append(2 * best + 1)
        else:
            prev = [i for i, s, _ in valid if s <= t]
            slots.append(0 if not prev else 2 * prev[-1] + 2)
    if any(slots[k] > slots[k + 1] for k in range(len(slots) - 1)):
        return None
    return slots


def _slots_by_sequence(expected: Sequence[Sequence[str]], heard: Sequence[Phone]) -> list[int]:
    n = len(expected)
    flat: list[str] = []
    owner: list[int] = []
    for wi, phones in enumerate(expected):
        for p in phones:
            flat.append(p)
            owner.append(wi)
    ops = align(flat, [h.text for h in heard])
    positions = [k for k, (_, ei, _) in enumerate(ops) if ei is not None]
    first_pos = positions[0] if positions else len(ops)
    last_pos = positions[-1] if positions else -1
    slots = [0] * len(heard)
    current = 0
    for k, (kind, ei, hj) in enumerate(ops):
        if ei is not None:
            current = owner[ei]
        if hj is None:
            continue
        slots[hj] = 0 if k < first_pos else 2 * n if k > last_pos else 2 * current + 1
    return slots


def _cuts_from_slots(slots: Sequence[int], n_slots: int) -> list[int]:
    return [sum(1 for s in slots if s < k) for k in range(n_slots + 1)]


def _refine(cuts: list[int], expected: Sequence[Sequence[str]], heard: Sequence[Phone],
            locked: Sequence[bool], radius: int = 4, max_passes: int = 6) -> list[int]:
    """Move each boundary by a few phones when that lowers the total edit cost."""
    texts = [h.text for h in heard]

    def is_locked(slot: int) -> bool:
        return slot % 2 == 1 and locked[(slot - 1) // 2]

    def slot_cost(slot: int, lo: int, hi: int) -> float:
        if slot % 2 == 1:
            return edit_cost(expected[(slot - 1) // 2], texts[lo:hi])
        return GAP_COST * (hi - lo)

    n_words = len(expected)
    for _ in range(max_passes):
        changed = False
        # single boundaries
        for k in range(1, len(cuts) - 1):
            if is_locked(k - 1) or is_locked(k):
                continue
            lo, hi = cuts[k - 1], cuts[k + 1]
            best_pos = cuts[k]
            best_cost = slot_cost(k - 1, lo, cuts[k]) + slot_cost(k, cuts[k], hi)
            for pos in range(max(lo, cuts[k] - radius), min(hi, cuts[k] + radius) + 1):
                c = slot_cost(k - 1, lo, pos) + slot_cost(k, pos, hi)
                if c < best_cost - 1e-9:
                    best_pos, best_cost = pos, c
            if best_pos != cuts[k]:
                cuts[k] = best_pos
                changed = True
        # an empty gap between two words: move both of its cuts together so a phone
        # can pass directly from one word to the other
        for g in range(1, n_words):
            left, right = 2 * g - 1, 2 * g + 1  # word slots around gap slot 2g
            if is_locked(left) or is_locked(right) or cuts[2 * g] != cuts[2 * g + 1]:
                continue
            lo, hi = cuts[left], cuts[right + 1]
            cur = cuts[2 * g]
            best_pos = cur
            best_cost = slot_cost(left, lo, cur) + slot_cost(right, cur, hi)
            for pos in range(max(lo, cur - radius), min(hi, cur + radius) + 1):
                c = slot_cost(left, lo, pos) + slot_cost(right, pos, hi)
                if c < best_cost - 1e-9:
                    best_pos, best_cost = pos, c
            if best_pos != cur:
                cuts[2 * g] = cuts[2 * g + 1] = best_pos
                changed = True
        if not changed:
            break
    return cuts


@dataclass
class WordAlignment:
    """The heard phones ``heard[lo:hi]`` given to one word and how they align with it."""
    lo: int
    hi: int
    expected: list[str]  # the pronunciation the word was compared with
    ops: list[tuple[str, Optional[int], Optional[int]]]  # (kind, expected index, *global* heard index)
    form: Optional[int] = None  # index into the word's alternative forms when one of them fitted better

    def matched(self) -> dict[int, int]:
        """Expected phone index -> global heard index, for the phones that were heard (same or different)."""
        return {ei: hj for kind, ei, hj in self.ops if ei is not None and hj is not None}


@dataclass
class SentenceAlignment:
    heard: list[Phone]
    cuts: list[int]
    method: str
    words: list[WordAlignment] = field(default_factory=list)

    def gap(self, g: int) -> list[int]:
        """Global heard indices in gap ``g`` (0 = before the first word, n = after the last)."""
        return list(range(self.cuts[2 * g], self.cuts[2 * g + 1]))

    def word_time(self, wi: int) -> Optional[tuple[float, float]]:
        """Start and end of a word in the recording, from its phones (None when nothing was heard)."""
        w = self.words[wi]
        if w.hi <= w.lo:
            return None
        start = self.heard[w.lo].start
        last_end = self.heard[w.hi - 1].end
        # a word lasts until the next sound, unless that is far away (a pause)
        end = min(self.heard[w.hi].start, last_end + 0.25) if w.hi < len(self.heard) else last_end + 0.12
        return start, max(end, last_end)


def align_sentence(words: Sequence[str], expected: Sequence[Sequence[str]], heard: Sequence[Phone],
                   spans: Optional[Sequence[Span]] = None, tolerance: float = 0.1,
                   forms: Optional[Sequence[Sequence[Sequence[str]]]] = None) -> SentenceAlignment:
    """Cut the heard phones into word slots and align every word with its slot.

    ``spans`` optionally gives, per word, the ``(start, end)`` of the heard word
    (from the Whisper alignment) or ``None`` for words Whisper did not hear.
    ``forms`` optionally gives, per word, other accepted pronunciations; the one
    closest to what was heard is used.
    """
    n = len(words)
    expected = [list(e) for e in expected][:n] + [[] for _ in range(n - len(expected))]
    heard = sorted(heard, key=lambda h: h.start)
    spans = list(spans) if spans is not None else []
    slots: Optional[list[int]] = None
    method = "sequence"
    if n and len(spans) == n and sum(1 for s in spans if s) >= max(1, n // 2):
        slots = _slots_by_time(heard, spans, tolerance)
        if slots is not None:
            method = "time"
    if slots is None:
        slots = _slots_by_sequence(expected, heard)
    cuts = _cuts_from_slots(slots, 2 * n + 1)
    locked = [method == "time" and spans[i] is None for i in range(n)]
    if n:
        cuts = _refine(cuts, expected, heard, locked)

    texts = [h.text for h in heard]
    out = SentenceAlignment(heard=heard, cuts=cuts, method=method)
    for wi in range(n):
        lo, hi = cuts[2 * wi + 1], cuts[2 * wi + 2]
        best, best_form = expected[wi], None
        alternatives = list(forms[wi]) if forms is not None and wi < len(forms) else []
        if alternatives and hi > lo:
            best_cost = edit_cost(best, texts[lo:hi])
            for k, alt in enumerate(alternatives):
                c = edit_cost(alt, texts[lo:hi])
                if c < best_cost - 1e-9:
                    best, best_form, best_cost = list(alt), k, c
        ops = [(kind, ei, None if hj is None else lo + hj) for kind, ei, hj in align(best, texts[lo:hi])]
        out.words.append(WordAlignment(lo=lo, hi=hi, expected=best, ops=ops, form=best_form))
    return out


def _word_feedback(heard: Sequence[Phone], wa: WordAlignment) -> tuple[list[PhoneResult], list[str]]:
    """Per-phone results and the heard phones of one word, in order."""
    results: list[PhoneResult] = []
    heard_list: list[str] = []
    for kind, ei, hj in wa.ops:
        if kind == "ins":
            heard_list.append(heard[hj].text)
        elif kind == "del":
            results.append(PhoneResult(wa.expected[ei], "del"))
        else:
            results.append(PhoneResult(wa.expected[ei], kind, heard[hj].text, heard[hj].prob))
            heard_list.append(heard[hj].text)
    return results, heard_list


def score_phonemes(words: Sequence[str], expected: Sequence[Sequence[str]], heard: Sequence[Phone],
                   spans: Optional[Sequence[Span]] = None, tolerance: float = 0.1,
                   stress: Optional[Sequence[Sequence[int]]] = None,
                   audio=None, sample_rate: int = 16000,
                   forms: Optional[Sequence[Sequence[Sequence[str]]]] = None,
                   alignment: Optional[SentenceAlignment] = None, acoustics=None) -> dict:
    """Per-word sound feedback.

    ``spans`` optionally gives, per display word, the ``(start, end)`` of the
    heard word inside the recording (from the Whisper alignment) or ``None``
    for words Whisper did not hear.  With ``stress`` (per-phone stress levels,
    aligned with ``expected``) and the recording ``audio`` (or its precomputed
    ``acoustics``), every word of two or more syllables also gets a
    stressed-syllable verdict.
    """
    n = len(words)
    expected = [list(e) for e in expected][:n] + [[] for _ in range(n - len(expected))]
    al = alignment or align_sentence(words, expected, heard, spans, tolerance, forms)
    heard = al.heard
    stress_levels = None
    if stress is not None:
        stress_levels = [list(s) for s in stress][:n] + [[] for _ in range(n - len(stress))]
    if acoustics is None and audio is not None and len(audio):
        from . import acoustics as acoustics_mod

        acoustics = acoustics_mod.analyze(audio, sample_rate)
    pauses = acoustics.pauses() if acoustics is not None else []

    out_words = []
    total_ok = total = 0
    stress_checked = stress_correct = 0
    for wi, word in enumerate(words):
        wa = al.words[wi]
        results, heard_list = _word_feedback(heard, wa)
        ok = sum(1 for r in results if r.status == "ok")
        total_ok += ok
        total += len(wa.expected)
        entry = {
            "text": word,
            "expected": wa.expected,
            "phones": [r.to_dict() for r in results],
            "heard": heard_list,
            "accuracy": round(100 * ok / len(wa.expected)) if wa.expected else None,
        }
        if wa.form is not None:
            entry["form"] = " ".join(wa.expected)
        if stress_levels is not None:
            entry["stress_levels"] = stress_levels[wi] if wa.form is None else [0] * len(wa.expected)
            # with less than half of its sounds recognised, the syllables of a word cannot be located
            if acoustics is not None and wa.form is None and wa.expected and 2 * ok >= len(wa.expected):
                from .stress import analyze_word

                verdict = analyze_word(expected[wi], stress_levels[wi], wa.matched(), heard, acoustics,
                                       word_time=al.word_time(wi), pause_after=_pause_after(al, wi, pauses))
                if verdict is not None:
                    entry["stress"] = verdict
                    if verdict["match"] is not None:
                        stress_checked += 1
                        stress_correct += int(verdict["match"])
        out_words.append(entry)
    gaps = [[heard[k].text for k in al.gap(g)] for g in range(n + 1)]
    return {
        "accuracy": round(100 * total_ok / total) if total else None,
        "method": al.method,
        "stress_checked": stress_checked,
        "stress_correct": stress_correct,
        "words": out_words,
        "extra_before": gaps[0] if n else [h.text for h in heard],
        "extra_between": [p for g in gaps[1:n] for p in g],
        "extra_after": gaps[n] if n else [],
        "expected": " ".join(p for w in al.words for p in w.expected),
        "heard": " ".join(h.text for h in heard),
    }


def _pause_after(al: SentenceAlignment, wi: int, pauses: Sequence[tuple[float, float]]) -> bool:
    """Is the word followed by a pause (or the end of the recording)?"""
    if wi == len(al.words) - 1:
        return True
    span = al.word_time(wi)
    if span is None:
        return False
    nxt = next((al.word_time(k) for k in range(wi + 1, len(al.words)) if al.word_time(k)), None)
    limit = nxt[0] if nxt else span[1] + 1.0
    return any(span[1] - 0.15 <= s and e <= limit + 0.15 for s, e in pauses)
