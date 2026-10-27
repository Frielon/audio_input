"""Compare what the learner said with the target sentence.

The learner's recording is transcribed with Whisper; the recognised words are
aligned against the target words with a sequence matcher.  Each target word
gets a status (``ok`` / ``close`` / ``wrong`` / ``missing``), extra words the
learner added are listed separately, and an overall score is derived from the
word statuses.  Whisper's per-word probability is reported as a *clarity* hint,
and the time span of every heard word inside the recording is returned so the
frontend can play back individual words.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .whisper_engine import RecognizedWord

# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_to_words(n: int) -> str:
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + number_to_words(rest))
    for value, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= value:
            head, rest = divmod(n, value)
            return number_to_words(head) + " " + name + ("" if rest == 0 else " " + number_to_words(rest))
    return str(n)


_ALIASES = {
    "ok": "okay", "mr": "mister", "mrs": "missus", "dr": "doctor", "vs": "versus", "&": "and",
    "'cause": "because", "cuz": "because", "til": "until", "'til": "until", "em": "them", "'em": "them",
}

_HOMOPHONES = [
    {"their", "there", "they're"}, {"to", "too", "two"}, {"your", "you're"}, {"its", "it's"},
    {"for", "four"}, {"know", "no"}, {"right", "write"}, {"here", "hear"}, {"one", "won"},
    {"by", "buy", "bye"}, {"see", "sea"}, {"be", "bee"}, {"hour", "our"}, {"where", "wear"},
    {"weather", "whether"}, {"which", "witch"}, {"whole", "hole"}, {"would", "wood"},
    {"knew", "new"}, {"piece", "peace"}, {"week", "weak"}, {"meet", "meat"}, {"whose", "who's"},
    {"than", "then"}, {"affect", "effect"}, {"brake", "break"}, {"aloud", "allowed"},
    {"made", "maid"}, {"sun", "son"}, {"i", "eye"}, {"die", "dye"}, {"flour", "flower"},
    {"night", "knight"}, {"plane", "plain"}, {"sail", "sale"}, {"steal", "steel"},
    {"tail", "tale"}, {"wait", "weight"}, {"way", "weigh"}, {"waste", "waist"}, {"you", "u"},
    {"are", "r"}, {"through", "threw"}, {"cent", "sent", "scent"}, {"hi", "high"},
]
_HOMOPHONE_INDEX = {w: i for i, group in enumerate(_HOMOPHONES) for w in group}

_INNER_SPLIT_RE = re.compile(r"[/—–]+|(?<=\w)-(?=\w)")
_STRIP_RE = re.compile(r"^[^\w$%]+|[^\w%]+$")


def _normalize_token(tok: str) -> list[str]:
    """Normalised comparison tokens for one whitespace-delimited word (may be several)."""
    tok = tok.replace("’", "'").replace("‘", "'").lower()
    parts = [p for p in _INNER_SPLIT_RE.split(tok) if p]
    if len(parts) > 1:  # well-known, either/or, word—word
        out: list[str] = []
        for p in parts:
            out.extend(_normalize_token(p))
        return out
    tok = _STRIP_RE.sub("", tok)
    if not tok:
        return []
    if tok.endswith("%"):
        return _normalize_token(tok[:-1]) + ["percent"]
    if tok.startswith("$") and tok[1:].replace(",", "").isdigit():
        return _normalize_token(tok[1:]) + ["dollars"]
    if tok.replace(",", "").isdigit():
        num = int(tok.replace(",", ""))
        return number_to_words(num).split() if num < 10 ** 12 else [tok]
    m = re.match(r"^(\d+)(st|nd|rd|th|s)$", tok)
    if m:
        words = number_to_words(int(m.group(1))).split()
        return words[:-1] + [words[-1] + m.group(2)] if m.group(2) == "s" else words
    tok = _ALIASES.get(tok, tok)
    return [tok]


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in (text or "").split():
        tokens.extend(_normalize_token(raw))
    return tokens


def display_words(text: str) -> list[str]:
    """Whitespace-delimited words as shown to the learner (punctuation kept).

    The frontend splits the sentence on whitespace too, so the two sides agree on
    word boundaries; purely punctuation tokens such as a lone dash are skipped.
    """
    return [w for w in (text or "").split() if _normalize_token(w)]


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #

@dataclass
class WordResult:
    text: str
    status: str  # ok | close | wrong | missing
    heard: Optional[str] = None
    confidence: Optional[float] = None
    start: Optional[float] = None  # position of the heard word(s) inside the recording
    end: Optional[float] = None

    def to_dict(self) -> dict:
        d = {"text": self.text, "status": self.status}
        if self.heard is not None and self.heard != self.text:
            d["heard"] = self.heard
        if self.confidence is not None:
            d["confidence"] = round(self.confidence, 2)
        if self.start is not None and self.end is not None:
            d["start"] = round(self.start, 3)
            d["end"] = round(self.end, 3)
        return d


@dataclass
class ScoreResult:
    score: int
    words: list[WordResult]
    extra: list[str]
    heard_text: str
    clarity: Optional[int]
    speech_seconds: float
    target_seconds: Optional[float]
    counts: dict = field(default_factory=dict)
    penalty: int = 0
    alignment: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        pace = None
        if self.target_seconds and self.speech_seconds > 0:
            pace = round(self.target_seconds / self.speech_seconds, 2)
        return {
            "score": self.score,
            "penalty": self.penalty,
            "words": [w.to_dict() for w in self.words],
            "extra": self.extra,
            "alignment": self.alignment,
            "heard": self.heard_text,
            "clarity": self.clarity,
            "speech_seconds": round(self.speech_seconds, 2),
            "target_seconds": self.target_seconds,
            "pace": pace,
            "counts": self.counts,
        }


def _same_sound(a: str, b: str) -> bool:
    if a == b or a.replace("'", "") == b.replace("'", ""):
        return True
    ia, ib = _HOMOPHONE_INDEX.get(a), _HOMOPHONE_INDEX.get(b)
    return ia is not None and ia == ib


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _flatten(tokens_per_word: list[list[str]]) -> tuple[list[str], list[int]]:
    flat, owner = [], []
    for i, toks in enumerate(tokens_per_word):
        for t in toks:
            flat.append(t)
            owner.append(i)
    return flat, owner


_STATUS_ORDER = {"ok": 0, "close": 1, "wrong": 2, "missing": 3}


def align_words(target_text: str, heard: Sequence[RecognizedWord],
                close_threshold: float = 0.6) -> tuple[list[WordResult], list[str], list[dict]]:
    """Align recognised words to the target sentence.

    Returns ``(per_target_word, extra_words, alignment)``.  ``alignment`` is the
    ordered sequence of target/heard pairs, including the extra words the learner
    added, each with the time span of the heard word inside the recording.
    """
    target_display = display_words(target_text)
    tgt_flat, tgt_owner = _flatten([_normalize_token(w) for w in target_display])
    hrd_flat, hrd_owner = _flatten([_normalize_token(w.text) for w in heard])

    token_status = ["missing"] * len(tgt_flat)
    token_match: list[Optional[int]] = [None] * len(tgt_flat)  # index into ``heard``
    events: list[tuple[str, int]] = []  # ("t", target token) / ("x", unmatched heard token), in order

    sm = difflib.SequenceMatcher(None, tgt_flat, hrd_flat, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                token_status[i1 + k] = "ok"
                token_match[i1 + k] = hrd_owner[j1 + k]
                events.append(("t", i1 + k))
        elif tag == "delete":
            events.extend(("t", i) for i in range(i1, i2))
        elif tag == "insert":
            events.extend(("x", j) for j in range(j1, j2))
        else:  # replace block: match similar words first, then pair leftovers in order
            targets, heards = range(i1, i2), range(j1, j2)
            candidates = sorted(
                ((1.0 if _same_sound(tgt_flat[i], hrd_flat[j]) else _similarity(tgt_flat[i], hrd_flat[j]), i, j)
                 for i in targets for j in heards),
                key=lambda c: (-c[0], c[1], c[2]))
            matched_t: set[int] = set()
            matched_h: set[int] = set()
            for sim, i, j in candidates:
                if sim < close_threshold:
                    break
                if i in matched_t or j in matched_h:
                    continue
                matched_t.add(i)
                matched_h.add(j)
                token_status[i] = "ok" if sim >= 0.999 else "close"
                token_match[i] = hrd_owner[j]
            left_t = [i for i in targets if i not in matched_t]
            left_h = [j for j in heards if j not in matched_h]
            for i, j in zip(left_t, left_h):  # said *something* in that slot, but not the word
                matched_h.add(j)
                token_status[i] = "wrong"
                token_match[i] = hrd_owner[j]
            events.extend(("t", i) for i in targets)
            events.extend(("x", j) for j in heards if j not in matched_h)

    # one result per display word; a word with several tokens takes the worst status
    results: list[WordResult] = []
    for wi, word in enumerate(target_display):
        idxs = [k for k in range(len(tgt_flat)) if tgt_owner[k] == wi]
        worst = max((token_status[k] for k in idxs), key=_STATUS_ORDER.__getitem__) if idxs else "missing"
        matches: list[int] = []
        for k in idxs:
            m = token_match[k]
            if m is not None and m not in matches:
                matches.append(m)
        confs = [heard[m].probability for m in matches]
        results.append(WordResult(
            text=word, status=worst,
            heard=" ".join(heard[m].text for m in matches) if matches else None,
            confidence=sum(confs) / len(confs) if confs else None,
            start=min(heard[m].start for m in matches) if matches else None,
            end=max(heard[m].end for m in matches) if matches else None,
        ))

    used_heard = {m for m in token_match if m is not None}
    extra: list[str] = []
    alignment: list[dict] = []
    seen_t: set[int] = set()
    seen_x: set[int] = set()
    for kind, idx in events:
        if kind == "t":
            wi = tgt_owner[idx]
            if wi in seen_t:
                continue
            seen_t.add(wi)
            r = results[wi]
            alignment.append({
                "target": r.text, "heard": r.heard, "status": r.status,
                "start": None if r.start is None else round(r.start, 3),
                "end": None if r.end is None else round(r.end, 3),
                "confidence": None if r.confidence is None else round(r.confidence, 2),
            })
        else:
            hi = hrd_owner[idx]
            if hi in seen_x or hi in used_heard:
                continue
            seen_x.add(hi)
            w = heard[hi]
            extra.append(w.text)
            alignment.append({"target": None, "heard": w.text, "status": "extra",
                              "start": round(w.start, 3), "end": round(w.end, 3),
                              "confidence": round(w.probability, 2)})
    return results, extra, alignment


def score_attempt(target_text: str, heard: Sequence[RecognizedWord], heard_text: str = "",
                  target_seconds: Optional[float] = None) -> ScoreResult:
    words, extra, alignment = align_words(target_text, heard)
    counts = {"ok": 0, "close": 0, "wrong": 0, "missing": 0}
    for w in words:
        counts[w.status] += 1
    n = len(words)
    score = round(100 * (counts["ok"] + 0.5 * counts["close"]) / n) if n else 0
    # a flood of extra words (reading the wrong line, rambling) costs a little
    allowed_extra = max(2, n // 2)
    penalty = 0
    if n and len(extra) > allowed_extra:
        penalty = min(20, 5 * (len(extra) - allowed_extra), score)
        score -= penalty
    confs = [w.confidence for w in words if w.confidence is not None and w.status in ("ok", "close")]
    clarity = round(100 * sum(confs) / len(confs)) if confs else None
    speech = (heard[-1].end - heard[0].start) if heard else 0.0
    return ScoreResult(
        score=score, words=words, extra=extra,
        heard_text=heard_text or " ".join(w.text for w in heard),
        clarity=clarity, speech_seconds=max(0.0, speech), target_seconds=target_seconds,
        counts=counts, penalty=penalty, alignment=alignment,
    )
