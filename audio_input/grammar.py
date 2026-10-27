"""Grammar feedback on what you said.

A grammar-correction model rewrites every sentence of the transcript with its
grammar fixed: CoEdIT (``grammarly/coedit-large``, a Flan-T5 that Grammarly
fine-tuned for text editing) by default, about 0.25 s per sentence on the Apple
GPU with a beam search of 4; of its candidates the most minimal plausible
correction is kept (see :meth:`GrammarChecker.choose`).  ERRANT, the standard annotation toolkit of grammatical error correction,
then extracts the single edits between the two versions and names their type
(subject–verb agreement, article, preposition, verb tense, …).

Only grammar is reported.  Punctuation, capitalisation, spelling and
contractions in a transcript are Whisper's doing, not the speaker's, so those
edits are dropped, and so is the "missing subject" of phrases that are normal in
conversation ("Sounds good.", "Got it.").  A change of modal (can -> could) or a
rewording is shown as a suggestion rather than an error.  The corrected
sentence is the transcript with exactly the reported edits applied.
"""
from __future__ import annotations

import difflib
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

log = logging.getLogger(__name__)

PREFIX = "Fix grammatical errors in this sentence: "
BEAMS = 4

LABELS = {
    "VERB:SVA": "subject–verb agreement",
    "VERB:TENSE": "verb tense",
    "VERB:FORM": "verb form",
    "VERB:INFL": "verb form",
    "VERB": "verb",
    "NOUN:NUM": "singular / plural",
    "NOUN:INFL": "noun form",
    "NOUN:POSS": "possessive",
    "NOUN": "word choice",
    "DET": "article",
    "PREP": "preposition",
    "PRON": "pronoun",
    "ADJ": "adjective",
    "ADJ:FORM": "comparative / superlative",
    "ADV": "adverb",
    "CONJ": "conjunction",
    "PART": "particle",
    "MORPH": "word form",
    "WO": "word order",
    "OTHER": "wording",
}
IGNORED = {"PUNCT", "ORTH", "SPELL", "CONTR", "UNK"}
ARTICLES = {"a", "an", "the"}
MODALS = {"can", "could", "will", "would", "may", "might", "shall", "should", "must"}
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "ahh", "eh", "hmm", "hm", "mm", "mmm"}
# Subjectless openings that are normal in speech; a model trained on written text "fixes" them.
_CASUAL = re.compile(r"^\s*(sounds?|looks?|seems?|makes? sense|got it|see you|thanks|thank you|hope|"
                     r"no problem|never mind|will do|good|great|nice|sure|just)\b", re.I)
_MIN_SIMILARITY = 0.5  # a "correction" this different from the sentence is a model failure, not an edit


@dataclass
class Edit:
    start: int          # character span in the transcript; start == end: something is missing here
    end: int
    original: str       # the transcript's text in the span ("" when something is missing)
    correction: str     # what should be there instead ("" when the original should go)
    kind: str           # ERRANT type, e.g. "R:VERB:SVA" (M: missing, U: unnecessary, R: replace)
    label: str          # "subject–verb agreement"
    message: str        # "“go” → “goes”"
    severity: str       # "error" | "suggestion"

    @property
    def op(self) -> str:
        if self.start == self.end:
            return "missing"
        return "unnecessary" if not self.correction else "replace"

    def to_dict(self) -> dict:
        return {**asdict(self), "op": self.op}


@dataclass
class GrammarResult:
    text: str
    corrected: str
    edits: list[Edit] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def errors(self) -> list[Edit]:
        return [e for e in self.edits if e.severity == "error"]

    def to_dict(self) -> dict:
        return {"text": self.text, "corrected": self.corrected, "edits": [e.to_dict() for e in self.edits],
                "elapsed": round(self.elapsed, 3)}


# --------------------------------------------------------------------------- #
# Edits
# --------------------------------------------------------------------------- #

def describe(kind: str, original: str, correction: str) -> tuple[str, str, str]:
    """``(label, message, severity)`` of one ERRANT edit."""
    op, _, base = kind.partition(":")
    label = LABELS.get(base, "grammar")
    words = {original.lower(), correction.lower()} - {""}
    if base == "DET" and not words <= ARTICLES:
        label = "determiner"
    severity = "error"
    if base == "OTHER" or (base == "VERB:TENSE" and words & MODALS):
        severity = "suggestion"
    if op == "U" and original.lower() in FILLERS:
        label, severity = "filler word", "suggestion"
    if original and correction:  # also "Is" -> "It is", which ERRANT types as a missing pronoun
        message = f"“{original}” → “{correction}”"
    elif correction:
        message = f"add “{correction}”"
    else:
        message = f"drop “{original}”"
    return label, message, severity


def keep(kind: str, text: str, start: int) -> bool:
    """Is this edit about the speaker's grammar (not Whisper's punctuation, casing or spelling)?"""
    _, _, base = kind.partition(":")
    if base in IGNORED or not base:
        return False
    if kind == "M:PRON":
        clause = re.split(r"[.!?,;:]\s*", text[:start])[-1]
        if not clause.strip() and _CASUAL.match(text[start:]):
            return False  # "Sounds good." needs no "it"
    return True


def apply_edits(text: str, edits: Sequence[Edit]) -> str:
    """The text with the edits applied (edits must not overlap)."""
    out = text
    for e in sorted(edits, key=lambda e: (e.start, e.end), reverse=True):
        left, right = out[:e.start], out[e.end:]
        if e.start == e.end:  # insertion before the word at ``start`` (or before punctuation)
            ins = e.correction
            if right and not right[0].isspace() and right[0] not in ".,!?;:)":
                ins += " "
            if left and not left[-1].isspace():
                ins = " " + ins
            out = left + ins + right
        elif not e.correction:  # deletion: also drop one of the spaces around it
            if left.endswith(" ") and (not right or right[0] == " " or right[0] in ".,!?;:"):
                left = left[:-1]
            elif not left and right.startswith(" "):
                right = right[1:]
            out = left + right
            if not left.strip() and out[:1].islower():
                out = out[:1].upper() + out[1:]
        else:
            out = left + e.correction + right
    return out


def _span(doc, start: int, end: int, length: int) -> tuple[int, int]:
    if start < end:
        return doc[start].idx, doc[end - 1].idx + len(doc[end - 1].text)
    pos = doc[start].idx if start < len(doc) else length
    return pos, pos


def extract_edits(annotator, sentence: str, corrected: str, offset: int = 0) -> list[Edit]:
    """ERRANT edits between a sentence and its correction, as character spans of the sentence (+ ``offset``)."""
    if corrected.strip() == sentence.strip():
        return []
    orig = annotator.parse(sentence, tokenise=True)
    cor = annotator.parse(corrected, tokenise=True)
    out: list[Edit] = []
    for e in annotator.annotate(orig, cor):
        start, end = _span(orig, e.o_start, e.o_end, len(sentence))
        if not keep(e.type, sentence, start):
            continue
        original = sentence[start:end]
        correction = "".join(t.text_with_ws for t in cor[e.c_start:e.c_end]).strip()
        if not correction and re.sub(r"[^\w']", "", original).lower() in FILLERS:
            continue  # "um", "uh": listed as filler words by the feedback, not as grammar
        label, message, severity = describe(e.type, original, correction)
        out.append(Edit(start + offset, end + offset, original, correction, e.type, label, message, severity))
    return out


# --------------------------------------------------------------------------- #
# Checker
# --------------------------------------------------------------------------- #

class GrammarChecker:
    """Lazily loaded correction model + ERRANT; thread-safe."""

    def __init__(self, model_name: str = "grammarly/coedit-large", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.loaded = False
        self._lock = threading.Lock()
        self._tok = self._model = self._annotator = None

    @property
    def annotator(self):
        """ERRANT (spaCy's small English pipeline); loaded on first use, independently of the model."""
        with self._lock:
            if self._annotator is None:
                import errant

                self._annotator = errant.load("en")
            return self._annotator

    def load(self) -> None:
        self.annotator  # noqa: B018  (loads it)
        with self._lock:
            if self.loaded:
                return
            t0 = time.time()
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = self.device
            if device == "auto":
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            # bfloat16 keeps T5's range (float16 can overflow in its feed-forward layers)
            dtype = torch.bfloat16 if device != "cpu" else torch.float32
            try:  # no network round trip when the model is cached
                self._tok = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, dtype=dtype, local_files_only=True)
            except OSError:
                log.info("Downloading %s (about 3 GB, once)…", self.model_name)
                self._tok = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, dtype=dtype)
            self._model = model.to(device=device, dtype=dtype).eval()
            self.device = device
            self._generate(["He go to school."])
            self.loaded = True
            log.info("Grammar model ready (%s on %s) in %.1fs", self.model_name, device, time.time() - t0)

    def _generate(self, sentences: list[str], n: int = BEAMS) -> list[list[str]]:
        """The model's ``n`` best corrections of every sentence, best first."""
        import torch

        enc = self._tok([PREFIX + s for s in sentences], return_tensors="pt", padding=True).to(self.device)
        limit = int(enc.input_ids.shape[1] * 1.5) + 16
        with torch.inference_mode():
            out = self._model.generate(**enc, max_new_tokens=limit, num_beams=n, num_return_sequences=n,
                                       do_sample=False)
        texts = [t.strip() for t in self._tok.batch_decode(out, skip_special_tokens=True)]
        return [texts[k * n:(k + 1) * n] for k in range(len(sentences))]

    def correct(self, sentences: list[str]) -> list[list[str]]:
        """Candidate corrections of every sentence, best first (an unchanged sentence means: no error)."""
        if not sentences:
            return []
        self.load()
        with self._lock:
            return self._generate(sentences)

    def choose(self, sentence: str, candidates: list[str], offset: int = 0) -> list[Edit]:
        """The edits of the most plausible minimal correction.

        The model's best guess wins when it leaves the sentence alone.  Otherwise
        the candidate with the most clear-cut corrections and the fewest
        rewordings (ERRANT type OTHER) is taken: "the case need a calculator"
        becomes "needs", not "in that case, a calculator is needed".
        """
        best, best_score = [], None
        for rank, cand in enumerate(candidates):
            if not cand or difflib.SequenceMatcher(None, sentence, cand).ratio() < _MIN_SIMILARITY:
                continue
            if cand.strip() == sentence.strip():
                if rank == 0:
                    return []
                continue
            annotator = self.annotator
            with self._lock:
                edits = extract_edits(annotator, sentence, cand, offset)
            if not edits:  # only punctuation or casing changed
                if rank == 0:
                    return []
                continue
            other = sum(1 for e in edits if e.kind.endswith(":OTHER"))
            clear = sum(1.0 if e.severity == "error" else 0.5 for e in edits if not e.kind.endswith(":OTHER"))
            score = clear - 3 * other - 0.15 * rank  # a rewording drags several "corrections" along with it
            if best_score is None or score > best_score:
                best, best_score = edits, score
        return best

    def sentences(self, text: str) -> list[tuple[int, str]]:
        """``(offset, sentence)`` pairs of the text."""
        annotator = self.annotator
        with self._lock:
            doc = annotator.nlp(text)
        return [(s.start_char, s.text) for s in doc.sents if s.text.strip()]

    def check(self, text: str) -> GrammarResult:
        t0 = time.time()
        text = text.strip()
        if not re.search(r"[A-Za-z]", text):
            return GrammarResult(text, text)
        parts = [(off, s) for off, s in self.sentences(text) if len(re.findall(r"[A-Za-z']+", s)) >= 2]
        candidates = self.correct([s for _, s in parts])
        edits: list[Edit] = []
        for (off, sentence), cands in zip(parts, candidates):
            edits += self.choose(sentence, cands, off)
        return GrammarResult(text, apply_edits(text, edits), edits, time.time() - t0)
