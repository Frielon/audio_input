"""One utterance -> what the overlay shows.

Words are the transcript's whitespace-separated words without pure punctuation,
the same "display words" parrot_shadowing analyses, so sound feedback and grammar
marks refer to the same indices.  For every word: the expected IPA with each
sound's verdict (ok / sub / del, and what was heard instead), its syllables and
stress (which syllable you made most prominent, and the model voice's), where
it is in the recording (to play it back), and the grammar edits that touch it.
Words that are missing are attached to the gap before the next word.

The coach's checks are grouped by focus area as in Parrot Shadowing's coach
panel (``coach_rows``): your focus areas first (by default final consonants,
TH, /r/, /s/–/z/ and word stress), each with the occurrences that went wrong,
the ones that were fine, and the teacher's tip.  ``issues`` is the flat list
of problems (grammar errors, sound problems, suggestions) for the history.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Optional, Sequence

from .grammar import FILLERS, GrammarResult

# The teacher's checklist of Parrot Shadowing (labels and tips as in its coach panel).
CATEGORIES = {
    "final": ("Final consonants",
              "Finish every word: tongue tip up for a final /n/ (line, one), a clear /t/ at the end of a phrase "
              "(root), both sounds of /sk/ (ask)."),
    "th": ("TH sounds",
           "Tongue tip between the teeth and blow. /θ/ (think, method) is voiceless, /ð/ (that, then, the) is "
           "voiced – never /s/, /z/, /t/ or /d/."),
    "r": ("/r/",
          "Pull the tongue back and curl it a little without touching the roof of the mouth; keep the r colour "
          "at the end (error = /ˈɛrər/)."),
    "sz": ("/s/ vs /z/",
           "/s/ is a hiss: closer = /ˈkloʊsər/. /z/ is a buzz you can feel in your throat: is = /ɪz/."),
    "stress": ("Word stress",
               "Make the stressed syllable longer, louder and higher, and squeeze the others: CAL·cu·la·tor, "
               "a·RITH·me·tic, ap·PROX·i·mate."),
    "uh": ("No extra “uh”",
           "Go straight from a final /t/ or /d/ into the next word – no vowel after it. Next‿point, not "
           "next-uh point."),
    "rhythm": ("Reduce small words",
               "Content words carry the beat. Say the small words quickly and softly: to /tə/, the /ðə/, a /ə/, "
               "and /ən/, of /əv/."),
    "pause": ("Pauses", "Pause briefly between thought groups – never inside a phrase like “the ‖ root”."),
    "focus": ("Focus & intonation",
              "Let your voice move: make the focus word of each phrase (usually its last important word) higher, "
              "longer and louder."),
}
COACH_LABELS = {k: v[0] for k, v in CATEGORIES.items()}
DEFAULT_FOCUS = ("final", "th", "r", "sz", "stress")
_TOKEN = re.compile(r"\S+")
_WORDLIKE = re.compile(r"[A-Za-z0-9$%]")


def _is_filler(word: str) -> bool:
    return re.sub(r"[^\w']", "", word).lower() in FILLERS


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))


def word_spans(text: str, words: Optional[Sequence[str]] = None) -> list[tuple[int, int]]:
    """Character spans of the display words (``words`` from parrot when available)."""
    if words is None:
        return [m.span() for m in _TOKEN.finditer(text) if _WORDLIKE.search(m.group())]
    spans, pos = [], 0
    for w in words:
        i = text.find(w, pos)
        if i < 0:  # should not happen: display words are tokens of the text
            i = pos
        spans.append((i, i + len(w)))
        pos = i + len(w)
    return spans


def _ipa(word: dict) -> list[dict]:
    levels = word.get("stress_levels") or []
    out = []
    for i, p in enumerate(word.get("phones", [])):
        item = {"p": p["p"], "s": p["status"]}
        if p.get("heard") and p["status"] == "sub":
            item["h"] = p["heard"]
        if i < len(levels) and levels[i]:
            item["st"] = levels[i]
        out.append(item)
    return out


def _stress(word: dict) -> Optional[dict]:
    """Which syllable you stressed (``ok`` None: could not tell), with the prominence of every syllable."""
    s = word.get("stress")
    if not s or (s.get("match") is None and not s.get("prominence")):
        return None
    return {"ok": s.get("match"), "expected": s["expected"], "said": s.get("detected"), "n": s["syllables"],
            "prom": s.get("prominence")}


def _sound_level(word: dict, flagged: bool) -> Optional[str]:
    acc = word.get("accuracy")
    if acc is None:
        return None
    if acc < 50:
        return "bad"
    if acc < 100 or flagged:
        return "warn"
    return "ok"


def coach_rows(checks: Sequence[dict], focus: Sequence[str] = DEFAULT_FOCUS,
               skip_words: Sequence[int] = ()) -> list[dict]:
    """The coach's checks by category: focus areas first, then the others (only those that were checked)."""
    order = [c for c in focus if c in CATEGORIES] + [c for c in CATEGORIES if c not in focus]
    rows = []
    for cat in order:
        items = [c for c in checks if c.get("cat") == cat and c.get("word") not in skip_words]
        if not items:
            continue
        bad = [c for c in items if c.get("ok") is False]
        unsure = [c for c in items if c.get("ok") is None and cat == "stress" and c.get("prominence")]
        good = [c for c in items if c.get("ok") is True]
        label, tip = CATEGORIES[cat]
        rows.append({
            "cat": cat, "label": label, "focus": cat in focus, "ok": len(good), "bad": len(bad),
            "tip": tip if bad else None,
            "items": [{"label": c.get("label"), "detail": c.get("detail", ""), "ok": c.get("ok"),
                       "t": c.get("t"), "word": c.get("word"),
                       **({"prom": c.get("prominence"), "exp": c.get("expected_syllable")}
                          if cat == "stress" and c.get("prominence") else {})}
                      for c in bad + unsure],
            "passed": [c.get("label") for c in good],
        })
    return rows


def build(text: str, pron: Optional[dict] = None, grammar: Optional[GrammarResult] = None,
          focus: Sequence[str] = DEFAULT_FOCUS) -> dict:
    """The overlay's data for one utterance (pron: parrot's analysis, or None / {"error": ...})."""
    ph_words = (pron or {}).get("phonemes", {}) or {}
    ph_words = ph_words.get("words") if isinstance(ph_words, dict) else None
    display = [w["text"] for w in ph_words] if ph_words else None
    spans = word_spans(text, display)
    n = len(spans)
    coach = (pron or {}).get("coach") or {}
    prosody = (pron or {}).get("prosody") or {}
    syllables = (pron or {}).get("syllables") or []
    fillers_at = [i for i, (a, b) in enumerate(spans) if _is_filler(text[a:b])]
    checks = [c for c in coach.get("checks", []) if c.get("word") not in fillers_at or c.get("cat") in
              ("rhythm", "focus", "pause")]
    failed = [c for c in checks if c.get("ok") is False]
    flagged_words = {c["word"] for c in failed if c.get("cat") not in ("rhythm", "focus", "pause")}

    words = []
    for i, (a, b) in enumerate(spans):
        item: dict = {"w": text[a:b]}
        if ph_words and i < len(ph_words):
            pw = ph_words[i]
            item["ipa"] = _ipa(pw)
            item["acc"] = pw.get("accuracy")
            heard = "".join(pw.get("heard") or [])
            if pw.get("form"):
                item["form"] = pw["form"].replace(" ", "")
            elif any(p["status"] != "ok" for p in pw.get("phones", [])):
                item["heard"] = heard or "–"  # only when a sound was wrong or missing (not for free variants)
            if i < len(syllables) and len(syllables[i]) > 1:
                item["syl"] = [s for s, _ in syllables[i]]  # where each syllable starts (phone index)
            stress = _stress(pw)
            if stress:
                item["stress"] = stress
            item["sound"] = _sound_level(pw, i in flagged_words)
        pr = prosody.get("words") or []
        if i < len(pr):
            item["fn"] = not pr[i].get("content", True)
            if pr[i].get("start") is not None:
                item["t"] = [pr[i]["start"], pr[i]["end"]]
        if i in fillers_at:
            item["filler"] = True  # how an "um" sounds is no pronunciation error
            for key in ("sound", "heard", "stress"):
                item.pop(key, None)
        words.append(item)

    inserts: list[dict] = []
    edits = grammar.edits if grammar else []
    for k, e in enumerate(edits):
        covered = [i for i, (a, b) in enumerate(spans) if a < e.end and e.start < b]
        if e.start == e.end or not covered:
            before = next((i for i, (a, _) in enumerate(spans) if a >= e.start), n)
            inserts.append({"before": before, "text": e.correction, "g": k, "severity": e.severity})
            continue
        for j, i in enumerate(covered):
            g = words[i].setdefault("g", [])
            g.append({"id": k, "op": e.op, "severity": e.severity,
                      "fix": e.correction if j == 0 else ""})  # the correction is written over the first word

    issues: list[dict] = []
    for k, e in enumerate(edits):
        if e.severity == "error":
            issues.append({"kind": "grammar", "label": e.label, "text": e.message, "g": k})
    for c in failed:
        issues.append({"kind": "sound", "cat": c["cat"], "label": COACH_LABELS.get(c["cat"], c["cat"]),
                       "word": c.get("label"), "text": c.get("detail", "")})
    for k, e in enumerate(edits):
        if e.severity != "error":
            issues.append({"kind": "suggestion", "label": e.label, "text": e.message, "g": k})
    fillers = Counter(re.sub(r"[^\w']", "", w["w"]).lower() for w in words if w.get("filler"))
    if fillers:
        issues.append({"kind": "suggestion", "label": "filler words",
                       "text": ", ".join(f"“{f}”" + (f" ×{n}" if n > 1 else "") for f, n in fillers.most_common())})

    ph = (pron or {}).get("phonemes") or {}
    summary = {
        "sounds": ph.get("accuracy"),
        "stress": [ph["stress_correct"], ph["stress_checked"]] if ph.get("stress_checked") else None,
        "grammar": sum(1 for e in edits if e.severity == "error"),
        "suggestions": sum(1 for e in edits if e.severity != "error"),
        "fillers": sum(fillers.values()),
        "checks": coach.get("summary") or {},
    }
    corrected = grammar.corrected if grammar and grammar.edits else None
    return {
        "text": text,
        "corrected": corrected,
        "corrected_marks": diff_marks(text, corrected) if corrected else [],
        "words": words,
        "inserts": inserts,
        "issues": issues,
        "coach": coach_rows(checks, focus) if coach else [],
        "summary": summary,
        "grammar_edits": [e.to_dict() for e in edits],
        "grammar_checked": grammar is not None,
        "analyzed": bool(ph_words) or grammar is not None,
        "sound_error": (pron or {}).get("error"),
    }


def diff_marks(original: str, corrected: str) -> list[dict]:
    """The corrected sentence word by word, ``c`` true for the words that changed."""
    a, b = original.split(), corrected.split()
    out: list[dict] = []
    sm = difflib.SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b], autojunk=False)
    for tag, _, _, j1, j2 in sm.get_opcodes():
        out += [{"t": w, "c": tag != "equal"} for w in b[j1:j2]]
    return out


def map_words(ours: Sequence[str], theirs: Sequence[str]) -> dict[int, int]:
    """Index pairs of the words two versions of a sentence have in common (``ours`` -> ``theirs``)."""
    a, b = [norm(w) for w in ours], [norm(w) for w in theirs]
    out: dict[int, int] = {}
    for tag, i1, i2, j1, _ in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.update({i1 + k: j1 + k for k in range(i2 - i1)})
    return out


def attach_model(payload: dict, model_words: Sequence[str], reference: dict) -> dict:
    """Add the model voice's word times (``mt``) and syllable prominence (``stress.model``) to the payload.

    ``reference`` is parrot's ``analyze_reference`` of the model reading, whose
    display words are ``model_words``; words that differ (a grammar correction)
    get nothing.
    """
    mapping = map_words([w["w"] for w in payload["words"]], model_words)
    ref_words = reference.get("words") or []
    ref_stress = reference.get("stress") or {}
    for i, j in mapping.items():
        w = payload["words"][i]
        if j < len(ref_words) and ref_words[j].get("start") is not None:
            w["mt"] = [ref_words[j]["start"], ref_words[j]["end"]]
        prom = ref_stress.get(str(j))
        if prom and w.get("stress") and len(prom) == len(w["stress"].get("prom") or prom):
            w["stress"]["model"] = prom
    for row in payload.get("coach", []):
        for item in row["items"]:
            if row["cat"] == "stress" and item.get("word") in mapping:
                prom = ref_stress.get(str(mapping[item["word"]]))
                if prom:
                    item["model"] = prom
    payload["model_ready"] = True
    return payload
