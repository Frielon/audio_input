"""Targeted feedback on a learner's known weak points.

The checks follow the kind of notes a pronunciation teacher gives: TH sounds,
final consonants, /s/ vs /z/, /r/, no extra "uh" after a final consonant, word
stress, rhythm (reduced function words), pauses between thought groups and an
expressive (not flat) pitch.  Every check is one occurrence in the sentence
with a verdict (``ok`` true / false, or None when the recording does not tell)
and the time span to play back.

Rules come from what the recognizer hears in *native* speech, so that a fluent
speaker is rarely flagged: for example a final /t/ or /d/ before a consonant is
often not released by native speakers either (``nex(t) point``), so it is not
checked; a final /n/ or /k/ almost always is, so a missing one is reported.
On 120 sentences of a native speaker the share of checks flagged as wrong is
1-4% for sounds, the extra "uh" and word stress, and about 5% for flat pitch.

The recognizer was trained on dictionary pronunciations, so it tends to "hear"
the dictionary form of a function word whatever was said.  Reduction is
therefore checked only for words whose dictionary form in context is already
the weak one (``to`` tə, ``the`` ðə, ``a`` ə): hearing the strong form there
(tuː, ðiː, eɪ) means it really was said.
"""
from __future__ import annotations

from typing import Optional, Sequence

from . import english
from .phoneme_engine import is_vowel
from .phoneme_scoring import SentenceAlignment

CATEGORIES = ("th", "final", "sz", "r", "uh", "stress", "rhythm", "pause", "focus")

_SIBILANTS = {"s": "s", "z": "z", "ʃ": "sh", "ʒ": "zh", "ts": "ts"}
_STOPS = {"t": "t", "d": "d", "ɾ": "d"}
_LABIODENTALS = {"f": "f", "v": "v"}
_VOICELESS = {"p", "t", "k", "f", "θ", "s", "ʃ", "tʃ", "h"}
_R_VOWELS = {"ɚ", "ɜː", "ɪɹ", "ɛɹ", "ʊɹ", "ɑːɹ", "ɔːɹ", "oːɹ", "aɪɚ"}
_R_UNRELIABLE = {"ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ"}  # native speakers are often heard without r colour here
_OBSTRUENTS = {"p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "tʃ", "dʒ"}
MIN_PITCH_RANGE = 3.5     # semitones: the 5th percentile of native sentences (median 5.7)
RATIO_OK = 0.8            # function-word time ratio: native median 0.63, 90th percentile 0.89
RATIO_BAD = 1.0           # word-by-word reading: median 1.1


def _is_rhotic(phone: Optional[str]) -> bool:
    return bool(phone) and ("ɹ" in phone or "ɚ" in phone or phone == "ɜː")


def _respell(word: str, letters: str) -> str:
    """``think`` heard with /s/ -> ``sink``: how the mistake sounds, in letters."""
    w = english.norm(word)
    i = w.find("th")
    return w[:i] + letters + w[i + 2:] if i >= 0 else w


def _ordinal(k: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")


class _Coach:
    def __init__(self, words: Sequence[str], al: SentenceAlignment, phonemes: dict, prosody: dict,
                 reference: Optional[dict], dictionary: Optional[Sequence[Sequence[str]]]):
        self.words = list(words)
        self.n = len(words)
        self.al = al
        self.ph = phonemes
        self.pr = prosody
        self.ref = reference or {}
        self.dictionary = [list(d) for d in dictionary] if dictionary is not None else [w.expected for w in al.words]
        self.checks: list[dict] = []
        self.content = [w["content"] for w in prosody["words"]]

    # ------------------------------------------------------------ helpers
    def add(self, cat: str, wi: int, ok: Optional[bool], detail: str, label: Optional[str] = None,
            span: Optional[tuple[int, int]] = None, **extra) -> None:
        first, last = span or (wi, wi)
        if span:  # play from the first to the last word that was actually heard
            timed = [k for k in range(first, last + 1) if self.al.word_time(k)]
            first, last = (timed[0], timed[-1]) if timed else (first, last)
        t0 = self.al.word_time(first)
        t1 = self.al.word_time(last)
        check = {"cat": cat, "word": wi, "ok": ok, "label": label or english.norm(self.words[wi]) or self.words[wi],
                 "detail": detail}
        if span:
            check["span"] = list(span)
        if t0 and t1:
            check["t"] = [round(max(0.0, t0[0] - 0.05), 3), round(t1[1] + 0.08, 3)]
        check.update(extra)
        self.checks.append(check)

    def heard_for(self, wi: int) -> dict[int, tuple[str, Optional[str]]]:
        """Expected phone index -> (kind, heard phone or None)."""
        out = {}
        for kind, ei, hj in self.al.words[wi].ops:
            if ei is not None:
                out[ei] = (kind, None if hj is None else self.al.heard[hj].text)
        return out

    def reliable(self, wi: int) -> bool:
        """Enough of the word was recognised for its phones to be judged one by one.

        Half of its sounds right, or a third when the number of sounds heard
        matches (``that`` heard as ``s ɑː t``: two sounds wrong, but clearly
        that word).
        """
        acc = self.ph["words"][wi].get("accuracy")
        if acc is None:
            return False
        wa = self.al.words[wi]
        return acc >= 50 or (acc >= 30 and abs((wa.hi - wa.lo) - len(wa.expected)) <= 1)

    def next_first_phone(self, wi: int) -> Optional[str]:
        if wi + 1 >= self.n:
            return None
        e = self.al.words[wi + 1].expected
        return e[0] if e else None

    def before_pause(self, wi: int) -> bool:
        return wi == self.n - 1 or english.breaks_after(self.words[wi]) or \
            bool(self.pr["words"][wi].get("pause_after"))

    # ------------------------------------------------------------- checks
    def th(self, wi: int) -> None:
        for ei, p in enumerate(self.al.words[wi].expected):
            if p not in ("θ", "ð"):
                continue
            kind, h = self.heard_for(wi).get(ei, ("del", None))
            if kind == "ok" or h in ("θ", "ð"):
                self.add("th", wi, True, f"/{p}/")
            elif h in _SIBILANTS:
                self.add("th", wi, False, f"sounded like /{h}/ – “{_respell(self.words[wi], _SIBILANTS[h])}”",
                         heard=h, expected=p)
            elif h in _STOPS:
                self.add("th", wi, False, f"sounded like /{h}/ – “{_respell(self.words[wi], _STOPS[h])}”",
                         heard=h, expected=p)
            elif h in _LABIODENTALS:
                self.add("th", wi, False, f"sounded like /{h}/ – “{_respell(self.words[wi], _LABIODENTALS[h])}”",
                         heard=h, expected=p)
            elif kind == "del" and self.content[wi]:
                self.add("th", wi, None, f"/{p}/ not clearly heard", expected=p)

    def final(self, wi: int) -> None:
        e = self.al.words[wi].expected
        if not e or is_vowel(e[-1]) or not self.content[wi]:
            return
        nucleus = max((k for k, p in enumerate(e) if is_vowel(p)), default=-1)
        if nucleus < 0:
            return
        nxt = self.next_first_phone(wi)
        consonant_next = nxt is not None and not is_vowel(nxt) and not self.before_pause(wi)
        if e[-1] in ("t", "d") and consonant_next:
            return  # unreleased or dropped by native speakers too: "nex(t) point"
        heard = self.heard_for(wi)
        problems, coda = [], e[nucleus + 1:]
        for ei in range(nucleus + 1, len(e)):
            p = e[ei]
            kind, h = heard.get(ei, ("del", None))
            if kind == "ok":
                continue
            if p in ("t", "d") and ei < len(e) - 1:
                continue  # inside a cluster ("lifts", "hands")
            if (p == "ŋ" and h == "n") or (p == "l" and consonant_next) or (p in ("s", "z") and h in ("s", "z")):
                continue  # "-in'" for "-ing", dark l, voicing is checked on its own
            if p == "n" and h == "m" and nxt in ("p", "b", "m"):
                continue  # "i(m) place"
            missing = kind == "del" or is_vowel(h)  # a vowel in its place is not the consonant at all
            problems.append(f"final /{p}/ missing" if missing else f"final /{p}/ sounded like /{h}/")
        label_sound = "/" + "".join(coda) + "/"
        if problems:
            self.add("final", wi, False, "; ".join(problems))
        else:
            self.add("final", wi, True, f"final {label_sound}")

    def sz(self, wi: int) -> None:
        e = self.al.words[wi].expected
        heard = self.heard_for(wi)
        for ei, p in enumerate(e):
            if p not in ("s", "z"):
                continue
            kind, h = heard.get(ei, ("del", None))
            final = ei == len(e) - 1
            if p == "z" and h == "s":
                if final and self.next_first_phone(wi) in _VOICELESS:
                    continue  # devoiced before a voiceless sound by native speakers too
                self.add("sz", wi, False, "/z/ sounded like /s/ – let it buzz", heard=h, expected=p)
            elif p == "s" and h == "z":
                self.add("sz", wi, False, "/s/ sounded like /z/ – no buzz", heard=h, expected=p)
            elif kind == "ok":
                between_vowels = 0 < ei < len(e) - 1 and is_vowel(e[ei - 1]) and is_vowel(e[ei + 1])
                if (p == "z" and final) or (p == "s" and between_vowels):
                    self.add("sz", wi, True, f"/{p}/")

    def r(self, wi: int) -> None:
        heard = self.heard_for(wi)
        for ei, p in enumerate(self.al.words[wi].expected):
            if p != "ɹ" and p not in _R_VOWELS:
                continue
            kind, h = heard.get(ei, ("del", None))
            if kind == "ok" or (p in _R_VOWELS and _is_rhotic(h)):
                if self.content[wi]:
                    self.add("r", wi, True, f"/{p}/")
            elif p == "ɹ":
                if kind == "del":
                    self.add("r", wi, False, "/r/ missing", expected=p)
                elif h in ("l", "w"):
                    self.add("r", wi, False, f"/r/ sounded like /{h}/", heard=h, expected=p)
            elif kind == "sub" and p not in _R_UNRELIABLE:
                self.add("r", wi, False, f"no r sound: /{p}/ sounded like /{h}/", heard=h, expected=p)

    def inserted_vowels(self, wi: int) -> list[int]:
        """Heard vowels between the final consonant of word ``wi`` and the next word ("next-uh point")."""
        wa = self.al.words[wi]
        e = wa.expected
        if not e or is_vowel(e[-1]):
            return []
        heard = self.al.heard
        nucleus = max((k for k, p in enumerate(e) if is_vowel(p)), default=-1)
        cand: list[int] = []
        after_nucleus = nucleus < 0
        for kind, ei, hj in wa.ops:
            if ei is not None and ei >= nucleus:
                if ei == nucleus:
                    after_nucleus = True
                    continue
            if not after_nucleus or hj is None:
                continue
            if is_vowel(heard[hj].text) and (ei is None or not is_vowel(e[ei])):
                # a vowel in place of a final consonant is reported as that consonant missing,
                # except after t/d, where "next-uh" is the typical pattern
                if ei is None or e[ei] in ("t", "d"):
                    cand.append(hj)
        gap = self.al.gap(wi + 1)
        if wi == self.n - 1:
            # only a vowel right after the last word counts (not noise or extra words much later)
            last_t = heard[wa.hi - 1].end if wa.hi > wa.lo else None
            if gap and last_t is not None and is_vowel(heard[gap[0]].text) and heard[gap[0]].start - last_t < 0.3:
                cand.append(gap[0])
            return cand
        cand += [k for k in gap if is_vowel(heard[k].text)]
        nxt = self.al.words[wi + 1]
        for kind, ei, hj in nxt.ops:
            if ei is not None and kind != "del":
                break
            if kind == "ins" and is_vowel(heard[hj].text):
                cand.append(hj)
        if cand and nxt.expected and is_vowel(nxt.expected[0]):
            first = next((o for o in nxt.ops if o[1] == 0), None)
            if first and first[0] == "del":
                cand = cand[:-1]  # that vowel is the next word's own first sound
        return cand

    def uh(self, wi: int) -> None:
        e = self.al.words[wi].expected
        if not e or e[-1] not in _OBSTRUENTS or not self.reliable(wi):
            return  # after vowels, nasals and l the recognizer's extra vowels are mostly noise
        if wi + 1 < self.n and self.al.words[wi + 1].hi <= self.al.words[wi + 1].lo:
            return  # the next word was not heard: the gap holds whatever was said instead
        extra = self.inserted_vowels(wi)
        last = wi == self.n - 1
        label = f"{english.norm(self.words[wi])}‿{english.norm(self.words[wi + 1])}" if not last \
            else f"{english.norm(self.words[wi])}."
        span = (wi, wi) if last else (wi, wi + 1)
        if extra:
            v = self.al.heard[extra[0]].text
            self.add("uh", wi, False, f"extra /{v}/ after the final /{e[-1]}/ – “{english.norm(self.words[wi])}-uh”",
                     label=label, span=span, heard=v)
        elif e[-1] in ("t", "d"):
            self.add("uh", wi, True, f"clean /{e[-1]}/, no extra vowel", label=label, span=span)

    def stress_check(self, wi: int) -> None:
        verdict = self.ph["words"][wi].get("stress")
        if not verdict or not self.content[wi]:
            return
        e = self.al.words[wi].expected
        levels = self.ph["words"][wi].get("stress_levels") or []
        syl = english.syllabify(e)
        marks = []
        for a, b in syl:
            lv = levels[a:b]
            marks.append(("ˈ" if 1 in lv else "ˌ" if 2 in lv else "") + "".join(e[a:b]))
        k_exp, k_det = verdict["expected"], verdict.get("detected")
        extra = {"syllables": marks, "expected_syllable": k_exp, "detected_syllable": k_det,
                 "prominence": verdict.get("prominence")}
        pattern = "·".join(marks)
        if len(syl) != verdict["syllables"]:
            extra["syllables"] = None  # syllabification disagrees with the vowel count: show no syllables
        if verdict["match"] is True:
            self.add("stress", wi, True, pattern, **extra)
        elif verdict["match"] is False:
            said = marks[k_det].lstrip("ˈˌ") if k_det is not None and k_det < len(marks) else "?"
            self.add("stress", wi, False, f"you stressed the {_ordinal(k_det + 1)} syllable “{said}”; "
                     f"say {pattern}", **extra)
        elif verdict["syllables"] >= 3 and verdict.get("prominence"):
            self.add("stress", wi, None, f"could not tell which syllable you stressed; expected {pattern}", **extra)

    def rhythm(self) -> None:
        for wi, w in enumerate(self.words):
            forms = english.weak_strong(w)
            if not forms or self.before_pause(wi) or self.content[wi] or self.dictionary[wi] not in forms[0]:
                continue  # only where the recognizer expects the weak form (see the module notes)
            kind = self.pr["words"][wi].get("form")
            heard = "".join(self.ph["words"][wi]["heard"])
            weak = "".join(self.dictionary[wi])
            if kind == "weak":
                self.add("rhythm", wi, True, f"reduced /{heard}/")
            elif kind == "strong":
                self.add("rhythm", wi, False, f"full /{heard}/ – reduce it to /{weak}/", heard=heard)
        ratio = self.pr.get("function_ratio")
        n_function = sum(1 for wi, w in enumerate(self.pr["words"])
                         if not w["content"] and w["syllables"] == 1 and "dps" in w)
        if ratio is None or n_function < 3:
            return
        ref = self.ref.get("function_ratio")
        ok_limit = RATIO_OK if ref is None else max(RATIO_OK, ref + 0.15)
        bad_limit = RATIO_BAD if ref is None else max(min(RATIO_BAD, ref + 0.35), ok_limit + 0.1)
        speaker = f"speaker {ref:.2f}×" if ref is not None else "fluent speech about 0.65×"
        ok = True if ratio <= ok_limit else False if ratio >= bad_limit else None
        tail = {True: "", None: " – you could shorten the small words a bit more",
                False: " – say the small words (to, the, and, of) faster and softer"}[ok]
        self.add("rhythm", 0, ok, f"small words took {ratio:.2f}× the time of content words ({speaker}){tail}",
                 label="whole sentence", span=(0, self.n - 1), ratio=ratio, reference=ref)

    def pauses(self) -> None:
        """Pauses where the speaker makes them (or, without a speaker, at punctuation)."""
        punct = {i for i in range(self.n - 1) if english.breaks_after(self.words[i])}
        if "pauses_after" in self.ref:
            required = {i for i in self.ref["pauses_after"] if 0 <= i < self.n - 1}
            optional = punct - required  # commas the speaker ran through
        else:
            required, optional = punct, set()
        expected = required | optional
        timed = [self.al.word_time(i) is not None for i in range(self.n)]
        for i in sorted(expected):
            if not (timed[i] and timed[i + 1]):
                continue
            word = english.norm(self.words[i])
            why = "the speaker pauses here" if i in required and "pauses_after" in self.ref else "punctuation"
            label = f"{self.words[i]} ‖ {self.words[i + 1]}"
            if self.pr["words"][i].get("pause_after"):
                self.add("pause", i, True, f"paused after “{word}” ({why})", label=label, span=(i, i + 1))
            elif i in required:
                self.add("pause", i, False, f"no pause after “{word}” – stop briefly here ({why})",
                         label=label, span=(i, i + 1))
        for i, w in enumerate(self.pr["words"][:-1]):
            if w.get("pause_after") and i not in expected and not w["content"] and timed[i + 1]:
                self.add("pause", i, False, f"a pause after “{english.norm(self.words[i])}” splits the phrase – "
                         f"keep “{english.norm(self.words[i])} {english.norm(self.words[i + 1])}” together",
                         label=f"{self.words[i]} ‖ {self.words[i + 1]}", span=(i, i + 1))

    def focus(self) -> None:
        rng = self.pr.get("pitch_range")
        if rng is None or self.n < 4:
            return
        ref = self.ref.get("pitch_range")
        ok = rng >= MIN_PITCH_RANGE and (ref is None or rng >= 0.6 * ref or rng >= 5.0)
        speaker = f" (speaker {ref:.1f})" if ref is not None else ""
        self.add("focus", 0, ok, f"your pitch moved over {rng:.1f} semitones{speaker}"
                 + ("" if ok else " – too flat: raise your voice on the focus words"),
                 label="intonation", span=(0, self.n - 1), range=rng, reference=ref)

    def run(self) -> dict:
        for wi in range(self.n):
            if self.al.words[wi].hi <= self.al.words[wi].lo:
                continue  # nothing heard for this word: the word-level score already says so
            if self.reliable(wi):
                self.th(wi)
                self.final(wi)
                self.sz(wi)
                self.r(wi)
            self.uh(wi)
            self.stress_check(wi)
        self.rhythm()
        self.pauses()
        self.focus()
        summary = {}
        for cat in CATEGORIES:
            decided = [c["ok"] for c in self.checks if c["cat"] == cat and c["ok"] is not None]
            if decided:
                summary[cat] = [sum(decided), len(decided)]
        return {"checks": self.checks, "summary": summary}


def coach(words: Sequence[str], al: SentenceAlignment, phonemes: dict, prosody: dict,
          reference: Optional[dict] = None, dictionary: Optional[Sequence[Sequence[str]]] = None) -> dict:
    """All targeted checks for one recording.

    ``phonemes`` is the result of :func:`phoneme_scoring.score_phonemes` and
    ``prosody`` that of :func:`prosody.analyze` for the same alignment;
    ``reference`` optionally carries the speaker's ``pauses_after`` (word
    indices), ``function_ratio`` and ``pitch_range`` for the same sentence;
    ``dictionary`` is espeak's pronunciation of every word in context.
    """
    return _Coach(words, al, phonemes, prosody, reference, dictionary).run()
