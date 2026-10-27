"""Bits of English phonology the feedback needs: function words, weak forms, thought groups.

Fluent speakers *reduce* function words (``and`` -> ``ən``, ``to`` -> ``tə``,
``can`` -> ``kən``) and stress content words; learners often give every word
the same weight.  The tables below let the phoneme scorer accept the reduced
pronunciations (they are not errors) and let the coach notice when a function
word was said in its full, strong form in the middle of a phrase.

Phones use the espeak IPA inventory of the phoneme recognizer.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

from .phoneme_engine import is_vowel

FUNCTION_WORDS = frozenset("""
a an the to of and or but nor for from at as than in on by with into onto
is are was were be been am do does did have has had can could will would shall should may might must
i you he she it we they me him her us them my your his its our their that which who if
i'm you're he's she's it's we're they're that's i've you've we've they've i'll you'll we'll they'll
i'd you'd he'd she'd we'd they'd there's
""".split())

# word -> (weak forms, strong forms).  Only words whose weak form is the normal
# pronunciation in the middle of a phrase are listed.
FORMS: dict[str, tuple[list[list[str]], list[list[str]]]] = {
    "a": ([["ə"]], [["eɪ"]]),
    "an": ([["ə", "n"]], [["æ", "n"]]),
    "the": ([["ð", "ə"], ["ð", "ɪ"], ["ð", "i"]], [["ð", "iː"]]),
    "to": ([["t", "ə"], ["t", "ʊ"]], [["t", "uː"]]),
    "of": ([["ə", "v"], ["ə"]], [["ʌ", "v"], ["ɑː", "v"]]),
    "and": ([["ə", "n", "d"], ["ə", "n"], ["n"], ["n̩"]], [["æ", "n", "d"], ["æ", "n"]]),
    "or": ([["ɚ"]], [["ɔːɹ"], ["oːɹ"]]),
    "but": ([["b", "ə", "t"]], [["b", "ʌ", "t"]]),
    "for": ([["f", "ɚ"]], [["f", "ɔːɹ"], ["f", "oːɹ"]]),
    "from": ([["f", "ɹ", "ə", "m"]], [["f", "ɹ", "ʌ", "m"], ["f", "ɹ", "ɑː", "m"]]),
    "at": ([["ə", "t"]], [["æ", "t"]]),
    "as": ([["ə", "z"]], [["æ", "z"]]),
    "than": ([["ð", "ə", "n"]], [["ð", "æ", "n"]]),
    "that": ([["ð", "ə", "t"]], [["ð", "æ", "t"]]),
    "can": ([["k", "ə", "n"], ["k", "n̩"]], [["k", "æ", "n"]]),
    "was": ([["w", "ə", "z"]], [["w", "ʌ", "z"], ["w", "ɑː", "z"]]),
    "were": ([["w", "ɚ"]], [["w", "ɜː"]]),
    "are": ([["ɚ"]], [["ɑːɹ"]]),
    "do": ([["d", "ə"]], [["d", "uː"]]),
    "does": ([["d", "ə", "z"]], [["d", "ʌ", "z"]]),
    "you": ([["j", "ə"]], [["j", "uː"]]),
    "your": ([["j", "ɚ"]], [["j", "ʊɹ"], ["j", "ɔːɹ"], ["j", "oːɹ"]]),
    "them": ([["ð", "ə", "m"], ["ə", "m"]], [["ð", "ɛ", "m"]]),
    "some": ([["s", "ə", "m"]], [["s", "ʌ", "m"]]),
    "us": ([["ə", "s"]], [["ʌ", "s"]]),
    "have": ([["h", "ə", "v"], ["ə", "v"]], [["h", "æ", "v"]]),
    "has": ([["h", "ə", "z"], ["ə", "z"]], [["h", "æ", "z"]]),
    "had": ([["h", "ə", "d"], ["ə", "d"]], [["h", "æ", "d"]]),
    "would": ([["w", "ə", "d"], ["ə", "d"]], [["w", "ʊ", "d"]]),
    "could": ([["k", "ə", "d"]], [["k", "ʊ", "d"]]),
    "should": ([["ʃ", "ə", "d"]], [["ʃ", "ʊ", "d"]]),
    "his": ([["ɪ", "z"]], [["h", "ɪ", "z"]]),
    "him": ([["ɪ", "m"]], [["h", "ɪ", "m"]]),
    "her": ([["ɚ"], ["h", "ɚ"]], [["h", "ɜː"]]),
    "will": ([["ə", "l"], ["əl"]], [["w", "ɪ", "l"]]),
}

# Words that are often pronounced with a vowel the recognizer hears as reduced even
# though they are content words; their forms are accepted but never flagged.
_EXTRA_FORMS: dict[str, list[list[str]]] = {
    "just": [["dʒ", "ə", "s", "t"], ["dʒ", "ɪ", "s", "t"], ["dʒ", "ə", "s"], ["dʒ", "ʌ", "s"]],
}

_PUNCT_BREAK_RE = re.compile(r"[,;:.!?…—–)\]\"”]+$")
_WORD_RE = re.compile(r"[^a-z']")


def norm(word: str) -> str:
    """Lowercase letters and apostrophes only: ``“Today,”`` -> ``today``."""
    return _WORD_RE.sub("", word.replace("’", "'").lower()).strip("'")


def is_function_word(word: str, stressed_in_context: bool = False) -> bool:
    """Function words are normally unstressed and reduced.

    ``that`` is a function word as a conjunction or relative pronoun but a
    stressed content word as a demonstrative; espeak's sentence-level stress
    (``stressed_in_context``) tells the two apart well enough.
    """
    w = norm(word)
    if w == "that" and stressed_in_context:
        return False
    return w in FUNCTION_WORDS


def alternative_forms(word: str) -> list[list[str]]:
    """Every accepted pronunciation of a function word (weak and strong)."""
    w = norm(word)
    if w in FORMS:
        weak, strong = FORMS[w]
        return [list(f) for f in weak + strong]
    return [list(f) for f in _EXTRA_FORMS.get(w, [])]


def weak_strong(word: str) -> Optional[tuple[list[list[str]], list[list[str]]]]:
    return FORMS.get(norm(word))


def breaks_after(word: str) -> bool:
    """Punctuation after a word marks the end of a thought group."""
    return bool(_PUNCT_BREAK_RE.search(word.strip()))


def thought_groups(words: Sequence[str], extra_breaks: Sequence[int] = ()) -> list[tuple[int, int]]:
    """``[(first, last)]`` word index ranges, split after punctuation and at ``extra_breaks``.

    ``extra_breaks`` are indices of words followed by a pause in a reference
    recording (the speaker's own phrasing).
    """
    n = len(words)
    if not n:
        return []
    cut_after = {i for i, w in enumerate(words[:-1]) if breaks_after(w)} | {i for i in extra_breaks if 0 <= i < n - 1}
    groups, start = [], 0
    for i in range(n):
        if i in cut_after or i == n - 1:
            groups.append((start, i))
            start = i + 1
    return groups


def default_focus(words: Sequence[str], group: tuple[int, int], content: Sequence[bool]) -> Optional[int]:
    """The last content word of a thought group: where new information usually goes."""
    first, last = group
    for i in range(last, first - 1, -1):
        if content[i]:
            return i
    return None


# Consonant clusters that can start an English syllable (in the recognizer's phones).
_ONSETS = {tuple(o.split()) for o in (
    "p b t d k ɡ f v θ ð s z ʃ ʒ h tʃ dʒ m n l ɹ w j ɾ ʔ".split() + [
        "p ɹ", "p l", "p j", "b ɹ", "b l", "b j", "t ɹ", "t w", "d ɹ", "d w", "k ɹ", "k l", "k w", "k j",
        "ɡ ɹ", "ɡ l", "ɡ w", "ɡ j", "f ɹ", "f l", "f j", "v j", "θ ɹ", "θ w", "ʃ ɹ", "m j", "h j",
        "s p", "s t", "s k", "s m", "s n", "s l", "s w", "s f",
        "s p ɹ", "s p l", "s p j", "s t ɹ", "s k ɹ", "s k w", "s k l", "s k j",
    ])}


def syllabify(phones: Sequence[str]) -> list[tuple[int, int]]:
    """Split a word's phones into syllables (``[(start, end)]`` index ranges, end exclusive).

    Every vowel is a nucleus; the consonants between two vowels go to the next
    syllable as long as they form a possible English onset (``kæl·kjʊ``).
    """
    nuclei = [i for i, p in enumerate(phones) if is_vowel(p)]
    if len(nuclei) < 2:
        return [(0, len(phones))] if phones else []
    starts = [0]
    for a, b in zip(nuclei, nuclei[1:]):
        cluster = list(phones[a + 1:b])
        split = len(cluster)  # default: everything is coda of the first syllable
        for k in range(len(cluster) + 1):
            if not cluster[k:] or tuple(cluster[k:]) in _ONSETS:
                split = k
                break
        starts.append(a + 1 + split)
    ends = starts[1:] + [len(phones)]
    return list(zip(starts, ends))
