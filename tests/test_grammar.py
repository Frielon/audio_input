"""Edit extraction and labelling (ERRANT + spaCy's small model; the correction model is not loaded)."""
import pytest

from audio_input import grammar
from audio_input.grammar import Edit, GrammarChecker, apply_edits, describe, keep


@pytest.fixture(scope="module")
def checker():
    return GrammarChecker()


def edits(checker, sentence, corrected):
    return grammar.extract_edits(checker.annotator, sentence, corrected)


def test_subject_verb_agreement(checker):
    (e,) = edits(checker, "He go to school every day.", "He goes to school every day.")
    assert (e.original, e.correction, e.kind, e.label, e.severity) == ("go", "goes", "R:VERB:SVA",
                                                                         "subject–verb agreement", "error")
    assert e.message == "“go” → “goes”"
    assert "He go to school every day."[e.start:e.end] == "go"


def test_missing_article_is_an_insertion(checker):
    s = "We use calculator to compute the mean."
    (e,) = edits(checker, s, "We use a calculator to compute the mean.")
    assert e.op == "missing" and e.start == e.end == s.index("calculator")
    assert e.label == "article" and e.message == "add “a”"


def test_unnecessary_preposition(checker):
    s = "I want to discuss about the problem."
    (e,) = edits(checker, s, "I want to discuss the problem.")
    assert e.op == "unnecessary" and e.kind == "U:PREP" and e.message == "drop “about”"
    assert apply_edits(s, [e]) == "I want to discuss the problem."


def test_contraction_inside_a_word(checker):
    s = "She don't like apples."
    (e,) = edits(checker, s, "She doesn't like apples.")
    assert e.kind == "R:VERB:SVA" and s[e.start:e.end] == "do"
    assert apply_edits(s, [e]) == "She doesn't like apples."


def test_punctuation_and_casing_are_whisper_s_not_yours(checker):
    assert edits(checker, "yesterday I went home", "Yesterday, I went home.") == []


def test_casual_subjectless_phrases_are_fine(checker):
    assert edits(checker, "Okay, sounds good.", "Okay, it sounds good.") == []
    (e,) = edits(checker, "Is very good idea.", "It is a very good idea.")[:1]
    assert e.correction.lower().startswith("it")  # a real missing subject is still reported


def test_dropped_fillers_are_not_grammar(checker):
    assert edits(checker, "So, um, I think we can go.", "So, I think we can go.") == []


def test_modal_change_is_only_a_suggestion():
    assert describe("R:VERB:TENSE", "can", "could")[2] == "suggestion"
    assert describe("R:VERB:TENSE", "meet", "met")[2] == "error"
    assert describe("R:OTHER", "do the homework", "finish the homework")[2] == "suggestion"
    assert describe("U:OTHER", "um", "")[:1] == ("filler word",)
    assert describe("R:DET", "this", "that")[0] == "determiner"


def test_keep():
    assert not keep("M:PUNCT", "Hi there", 2)
    assert not keep("R:ORTH", "hi", 0)
    assert not keep("R:SPELL", "teh", 0)
    assert keep("R:VERB:SVA", "He go", 3)


def test_apply_edits_insert_delete_replace():
    text = "Yesterday I meet my advisor and he say we should use calculator."
    es = [
        Edit(12, 16, "meet", "met", "R:VERB:TENSE", "", "", "error"),
        Edit(35, 38, "say", "said", "R:VERB:TENSE", "", "", "error"),
        Edit(53, 53, "", "a", "M:DET", "", "", "error"),
    ]
    assert apply_edits(text, es) == "Yesterday I met my advisor and he said we should use a calculator."
    assert apply_edits("Have I went home?", [Edit(0, 4, "Have", "", "U:VERB", "", "", "error")]) == "I went home?"
    assert apply_edits("I asked yesterday?", [Edit(17, 17, "", "for", "M:PREP", "", "", "error")]) == "I asked yesterday for?"


def test_check_with_a_stub_model(checker, monkeypatch):
    fixes = {"He go to school.": "He goes to school.", "She like it.": "She likes it."}
    monkeypatch.setattr(checker, "correct", lambda sents: [[fixes.get(s, s)] for s in sents])
    text = "He go to school. She like it. Okay."
    r = checker.check(text)
    assert r.corrected == "He goes to school. She likes it. Okay."
    assert [text[e.start:e.end] for e in r.edits] == ["go", "like"]
    assert len(r.errors) == 2


def test_wild_corrections_are_ignored(checker, monkeypatch):
    monkeypatch.setattr(checker, "correct", lambda sents: [["Completely unrelated output from the model."]] * len(sents))
    r = checker.check("I went to the shop to buy some milk.")
    assert r.edits == [] and r.corrected == "I went to the shop to buy some milk."


def test_the_minimal_correction_wins_over_a_rewrite(checker):
    s = "He think the method is correct, but the case need a calculator."
    cands = ["He thinks the method is correct, but in this case, a calculator is needed.",
             "He thinks the method is correct, but the case needs a calculator.",
             "He thinks the method is correct, but the case need a calculator."]
    fixes = [(e.original, e.correction) for e in checker.choose(s, cands)]
    assert fixes == [("think", "thinks"), ("need", "needs")]


def test_a_complete_fix_beats_a_partial_one(checker):
    s = "He think the case need a calculator."
    cands = ["He thinks the case needs a calculator.", "He thinks the case need a calculator."]
    assert len(checker.choose(s, cands)) == 2
    assert len(checker.choose(s, cands[::-1])) == 2


def test_unchanged_best_guess_means_no_error(checker):
    s = "Maybe we can meet on Tuesday."
    assert checker.choose(s, [s, "Maybe we could meet on Tuesday."]) == []
    assert checker.choose(s, ["Maybe, we can meet on Tuesday.", "Maybe we could meet on Tuesday."]) == []
