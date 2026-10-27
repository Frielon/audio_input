from audio_input import feedback
from audio_input.grammar import Edit, GrammarResult


def phones(word, ipa, status=None, heard=None, stress=None, acc=100):
    status = status or ["ok"] * len(ipa)
    out = []
    for i, p in enumerate(ipa):
        d = {"p": p, "status": status[i]}
        if heard and status[i] == "sub":
            d["heard"] = heard[i]
        out.append(d)
    return {"text": word, "expected": ipa, "phones": out, "accuracy": acc,
            "heard": [heard[i] if heard and status[i] == "sub" else p for i, p in enumerate(ipa) if status[i] != "del"],
            "stress_levels": stress or [0] * len(ipa)}


def pron_for(words):
    return {
        "phonemes": {"accuracy": 92, "stress_correct": 1, "stress_checked": 1, "words": words},
        "prosody": {"words": [{"content": w["text"].lower() not in ("i", "the", "to")} for w in words]},
        "coach": {"checks": [{"cat": "th", "word": 1, "ok": False, "label": "think",
                              "detail": "sounded like /s/ – “sink”"},
                             {"cat": "final", "word": 3, "ok": True, "label": "go", "detail": "final"}],
                  "summary": {"th": [0, 1], "final": [1, 1]}},
    }


TEXT = "I think he go to school."
WORDS = [
    phones("I", ["aɪ"]),
    phones("think", ["θ", "ɪ", "ŋ", "k"], ["sub", "ok", "ok", "ok"], ["s", "ɪ", "ŋ", "k"], [0, 1, 0, 0], acc=75),
    phones("he", ["h", "iː"]),
    phones("go", ["ɡ", "oʊ"]),
    phones("to", ["t", "ə"]),
    phones("school.", ["s", "k", "uː", "l"]),
]


def test_word_spans():
    assert feedback.word_spans("Hi — there, you.") == [(0, 2), (5, 11), (12, 16)]
    assert feedback.word_spans("Hi — there", ["Hi", "there"]) == [(0, 2), (5, 10)]


def test_build_marks_sounds_and_grammar():
    g = GrammarResult(TEXT, "I think he goes to school.",
                      [Edit(11, 13, "go", "goes", "R:VERB:SVA", "subject–verb agreement", "“go” → “goes”", "error")])
    p = feedback.build(TEXT, pron_for(WORDS), g)
    assert [w["w"] for w in p["words"]] == TEXT.split()
    think = p["words"][1]
    assert think["sound"] == "warn" and think["heard"] == "sɪŋk"
    assert think["ipa"][0] == {"p": "θ", "s": "sub", "h": "s"} and think["ipa"][1]["st"] == 1
    assert p["words"][3]["g"] == [{"id": 0, "op": "replace", "severity": "error", "fix": "goes"}]
    assert p["words"][0]["fn"] is True and p["words"][1]["fn"] is False
    kinds = [(i["kind"], i["label"]) for i in p["issues"]]
    assert kinds == [("grammar", "subject–verb agreement"), ("sound", "TH sounds")]
    assert p["corrected"] == "I think he goes to school."
    assert [m["t"] for m in p["corrected_marks"] if m["c"]] == ["goes"]
    assert p["summary"]["grammar"] == 1 and p["summary"]["sounds"] == 92
    assert p["grammar_checked"] and p["analyzed"]


def test_missing_word_becomes_an_insert():
    text = "We use calculator."
    g = GrammarResult(text, "We use a calculator.", [Edit(7, 7, "", "a", "M:DET", "article", "add “a”", "error")])
    p = feedback.build(text, None, g)
    assert p["inserts"] == [{"before": 2, "text": "a", "g": 0, "severity": "error"}]
    assert "ipa" not in p["words"][0]


def test_suggestions_come_last_and_unnecessary_words_are_marked():
    text = "Um, I want to discuss about it."
    g = GrammarResult(text, "", [
        Edit(14, 21, "discuss", "talk", "R:OTHER", "wording", "“discuss” → “talk”", "suggestion"),
        Edit(22, 27, "about", "", "U:PREP", "preposition", "drop “about”", "error"),
    ])
    p = feedback.build(text, None, g)
    assert [(i["kind"], i["label"]) for i in p["issues"]] == [
        ("grammar", "preposition"), ("suggestion", "wording"), ("suggestion", "filler words")]
    assert p["issues"][-1]["text"] == "“um”" and p["words"][0]["filler"]
    assert p["words"][5]["g"][0]["op"] == "unnecessary"
    assert p["summary"] == {"sounds": None, "stress": None, "grammar": 1, "suggestions": 1, "fillers": 1,
                            "checks": {}}


def test_sound_analysis_failure_is_reported():
    p = feedback.build("Hello there.", {"error": "model missing"}, None)
    assert p["sound_error"] == "model missing" and not p["analyzed"]


def test_focus_areas_come_first_with_the_tip():
    checks = [
        {"cat": "uh", "word": 3, "ok": False, "label": "go‿to", "detail": "extra /ə/"},
        {"cat": "th", "word": 1, "ok": False, "label": "think", "detail": "sounded like /s/", "t": [0.2, 0.5]},
        {"cat": "final", "word": 5, "ok": True, "label": "school", "detail": "final /l/"},
        {"cat": "stress", "word": 7, "ok": None, "label": "calculator", "detail": "could not tell",
         "prominence": [0.2, 0.5, 0.2, 0.1], "expected_syllable": 0},
    ]
    rows = feedback.coach_rows(checks, ["final", "th", "stress"])
    assert [r["cat"] for r in rows] == ["final", "th", "stress", "uh"]
    final, th, stress, uh = rows
    assert final["bad"] == 0 and final["passed"] == ["school"] and final["tip"] is None
    assert th["bad"] == 1 and th["tip"].startswith("Tongue tip") and th["items"][0]["t"] == [0.2, 0.5]
    assert stress["items"][0]["prom"] == [0.2, 0.5, 0.2, 0.1] and stress["items"][0]["exp"] == 0
    assert uh["focus"] is False


def test_syllables_times_and_stress_for_the_overlay():
    calc = phones("calculator", ["k", "æ", "l", "k", "j", "ʊ", "l", "eɪ", "ɾ", "ɚ"],
                  stress=[0, 1, 0, 0, 0, 0, 0, 2, 0, 0])
    calc["stress"] = {"syllables": 4, "expected": 0, "detected": 1, "match": False, "prominence": [0.2, 0.5, 0.2, 0.1]}
    pron = {"phonemes": {"accuracy": 100, "words": [phones("a", ["ə"]), calc]},
            "prosody": {"words": [{"content": False, "start": 0.1, "end": 0.2},
                                  {"content": True, "start": 0.2, "end": 0.9}]},
            "coach": {"checks": [], "summary": {}},
            "syllables": [[(0, 1)], [(0, 3), (3, 6), (6, 8), (8, 10)]]}
    p = feedback.build("a calculator", pron, None)
    w = p["words"][1]
    assert w["syl"] == [0, 3, 6, 8] and "syl" not in p["words"][0]
    assert w["t"] == [0.2, 0.9]
    assert w["stress"] == {"ok": False, "expected": 0, "said": 1, "n": 4, "prom": [0.2, 0.5, 0.2, 0.1]}


def test_model_voice_is_matched_word_by_word():
    pron = pron_for(WORDS)
    pron["phonemes"]["words"][1]["stress"] = {"syllables": 2, "expected": 0, "detected": 0, "match": True,
                                              "prominence": [0.7, 0.3]}
    p = feedback.build(TEXT, pron, None)
    ref = {"words": [{"start": 0.0, "end": 0.1}, {"start": 0.1, "end": 0.4}, {"start": 0.4, "end": 0.5},
                     {"start": 0.5, "end": 0.8}, {"start": 0.8, "end": 0.9}, {"start": 0.9, "end": 1.3}],
           "stress": {"1": [0.8, 0.2]}}
    # the model reads the corrected sentence: "goes" is not "go", the other words line up
    feedback.attach_model(p, ["I", "think", "he", "goes", "to", "school."], ref)
    assert p["words"][1]["mt"] == [0.1, 0.4] and p["words"][1]["stress"]["model"] == [0.8, 0.2]
    assert "mt" not in p["words"][3] and p["words"][5]["mt"] == [0.9, 1.3]
    assert p["model_ready"]
