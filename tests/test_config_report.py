import json

import pytest

from audio_input import config, report
from audio_input.history import History


def test_defaults_file_and_overrides(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('hotkey = "f13"\nvocabulary = ["PyTorch", "Berkeley"]\noverlay_seconds = 10\n'
                    'input_device = 5\nphonemes = "off"\n')
    s = config.load(path, {"insert": "type", "grammar": None})
    assert s.hotkey == "f13" and s.vocabulary == ["PyTorch", "Berkeley"] and s.overlay_seconds == 10.0
    assert s.input_device == "5" and s.phonemes is False and s.insert == "type" and s.grammar is True


def test_save_hotkey(tmp_path):
    path = tmp_path / "config.toml"
    config.save_hotkey("right_option", path)
    assert config.load(path).hotkey == "right_option"
    path.write_text('# mine\nvocabulary = ["Adam"]\nhotkey = "f13"\n')
    config.save_hotkey("left_option", path)
    s = config.load(path)
    assert s.hotkey == "left_option" and s.vocabulary == ["Adam"] and path.read_text().startswith("# mine")


def test_bad_values(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('insert = "telepathy"\n')
    with pytest.raises(ValueError):
        config.load(path)
    with pytest.raises(FileNotFoundError):
        config.load(tmp_path / "missing.toml")


def test_report_counts_what_keeps_coming_back(tmp_path):
    h = History(tmp_path)
    payload = {
        "text": "He go to school.", "corrected": "He goes to school.",
        "grammar_edits": [{"kind": "R:VERB:SVA", "label": "subject–verb agreement", "original": "go",
                           "correction": "goes", "severity": "error"}],
        "issues": [{"kind": "sound", "cat": "th", "word": "think", "text": "sounded like /s/"}],
        "words": [{"w": "think", "acc": 75, "heard": "sɪŋk", "sound": "warn"}],
        "summary": {"checks": {"th": [0, 1], "final": [2, 2]}},
    }
    h.add(payload, app="Notes", seconds=2.0)
    h.add(payload, app="Notes", seconds=2.0)
    h.add({"text": "All good here.", "issues": [], "words": [], "summary": {"checks": {"final": [1, 1]}}})
    s = report.summarize(list(h.records()))
    assert s["utterances"] == 3
    assert s["grammar"] == [("subject–verb agreement", 2)]
    assert s["checks"]["final"] == [5, 5] and s["checks"]["th"] == [0, 2]
    assert s["weak_words"] == [("think", 2)]
    assert s["practice"] == ["He goes to school."]
    assert "subject–verb agreement" in report.text_report(s)
    out = report.write_html(tmp_path)
    assert "He goes to school." in out.read_text()
    line = json.loads(h.path.read_text().splitlines()[0])
    assert line["app"] == "Notes" and line["grammar"][0]["correction"] == "goes"
