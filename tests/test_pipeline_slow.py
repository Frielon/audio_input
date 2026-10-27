"""The whole pipeline on synthetic speech (macOS `say`): Whisper, the parrot sound analysis, the grammar model.

Loads every model (a few seconds when cached; the first run downloads ~5 GB), so it only runs with
AUDIO_INPUT_SLOW_TESTS=1.
"""
import os
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("AUDIO_INPUT_SLOW_TESTS") != "1" or shutil.which("say") is None,
                                reason="set AUDIO_INPUT_SLOW_TESTS=1 (macOS) to run the models")


@pytest.fixture(scope="module")
def engine():
    from audio_input import config
    from audio_input.pipeline import Engine

    e = Engine(config.load(overrides={}))
    for fut in e.warm_up_feedback():
        assert fut.result() is None
    yield e
    e.shutdown()


def synth(tmp_path, text):
    from faster_whisper import decode_audio

    wav = tmp_path / "speech.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(wav), "--data-format=LEI16@16000", text], check=True)
    return decode_audio(str(wav), sampling_rate=16000)


def test_grammar_and_sounds(engine, tmp_path):
    # ("My advisor ask me" would not do: "ask me" and "asked me" sound the same, and Whisper writes "asked")
    p = engine.process(synth(tmp_path, "She don't like apples, and he go to school every day."))
    assert "go to school" in p["text"]
    fixes = {(e["original"], e["correction"]) for e in p["grammar_edits"]}
    assert ("go", "goes") in fixes and ("do", "does") in fixes
    assert "he goes to school" in p["corrected"]
    assert p["summary"]["sounds"] >= 80
    assert all("ipa" in w for w in p["words"])


def test_fillers_are_kept_for_feedback(engine, tmp_path):
    p = engine.process(synth(tmp_path, "Um, so I think, uh, we should go with the second option."))
    assert p["summary"]["fillers"] >= 1
    assert p["issues"][-1]["label"] == "filler words"


def test_correct_sentence_has_no_grammar_errors(engine, tmp_path):
    p = engine.process(synth(tmp_path, "Let me know if you have any questions about the project."))
    assert p["summary"]["grammar"] == 0 and p["corrected"] is None


def test_silence(engine):
    import numpy as np

    assert engine.process(np.zeros(16000 * 2, dtype=np.float32)) is None
