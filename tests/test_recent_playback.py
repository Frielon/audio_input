"""The last recordings, playback helpers, the model voice and the overlay's buttons."""
import shutil

import numpy as np
import pytest

from audio_input import player
from audio_input.recent import Recent


def tone(seconds, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_keeps_the_last_n(tmp_path, monkeypatch):
    r = Recent(tmp_path, keep=3)
    stamps = iter(f"20261026-1200{i:02d}" for i in range(10))
    monkeypatch.setattr("audio_input.recent.time.strftime", lambda fmt: next(stamps))
    ids = [r.add({"text": f"sentence {i}"}, tone(0.5)) for i in range(5)]
    assert r.ids() == ids[-3:]
    assert not list(tmp_path.glob(f"{ids[0]}.*"))
    assert r.position(ids[-1]) == (3, 3) and r.neighbour(ids[-1], -1) == ids[-2] and r.neighbour(ids[-1], 1) is None
    assert r.load(ids[-1])["text"] == "sentence 4"
    audio = r.audio(ids[-1])
    assert len(audio) == 8000 and abs(float(np.abs(audio).max()) - 0.3) < 0.01
    r.save(ids[-1], {"text": "sentence 4", "model_ready": True})
    assert r.load(ids[-1])["model_ready"]
    assert r.model_audio(ids[-1]) is None
    r.save_model_audio(ids[-1], tone(0.2))
    r2 = Recent(tmp_path, keep=3)  # after a restart
    assert r2.ids() == ids[-3:] and len(r2.model_audio(ids[-1])) == 3200


def test_nothing_kept_when_disabled(tmp_path):
    assert Recent(tmp_path, keep=0).add({"text": "x"}, tone(0.1)) is None


def test_segment_and_join():
    audio = tone(2.0)
    seg = player.segment(audio, 0.5, 1.0)
    assert len(seg) == int(1.1 * 16000) - int(0.44 * 16000)  # 60 ms before, 100 ms after the word
    both = player.join([tone(1.0), None, tone(0.5)], gap=0.5)
    assert len(both) == 16000 + 8000 + 8000


@pytest.mark.skipif(shutil.which("say") is None, reason="needs macOS `say`")
def test_model_voice_is_cached(tmp_path):
    from audio_input.voice import Voice

    v = Voice(tmp_path, "Samantha")
    a = v.speak("He goes to school.")
    assert 0.6 < len(a) / 16000 < 3.0
    assert len(list(tmp_path.glob("*.wav"))) == 1
    assert np.allclose(v.speak("He goes to  school."), a, atol=1e-4)  # same text: from the cache


def test_overlay_buttons_play_the_right_audio(tmp_path, monkeypatch):
    pytest.importorskip("AppKit")
    from audio_input import app as app_mod
    from audio_input import config

    played = []
    a = app_mod.App(config.Settings(history=False))
    a.recent = Recent(tmp_path)
    a.player = type("P", (), {"play": lambda self, x: played.append(None if x is None else len(x)),
                              "stop": lambda self: None})()
    a.engine.voice = type("V", (), {"speak": lambda self, text: tone(0.3)})()
    me, model = tone(1.0), tone(0.8)
    rid = a.recent.add({"text": "He go home.", "words": [{"w": "He", "t": [0.1, 0.3], "mt": [0.0, 0.2]},
                                                         {"w": "go", "t": [0.3, 0.6]}, {"w": "home."}]}, me)
    a.recent.save_model_audio(rid, model)
    for act in ("play:me", "play:model", "play:compare", "w:0", "m:0", "m:1", "t:0.2:0.4", "w:2"):
        a._play(rid, act)
    gap = int(0.45 * 16000)
    word = len(player.segment(me, 0.1, 0.3))
    assert played == [16000, 12800, 12800 + gap + 16000, word, len(player.segment(model, 0.0, 0.2)), 4800,
                      len(player.segment(me, 0.2, 0.4)), None]  # "home." has no time: nothing to play
