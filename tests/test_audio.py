"""Recorder helpers and speech trimming (Silero VAD; no Whisper model needed)."""
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

from audio_input import recorder, speech


def test_resample_keeps_length_and_pitch():
    sr = 48000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    y = recorder.resample(x, sr, 16000)
    assert abs(len(y) - 16000) <= 1
    peak = np.argmax(np.abs(np.fft.rfft(y))) * 16000 / len(y)
    assert abs(peak - 440) < 3


def test_resolve_device(monkeypatch):
    devices = [{"index": 2, "name": "MacBook Pro Microphone", "channels": 1, "samplerate": 48000, "default": True},
               {"index": 5, "name": "Logitech BRIO", "channels": 2, "samplerate": 48000, "default": False}]
    monkeypatch.setattr(recorder, "input_devices", lambda: devices)
    assert recorder.resolve_device(None) is None
    assert recorder.resolve_device("brio") == 5
    assert recorder.resolve_device("5") == 5
    with pytest.raises(ValueError):
        recorder.resolve_device("AirPods")


class FakeSoundDevice:
    """Just enough of sounddevice to see what Recorder.start() asks of PortAudio."""

    class PortAudioError(Exception):
        pass

    def __init__(self):
        self.calls = []
        self.default = types.SimpleNamespace(device=(2, 1))
        self._initialized = 1

    def stop(self):
        self.calls.append("stop playback")

    def _terminate(self):
        self.calls.append("terminate")
        self._initialized -= 1

    def _initialize(self):
        self.calls.append("initialize")
        self._initialized += 1

    def query_devices(self, index):
        return {"name": f"device {index}", "default_samplerate": 48000.0}

    def InputStream(self, device=None, **kwargs):
        self.calls.append(f"open {device}")
        return types.SimpleNamespace(start=lambda: self.calls.append("start"), stop=lambda: None, close=lambda: None)


@pytest.fixture
def fake_sd(monkeypatch):
    sd = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.delenv("AUDIO_INPUT_FAKE_MIC", raising=False)
    return sd


def test_default_microphone_is_looked_up_for_every_recording(fake_sd):
    # PortAudio reads the devices once; restarting it makes a changed macOS default count
    rec = recorder.Recorder()
    for _ in range(2):
        rec.start()
        rec.stop()
    assert fake_sd.calls == ["stop playback", "terminate", "initialize", "open None", "start"] * 2


def test_chosen_microphone_leaves_portaudio_alone(fake_sd, monkeypatch):
    monkeypatch.setattr(recorder, "input_devices", lambda: [
        {"index": 5, "name": "Logitech BRIO", "channels": 2, "samplerate": 48000, "default": False}])
    rec = recorder.Recorder("brio")
    rec.start()
    rec.stop()
    assert fake_sd.calls == ["open 5", "start"]


def test_restart_after_a_failed_restart(fake_sd):
    fake_sd._initialized = 0  # PortAudio was terminated, and initializing it again failed
    recorder._restart_portaudio(fake_sd)
    assert fake_sd.calls == ["stop playback", "initialize"]


def test_playback_waits_for_a_portaudio_restart():
    from audio_input.player import Player

    assert Player()._lock is recorder.PORTAUDIO_LOCK


def test_silence_is_not_speech():
    assert speech.trim_to_speech(np.zeros(16000 * 2, dtype=np.float32)) is None
    noise = (np.random.default_rng(0).standard_normal(16000 * 2) * 0.003).astype(np.float32)
    assert speech.trim_to_speech(noise) is None
    assert speech.trim_to_speech(np.zeros(1000, dtype=np.float32)) is None


@pytest.mark.skipif(shutil.which("say") is None, reason="needs macOS `say`")
def test_speech_is_trimmed(tmp_path):
    wav = tmp_path / "hi.wav"
    subprocess.run(["say", "-o", str(wav), "--data-format=LEI16@16000", "Hello there, how are you?"], check=True)
    from faster_whisper import decode_audio

    voice = decode_audio(str(wav), sampling_rate=16000)
    padded = np.concatenate([np.zeros(16000 * 2, np.float32), voice, np.zeros(16000 * 2, np.float32)])
    clip = speech.trim_to_speech(padded)
    assert clip is not None
    assert len(voice) * 0.8 < len(clip) < len(voice) + 16000


def test_whisper_no_speech_filter():
    assert speech._keep(0.1, -0.2)
    assert speech._keep(0.9, -0.2)       # confident text despite a high no-speech probability
    assert not speech._keep(0.9, -1.5)
