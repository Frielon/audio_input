"""Recorder helpers and speech trimming (Silero VAD; no Whisper model needed)."""
import shutil
import subprocess

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
