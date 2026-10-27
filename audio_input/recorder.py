"""Microphone capture for push-to-talk: 16 kHz mono float32, opened only while the key is held.

The stream is opened when the key goes down (about 0.1 s for a USB or built-in
microphone; Bluetooth headsets need longer to switch to their microphone
profile, so start speaking once the overlay says "Listening").  Keeping the
microphone closed in between means the orange microphone indicator only shows
while you dictate.

``AUDIO_INPUT_FAKE_MIC=/path/to/file.wav`` replaces the microphone with that
recording (for tests).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

SR = 16000


def input_devices() -> list[dict]:
    import sounddevice as sd

    default = sd.default.device[0]
    return [{"index": i, "name": d["name"], "channels": d["max_input_channels"],
             "samplerate": d["default_samplerate"], "default": i == default}
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


def resolve_device(spec: Optional[str]) -> Optional[int]:
    """Index of the input device named (or partly named) ``spec``; None for the system default."""
    if spec is None or str(spec).strip() == "":
        return None
    spec = str(spec).strip()
    devices = input_devices()
    if spec.isdigit():
        if any(d["index"] == int(spec) for d in devices):
            return int(spec)
        raise ValueError(f"no input device with index {spec}")
    for d in devices:
        if d["name"].lower() == spec.lower():
            return d["index"]
    for d in devices:
        if spec.lower() in d["name"].lower():
            return d["index"]
    names = ", ".join(d["name"] for d in devices)
    raise ValueError(f"no input device matching {spec!r} (available: {names})")


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """Windowed-sinc low-pass, then linear interpolation (good enough for speech recognition)."""
    if sr_in == sr_out or not len(x):
        return x.astype(np.float32)
    if sr_out < sr_in:
        cutoff = 0.45 * sr_out / sr_in  # cycles per input sample
        n = np.arange(-32, 33)
        h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(len(n))
        x = np.convolve(x, h / h.sum(), mode="same")
    t_out = np.arange(int(len(x) * sr_out / sr_in)) * (sr_in / sr_out)
    return np.interp(t_out, np.arange(len(x)), x).astype(np.float32)


class Recorder:
    def __init__(self, device: Optional[str] = None, max_seconds: float = 120.0):
        self.device = resolve_device(device)
        self.max_seconds = max_seconds
        self.fake = os.environ.get("AUDIO_INPUT_FAKE_MIC")
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._sr = SR
        self._samples = 0
        self._t0 = 0.0
        self._level = 0.0
        self._live_at: Optional[float] = None
        self.recording = False
        self.silent = False  # the last recording was digital silence (muted, lid closed, no permission)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0 if self.recording else 0.0

    @property
    def live(self) -> bool:
        """Sound is arriving (a Bluetooth headset sends zeros while it switches to its microphone)."""
        return bool(self.fake) or self._live_at is not None

    @property
    def level(self) -> float:
        """Loudness of the last block, 0 (below -60 dBFS) .. 1 (-10 dBFS)."""
        if self.fake and self.recording:
            return 0.35 + 0.3 * abs(np.sin(self.elapsed * 7))
        return self._level

    def device_name(self) -> str:
        import sounddevice as sd

        if self.fake:
            return f"file {os.path.basename(self.fake)}"
        idx = self.device if self.device is not None else sd.default.device[0]
        return sd.query_devices(idx)["name"]

    def _callback(self, indata, frames, time_info, status) -> None:
        block = indata[:, 0].copy()
        if self._live_at is None and np.any(block):
            self._live_at = time.monotonic()
        rms = float(np.sqrt(np.mean(block ** 2))) if len(block) else 0.0
        self._level = float(np.clip((20 * np.log10(rms + 1e-9) + 60) / 50, 0.0, 1.0))
        with self._lock:
            if self._samples < self.max_seconds * self._sr:
                self._chunks.append(block)
                self._samples += len(block)

    def start(self) -> None:
        if self.recording:
            return
        with self._lock:
            self._chunks, self._samples, self._level = [], 0, 0.0
        self._live_at = None
        self._t0 = time.monotonic()
        if self.fake:
            self.recording = True
            return
        import sounddevice as sd

        try:
            self._sr = SR
            self._stream = sd.InputStream(device=self.device, samplerate=SR, channels=1, dtype="float32",
                                          callback=self._callback)
        except sd.PortAudioError:  # the device cannot do 16 kHz itself: record at its own rate
            self._sr = int(sd.query_devices(self.device if self.device is not None else sd.default.device[0])
                           ["default_samplerate"])
            self._stream = sd.InputStream(device=self.device, samplerate=self._sr, channels=1, dtype="float32",
                                          callback=self._callback)
        self._stream.start()
        self.recording = True

    def _close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # pragma: no cover - device unplugged while recording
                log.exception("closing the microphone failed")
        self.recording = False

    def stop(self) -> np.ndarray:
        """Stop and return the recording (16 kHz mono float32)."""
        if self.fake:
            held = self.elapsed
            self.recording = False
            from faster_whisper import decode_audio

            audio = decode_audio(self.fake, sampling_rate=SR)
            log.info("fake microphone: %.1fs held, returning %s", held, self.fake)
            self.silent = False
            return audio.astype(np.float32)
        self._close()
        with self._lock:
            audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
            self._chunks = []
        audio = resample(audio, self._sr, SR)
        self.silent = len(audio) > SR // 4 and float(np.abs(audio).max(initial=0.0)) < 1e-5
        return audio

    def cancel(self) -> None:
        self._close()
        with self._lock:
            self._chunks = []
