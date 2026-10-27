"""Signal measurements shared by the stress, rhythm and pause analyses.

Computed once per recording:

* a 5 ms loudness envelope: total level, the level of the 1-4 kHz band (where
  stressed vowels gain the most energy) and the spectral tilt between that band
  and the band below 1 kHz;
* a pitch (F0) track from the autocorrelation of 40 ms frames;
* the pauses: stretches of silence between the first and the last sound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

SR = 16000
ENV_FRAME, ENV_HOP = 0.025, 0.005
F0_FRAME, F0_HOP = 0.04, 0.01
F0_MIN, F0_MAX = 70.0, 400.0
VOICING = 0.45      # minimum normalised autocorrelation peak of a voiced frame
MIN_PAUSE = 0.18    # shorter silences are stop closures or breaths inside a phrase
CONTOUR_STEP = 0.02


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if len(x) < size:
        x = np.pad(x, (0, size - len(x)))
    return np.lib.stride_tricks.sliding_window_view(x, size)[::hop]


def _median3(v: np.ndarray) -> np.ndarray:
    """3-point median over the voiced frames only (removes single-frame octave jumps)."""
    out = v.copy()
    for i in range(1, len(v) - 1):
        trio = v[i - 1:i + 2]
        if not np.isnan(trio).any():
            out[i] = float(np.median(trio))
    return out


@dataclass
class Acoustics:
    sr: int
    duration: float
    t: np.ndarray       # envelope frame centres (s)
    db: np.ndarray      # level (dB)
    hi_db: np.ndarray   # 1-4 kHz band level (dB)
    tilt: np.ndarray    # 1-4 kHz minus 60-1000 Hz level (dB)
    f0_t: np.ndarray    # pitch frame centres (s)
    f0: np.ndarray      # Hz, NaN where unvoiced

    # ------------------------------------------------------------- windows
    def env_mask(self, start: float, end: float) -> np.ndarray:
        return (self.t >= start) & (self.t < end)

    def f0_in(self, start: float, end: float) -> np.ndarray:
        m = (self.f0_t >= start) & (self.f0_t < end)
        v = self.f0[m]
        return v[~np.isnan(v)]

    def peak_db(self, start: float, end: float) -> Optional[float]:
        m = self.env_mask(start, end)
        return float(self.db[m].max()) if m.any() else None

    # -------------------------------------------------------------- speech
    def silence_threshold(self) -> float:
        level = float(np.percentile(self.db, 95))
        floor = float(np.percentile(self.db, 10))
        return min(level - 12.0, max(level - 30.0, floor + 6.0))

    def speech_bounds(self) -> Optional[tuple[float, float]]:
        idx = np.where(self.db > self.silence_threshold())[0]
        if not len(idx):
            return None
        return float(self.t[idx[0]] - ENV_FRAME / 2), float(self.t[idx[-1]] + ENV_FRAME / 2)

    def pauses(self, min_pause: float = MIN_PAUSE) -> list[tuple[float, float]]:
        """Silent stretches of at least ``min_pause`` between the first and the last sound."""
        speech = self.db > self.silence_threshold()
        idx = np.where(speech)[0]
        if not len(idx):
            return []
        out: list[tuple[float, float]] = []
        k, last = idx[0], idx[-1]
        while k <= last:
            if speech[k]:
                k += 1
                continue
            j = k
            while j <= last and not speech[j]:
                j += 1
            start, end = self.t[k] - ENV_HOP / 2, self.t[j - 1] + ENV_HOP / 2
            if end - start >= min_pause:
                out.append((round(float(start), 3), round(float(end), 3)))
            k = j
        return out

    # --------------------------------------------------------------- pitch
    def f0_median(self) -> Optional[float]:
        v = self.f0[~np.isnan(self.f0)]
        return float(np.median(v)) if len(v) >= 3 else None

    def pitch_range(self) -> Optional[float]:
        """Spread of the voice pitch in semitones (10th to 90th percentile)."""
        v = self.f0[~np.isnan(self.f0)]
        if len(v) < 10:
            return None
        lo, hi = np.percentile(v, [10, 90])
        return round(12 * math.log2(hi / lo), 1)

    def contour(self, start: float = 0.0, end: Optional[float] = None, step: float = CONTOUR_STEP) -> dict:
        """Pitch (semitones around the median) and loudness (0..1) sampled every ``step`` s for drawing."""
        end = self.duration if end is None else end
        ref = self.f0_median()
        times = np.arange(start, end, step)
        pitch: list[Optional[float]] = []
        level: list[float] = []
        top = float(np.percentile(self.db, 98)) if len(self.db) else 0.0
        for t in times:
            v = self.f0_in(t - step / 2, t + step / 2)
            pitch.append(round(12 * math.log2(float(np.median(v)) / ref), 1) if len(v) and ref else None)
            m = self.env_mask(t - step / 2, t + step / 2)
            d = float(self.db[m].max()) if m.any() else -120.0
            level.append(round(max(0.0, min(1.0, (d - (top - 40)) / 40)), 2))
        return {"start": round(start, 3), "step": step, "pitch": pitch, "level": level}


def analyze(audio: np.ndarray, sr: int = SR) -> Acoustics:
    x = np.asarray(audio, dtype=np.float64)
    duration = len(x) / sr

    # loudness envelope and band levels
    size, hop = int(sr * ENV_FRAME), int(sr * ENV_HOP)
    fr = _frames(x, size, hop)
    spec = np.abs(np.fft.rfft(fr * np.hanning(size), axis=1)) ** 2
    freqs = np.fft.rfftfreq(size, 1 / sr)
    total = spec.sum(axis=1) / size + 1e-12
    hi = spec[:, (freqs >= 1000) & (freqs <= 4000)].sum(axis=1) / size + 1e-12
    lo = spec[:, (freqs >= 60) & (freqs < 1000)].sum(axis=1) / size + 1e-12
    t = np.arange(len(fr)) * ENV_HOP + ENV_FRAME / 2

    # pitch: autocorrelation of every frame through the FFT, peak picked with parabolic refinement
    size, hop = int(sr * F0_FRAME), int(sr * F0_HOP)
    fr = _frames(x, size, hop).astype(np.float64)
    fr = fr - fr.mean(axis=1, keepdims=True)
    ac = np.fft.irfft(np.abs(np.fft.rfft(fr, n=2 * size, axis=1)) ** 2, n=2 * size, axis=1)[:, :size]
    energy = ac[:, 0]
    lag_min, lag_max = int(sr / F0_MAX), min(int(sr / F0_MIN), size - 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        acn = ac / np.where(energy > 0, energy, np.inf)[:, None]
    seg = acn[:, lag_min:lag_max + 1]
    k = seg.argmax(axis=1)
    rows = np.arange(len(seg))
    peak = seg[rows, k]
    left = seg[rows, np.clip(k - 1, 0, seg.shape[1] - 1)]
    right = seg[rows, np.clip(k + 1, 0, seg.shape[1] - 1)]
    denom = left - 2 * peak + right
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(np.abs(denom) > 1e-9, 0.5 * (left - right) / denom, 0.0)
    f0 = sr / (lag_min + k + np.clip(delta, -0.5, 0.5))
    frame_db = 10 * np.log10(energy / size + 1e-12)
    loud = float(np.percentile(frame_db, 95)) if len(frame_db) else 0.0
    voiced = (peak >= VOICING) & (frame_db > loud - 30.0) & (np.abs(fr).max(axis=1) > 1e-4)
    f0 = _median3(np.where(voiced, f0, np.nan))
    return Acoustics(sr=sr, duration=duration, t=t, db=10 * np.log10(total), hi_db=10 * np.log10(hi),
                     tilt=10 * np.log10(hi / lo), f0_t=np.arange(len(fr)) * F0_HOP + F0_FRAME / 2, f0=f0)
