"""Speech to text with word timings.

Two Whisper backends:

* ``mlx``: mlx-whisper on the Apple GPU, ``large-v3-turbo`` by default (about
  0.8 s per utterance on an M4 Pro, and much more robust to accents than the
  small models);
* ``faster``: faster-whisper on the CPU, ``small.en`` by default (the model
  parrot_shadowing uses, about 1 s per utterance).

Silence never reaches Whisper: a voice-activity detector (Silero, shipped with
faster-whisper) finds the speech, the recording is trimmed to it, and a
recording without speech is dropped (Whisper "hears" words such as "Thank you."
in silence or noise).  Segments Whisper itself rates as no speech are dropped
too.
"""
from __future__ import annotations

import logging
import os
import platform
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

SR = 16000
EDGE_PAD = 0.25          # seconds of audio kept before the first and after the last speech
MIN_SPEECH = 0.2         # seconds of detected speech below which a recording is ignored
NO_SPEECH_PROB = 0.6     # Whisper's own no-speech test, as in openai-whisper
LOW_LOGPROB = -1.0


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float


@dataclass
class Transcript:
    text: str
    words: list[Word]
    audio: np.ndarray    # the trimmed recording; word times refer to it
    backend: str
    elapsed: float


def speech_regions(audio: np.ndarray) -> list[tuple[int, int]]:
    """Sample ranges that contain speech."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=500, speech_pad_ms=100)
    return [(r["start"], r["end"]) for r in get_speech_timestamps(np.asarray(audio, dtype=np.float32), opts)]


def trim_to_speech(audio: np.ndarray) -> Optional[np.ndarray]:
    """The recording cut to its speech (with a little margin), or None when there is no speech."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < SR * MIN_SPEECH or float(np.abs(audio).max(initial=0.0)) < 1e-4:
        return None
    regions = speech_regions(audio)
    if sum(e - s for s, e in regions) < SR * MIN_SPEECH:
        return None
    pad = int(EDGE_PAD * SR)
    return audio[max(0, regions[0][0] - pad): min(len(audio), regions[-1][1] + pad)]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _keep(seg_no_speech: float, seg_logprob: float) -> bool:
    return not (seg_no_speech > NO_SPEECH_PROB and seg_logprob < LOW_LOGPROB)


def resolve_backend(name: str) -> str:
    if name != "auto":
        return name
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_whisper  # noqa: F401
            return "mlx"
        except Exception:
            pass
    return "faster"


class Transcriber:
    """Lazily loaded Whisper.  Call :meth:`load` and :meth:`transcribe` from one thread (MLX streams are per thread)."""

    def __init__(self, backend: str = "auto", mlx_model: str = "mlx-community/whisper-large-v3-turbo",
                 whisper_model: str = "small.en", beam_size: int = 5, vocabulary: Optional[list[str]] = None,
                 prompt: Optional[str] = None):
        self.backend = resolve_backend(backend)
        self.mlx_model = mlx_model
        self.whisper_model = whisper_model
        self.beam_size = beam_size
        glossary = ("Glossary: " + ", ".join(vocabulary) + ".") if vocabulary else ""
        self.prompt = " ".join(p for p in (prompt or "", glossary) if p) or None
        self._faster = None
        self._lock = threading.Lock()
        self.loaded = False

    @property
    def model_name(self) -> str:
        return self.mlx_model if self.backend == "mlx" else self.whisper_model

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            t0 = time.time()
            if self.backend == "faster":
                from faster_whisper import WhisperModel

                self._faster = WhisperModel(self.whisper_model, device="cpu", compute_type="int8",
                                            cpu_threads=min(8, os.cpu_count() or 4))
            else:
                # one short pass loads the weights and compiles the GPU kernels
                rng = np.random.default_rng(0)
                self._run_mlx((rng.standard_normal(SR // 2) * 1e-3).astype(np.float32))
            self.loaded = True
            log.info("Whisper ready (%s, %s) in %.1fs", self.backend, self.model_name, time.time() - t0)

    def transcribe(self, audio: np.ndarray, trim: bool = True) -> Optional[Transcript]:
        """Words and text of a 16 kHz mono recording; None when no speech was found."""
        self.load()
        t0 = time.time()
        clip = trim_to_speech(audio) if trim else np.asarray(audio, dtype=np.float32)
        if clip is None:
            return None
        with self._lock:
            text, words = self._run_mlx(clip) if self.backend == "mlx" else self._run_faster(clip)
        if not text or not words:
            return None
        return Transcript(text=text, words=words, audio=clip, backend=self.backend, elapsed=time.time() - t0)

    # ------------------------------------------------------------- backends
    def _run_faster(self, audio: np.ndarray) -> tuple[str, list[Word]]:
        segments, _ = self._faster.transcribe(
            audio, language="en", task="transcribe", beam_size=self.beam_size, word_timestamps=True,
            condition_on_previous_text=False, initial_prompt=self.prompt, vad_filter=False)
        texts: list[str] = []
        words: list[Word] = []
        for seg in segments:
            if not _keep(seg.no_speech_prob, seg.avg_logprob):
                continue
            texts.append(seg.text)
            words += [Word(w.word.strip(), float(w.start), float(w.end), float(w.probability))
                      for w in seg.words or [] if w.word.strip()]
        return _clean(" ".join(texts)), words

    def _run_mlx(self, audio: np.ndarray) -> tuple[str, list[Word]]:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self.mlx_model, language="en", task="transcribe", word_timestamps=True,
            condition_on_previous_text=False, initial_prompt=self.prompt, temperature=(0.0, 0.2, 0.4, 0.6),
            verbose=None)
        texts: list[str] = []
        words: list[Word] = []
        for seg in result.get("segments", []):
            if not _keep(seg.get("no_speech_prob", 0.0), seg.get("avg_logprob", 0.0)):
                continue
            texts.append(seg.get("text", ""))
            words += [Word(w["word"].strip(), float(w["start"]), float(w["end"]), float(w.get("probability", 1.0)))
                      for w in seg.get("words", []) if w["word"].strip()]
        return _clean(" ".join(texts)), words
