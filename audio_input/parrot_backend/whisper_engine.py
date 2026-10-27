"""Lazy, shared faster-whisper model used for both video transcription and scoring."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("PARROT_WHISPER_MODEL", "small.en")
DEVICE = os.environ.get("PARROT_WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("PARROT_WHISPER_COMPUTE", "int8")

_model = None
_model_lock = threading.Lock()


@dataclass
class RecognizedWord:
    text: str
    start: float
    end: float
    probability: float


def is_loaded() -> bool:
    return _model is not None


def get_model():
    """Load the model on first use (downloads weights the very first time)."""
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            log.info("Loading Whisper model %s (%s/%s)…", MODEL_NAME, DEVICE, COMPUTE_TYPE)
            _model = WhisperModel(
                MODEL_NAME,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=min(8, os.cpu_count() or 4),
                num_workers=2,  # a long video transcription must not block scoring
            )
            log.info("Whisper model ready.")
        return _model


def warm_up_async() -> None:
    threading.Thread(target=get_model, name="whisper-warmup", daemon=True).start()


def transcribe(audio, *, progress_cb: Optional[Callable[[float], None]] = None,
               beam_size: int = 5) -> tuple[list[RecognizedWord], dict]:
    """Transcribe a path / file-like / float32 array and return timed words."""
    model = get_model()
    segments, info = model.transcribe(
        audio,
        language="en",
        task="transcribe",
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
    )
    words: list[RecognizedWord] = []
    texts: list[str] = []
    for seg in segments:
        texts.append(seg.text.strip())
        for w in seg.words or []:
            text = w.word.strip()
            if text:
                words.append(RecognizedWord(text, float(w.start), float(w.end), float(w.probability)))
        if progress_cb and info.duration:
            progress_cb(min(1.0, seg.end / info.duration))
    return words, {"text": " ".join(t for t in texts if t), "duration": float(info.duration or 0.0),
                   "model": MODEL_NAME}
