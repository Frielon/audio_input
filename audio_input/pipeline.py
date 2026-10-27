"""Recording -> transcript -> feedback (sounds and grammar run side by side)."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

import numpy as np

from . import feedback, voice
from .config import DATA_DIR, Settings
from .grammar import GrammarChecker, GrammarResult
from .inserter import strip_fillers
from .parrot import Pronunciation
from .speech import Transcriber, Transcript

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.transcriber = Transcriber(settings.whisper_backend, settings.mlx_model, settings.whisper_model,
                                       settings.beam_size, settings.vocabulary, settings.whisper_prompt)
        self.pron = Pronunciation() if settings.phonemes else None
        self.grammar = GrammarChecker(settings.grammar_model, settings.grammar_device) if settings.grammar else None
        self.voice = voice.Voice(DATA_DIR / "tts", settings.tts_voice) \
            if settings.model_voice and voice.available() else None
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="feedback")
        # Models are loaded one at a time: transformers' lazy imports are not thread-safe.
        self._load_lock = threading.Lock()

    def warm_up_feedback(self) -> list[Future]:
        """Load the sound and grammar models in the background (Whisper is loaded by its own thread)."""
        futures = []
        if self.pron is not None:
            futures.append(self._pool.submit(self._load, "sounds", self.pron.load))
        if self.grammar is not None:
            futures.append(self._pool.submit(self._load, "grammar", self.grammar.load))
        return futures

    def _load(self, name: str, fn) -> Optional[str]:
        try:
            with self._load_lock:
                fn()
            return None
        except Exception as exc:
            log.exception("Loading the %s model failed", name)
            return f"{name}: {exc}"

    def transcribe(self, audio: np.ndarray) -> Optional[Transcript]:
        return self.transcriber.transcribe(audio)

    def _sounds(self, tr: Transcript) -> Optional[dict]:
        if self.pron is None:
            return None
        error = self._load("sounds", self.pron.load) if not self.pron.loaded else None
        if error or not self.pron.available():
            return {"error": self.pron.error or error}
        try:
            return self.pron.analyze(tr.text, tr.audio, tr.words)
        except Exception as exc:  # the transcript and grammar are still useful
            log.exception("Sound analysis failed")
            return {"error": str(exc)}

    def _grammar(self, text: str) -> Optional[GrammarResult]:
        if self.grammar is None:
            return None
        if not self.grammar.loaded and self._load("grammar", self.grammar.load):
            return None
        try:
            return self.grammar.check(text)
        except Exception:
            log.exception("Grammar check failed")
            return None

    def analyze(self, tr: Transcript) -> tuple[dict, Optional[GrammarResult]]:
        """The overlay payload for a transcript, plus the grammar result (for inserting the corrected text)."""
        t0 = time.time()
        sounds = self._pool.submit(self._sounds, tr)
        grammar = self._pool.submit(self._grammar, tr.text)
        pron, gram = sounds.result(), grammar.result()
        payload = feedback.build(tr.text, pron, gram, self.settings.focus)
        payload["timing"] = {"transcribe": round(tr.elapsed, 2), "feedback": round(time.time() - t0, 2)}
        payload["models"] = {"speech": f"{tr.backend}:{self.transcriber.model_name}"}
        return payload, gram

    @staticmethod
    def model_text(payload: dict) -> str:
        """What the model voice reads: the corrected sentence, without filler words."""
        return strip_fillers(payload.get("corrected") or payload["text"])

    def model_reference(self, payload: dict) -> tuple[Optional[np.ndarray], dict]:
        """The model voice reading the sentence, and the payload with its word times and stress added.

        The reading is analysed like parrot's reference speaker, so every word
        you share with it gets the model's syllable prominence next to yours.
        """
        if self.voice is None:
            return None, payload
        text = self.model_text(payload)
        audio = self.voice.speak(text)
        if self.pron is not None and self.pron.loaded and len(audio):
            try:
                ref = self.pron.analyze_reference(text, audio)
                payload = feedback.attach_model(payload, self.pron.display_words(text), ref)
            except Exception:  # the recording still plays
                log.exception("Analysing the model voice failed")
        return audio, payload

    def process(self, audio: np.ndarray) -> Optional[dict]:
        """Everything at once (command-line use)."""
        tr = self.transcribe(audio)
        if tr is None:
            return None
        payload, _ = self.analyze(tr)
        return payload

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
