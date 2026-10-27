"""Every utterance and its feedback, one JSON object per line in ``data/history.jsonl``."""
from __future__ import annotations

import json
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

log = logging.getLogger(__name__)


class History:
    def __init__(self, data_dir: Path, keep_audio: bool = False):
        self.path = Path(data_dir) / "history.jsonl"
        self.audio_dir = Path(data_dir) / "recordings"
        self.keep_audio = keep_audio
        self._lock = threading.Lock()

    def add(self, payload: dict, audio: Optional[np.ndarray] = None, app: Optional[str] = None,
            seconds: Optional[float] = None) -> dict:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        record = {
            "time": stamp,
            "app": app,
            "seconds": None if seconds is None else round(seconds, 2),
            "text": payload.get("text"),
            "corrected": payload.get("corrected"),
            "grammar": [{"kind": e["kind"], "label": e["label"], "original": e["original"],
                         "correction": e["correction"], "severity": e["severity"]}
                        for e in payload.get("grammar_edits", [])],
            "sounds": [{"cat": i["cat"], "word": i.get("word"), "detail": i.get("text")}
                       for i in payload.get("issues", []) if i["kind"] == "sound"],
            "words": [{"w": w["w"], "acc": w.get("acc"), "heard": w.get("heard")}
                      for w in payload.get("words", []) if w.get("sound") in ("warn", "bad")],
            "summary": payload.get("summary"),
        }
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.keep_audio and audio is not None and len(audio):
                    self.audio_dir.mkdir(parents=True, exist_ok=True)
                    wav = self.audio_dir / f"{stamp.replace(':', '-')}.wav"
                    write_wav(wav, audio)
                    record["audio"] = str(wav)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                log.exception("could not write the history")
        return record

    def records(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def write_wav(path: Path, audio: np.ndarray, sr: int = 16000) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
