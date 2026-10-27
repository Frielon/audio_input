"""Phoneme recognition for sound-level pronunciation feedback.

A wav2vec2 CTC model fine-tuned to emit espeak IPA phones
(``facebook/wav2vec2-lv-60-espeak-cv-ft`` by default) transcribes the learner's
recording into phones, and the same espeak phonemizer produces the *expected*
phones of the target sentence (with lexical stress marks), so both sides share
one phone inventory.

Everything here is optional: when the heavy dependencies (torch, transformers,
phonemizer, espeakng-loader) are missing or ``PARROT_PHONEMES=0`` is set, the app
keeps working without the phoneme panel.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("PARROT_PHONEME_MODEL", "facebook/wav2vec2-lv-60-espeak-cv-ft")
DEVICE = os.environ.get("PARROT_PHONEME_DEVICE", "cpu")  # "mps" works on Apple Silicon
ENABLED = os.environ.get("PARROT_PHONEMES", "1") != "0"
LANGUAGE = "en-us"
FRAME_SEC = 0.02  # wav2vec2 emits one frame every 20 ms
PAD_SEC = 0.25    # silence added around the audio: a sound at the very edge is often missed
_SPECIAL = {"<pad>", "<s>", "</s>", "<unk>", "|"}
_STRESS_MARKS = {"ˈ": 1, "ˌ": 2}  # primary, secondary
_VOWELS = set(
    "ə ɚ ɐ ɪ iː i ɛ æ ɑː ɑ ɔː ɔ ʊ uː u ʌ ɜː ɜ eɪ aɪ oʊ aʊ ɔɪ ɪə eə ʊə e o a y ᵻ eː oː ɑ̃ ɔ̃ ɛ̃ ø œ".split())
# The phones espeak uses for (American) English, plus two variants the model emits for
# English speech.  The model is multilingual: without this restriction an accented
# vowel can come out as, say, a Mandarin toned vowel that matches nothing.
ENGLISH_PHONES = frozenset("""
aɪ aɪɚ aɪə aʊ b d dʒ eɪ f h i iə iː j k l m n n̩ oʊ p s t tʃ uː v w x z æ ð ŋ ɐ ɑː ɑːɹ ɔ ɔɪ ɔː ɔːɹ
ə əl ɚ ɛ ɛɹ ɜː ɡ ɪ ɪɹ ɹ ɾ ʃ ʊ ʊɹ ʌ ʒ ʔ θ ᵻ oːɹ oɪ
""".split())


@dataclass
class Phone:
    text: str
    start: float
    end: float
    prob: float


def is_vowel(p: str) -> bool:
    """Vowels and syllabic sounds (``əl``, ``ɚ``) carry a syllable."""
    return p in _VOWELS or p[:1] in _VOWELS


_lock = threading.Lock()
_infer_lock = threading.Lock()  # espeak-ng is not thread-safe; torch inference is serialised too
_model = None
_processor = None
_device = DEVICE
_backend = None
_allowed = None  # bool mask over the model vocabulary (English phones + blank)
_unavailable: Optional[str] = None
_phonemizer_ok: Optional[bool] = None


def phonemizer_available() -> bool:
    """True when espeak phonemization works (needed for IPA + stress, no torch required)."""
    global _phonemizer_ok
    if _phonemizer_ok is None:
        try:
            import espeakng_loader  # noqa: F401
            import phonemizer  # noqa: F401
            _phonemizer_ok = True
        except Exception as exc:  # pragma: no cover - depends on the environment
            log.warning("IPA/stress annotation unavailable: %s", exc)
            _phonemizer_ok = False
    return _phonemizer_ok


def available() -> bool:
    """True when the optional dependencies import; the model itself loads lazily."""
    global _unavailable
    if not ENABLED:
        _unavailable = "disabled with PARROT_PHONEMES=0"
        return False
    if _unavailable:
        return False
    try:
        import phonemizer  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the environment
        _unavailable = f"missing dependency ({exc})"
        log.warning("Phoneme feedback unavailable: %s", _unavailable)
        return False
    return True


def unavailable_reason() -> Optional[str]:
    return _unavailable


def is_loaded() -> bool:
    return _model is not None and _backend is not None


def _setup_espeak() -> None:
    """Point phonemizer at the espeak-ng library bundled by ``espeakng-loader``."""
    try:
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    except Exception as exc:  # a system-wide espeak-ng may still be found
        log.debug("espeakng_loader not used: %s", exc)


def get_phonemizer():
    global _backend
    with _lock:
        if _backend is None:
            _setup_espeak()
            from phonemizer.backend import EspeakBackend

            quiet = logging.getLogger("phonemizer")
            quiet.setLevel(logging.ERROR)
            _backend = EspeakBackend(LANGUAGE, with_stress=True, language_switch="remove-flags", logger=quiet)
        return _backend


def _parse_token(token: str) -> tuple[str, int]:
    """Split ``ˈɑː`` into the phone and its stress level (0 none, 1 primary, 2 secondary)."""
    stress = 0
    while token and token[0] in _STRESS_MARKS:
        level = _STRESS_MARKS[token[0]]
        stress = 1 if (stress == 1 or level == 1) else level
        token = token[1:]
    for mark in _STRESS_MARKS:
        token = token.replace(mark, "")
    return token, stress


def _phonemize_groups(texts: Sequence[str]) -> list[list[list[tuple[str, int]]]]:
    """For every input string: its espeak words, each a list of ``(phone, stress)``."""
    from phonemizer.separator import Separator

    backend = get_phonemizer()
    sep = Separator(phone=" ", word="| ", syllable="")
    texts = list(texts)
    if not texts:
        return []
    with _infer_lock:
        out = backend.phonemize(texts, separator=sep, strip=True)
        if len(out) != len(texts):  # should not happen, but never mis-assign phones to words
            out = [backend.phonemize([t], separator=sep, strip=True)[0] for t in texts]
    result: list[list[list[tuple[str, int]]]] = []
    for text in out:
        groups: list[list[tuple[str, int]]] = []
        for chunk in text.split("|"):
            tokens: list[tuple[str, int]] = []
            pending = 0
            for raw in chunk.split():
                phone, stress = _parse_token(raw)
                if not phone:  # a lone stress mark applies to the next phone
                    pending = stress or pending
                    continue
                if pending and not stress:
                    stress = pending
                pending = 0
                tokens.append((phone, stress))
            if tokens:
                groups.append(tokens)
        result.append(groups)
    return result


def phonemize_words(words: Sequence[str]) -> list[list[str]]:
    """Citation-form phones of each word taken on its own."""
    return [[p for group in groups for p, _ in group] for groups in _phonemize_groups(words)]


def _phonemize_sentence_tokens(words: Sequence[str]) -> list[list[tuple[str, int]]]:
    """Per-word ``(phone, stress)`` tokens, phonemized *in context*.

    espeak reduces function words in connected speech (``to`` becomes ``t ə``
    rather than the citation form ``t uː``), which is what a fluent speaker says,
    so the whole sentence is phonemized at once and its espeak words are handed
    back to the display words.  A display word may expand to several espeak words
    (``25`` -> ``twenty five``); the per-word citation forms tell how many.
    """
    words = list(words)
    if not words:
        return []
    per_word = _phonemize_groups(words)
    counts = [len(groups) for groups in per_word]
    sentence_groups = _phonemize_groups([" ".join(words)])[0]
    if sum(counts) != len(sentence_groups):  # espeak split the sentence differently: fall back
        return [[t for group in groups for t in group] for groups in per_word]
    out: list[list[tuple[str, int]]] = []
    pos = 0
    for count in counts:
        out.append([t for group in sentence_groups[pos:pos + count] for t in group])
        pos += count
    return out


def phonemize_sentence(words: Sequence[str]) -> list[list[str]]:
    """Expected phones per word (connected speech, no stress marks)."""
    return [[p for p, _ in tokens] for tokens in _phonemize_sentence_tokens(words)]


def phonemize_sentence_stress(words: Sequence[str]) -> tuple[list[list[str]], list[list[int]]]:
    """Expected phones per word plus, aligned with them, the stress level of each phone."""
    tokens = _phonemize_sentence_tokens(words)
    return [[p for p, _ in t] for t in tokens], [[s for _, s in t] for t in tokens]


def get_model():
    global _model, _processor, _device, _allowed
    with _lock:
        if _model is None:
            import torch
            from transformers import AutoProcessor, Wav2Vec2ForCTC

            _setup_espeak()  # the processor's tokenizer loads espeak too
            log.info("Loading phoneme model %s (%s)…", MODEL_NAME, _device)
            processor = AutoProcessor.from_pretrained(MODEL_NAME)
            model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).eval()
            if _device != "cpu":
                try:
                    model = model.to(_device)
                    with torch.inference_mode():  # make sure the device really works
                        model(torch.zeros(1, 16000, device=_device))
                except Exception as exc:
                    log.warning("Device %s unusable for the phoneme model (%s); using CPU", _device, exc)
                    _device = "cpu"
                    model = model.to("cpu")
            vocab = processor.tokenizer.get_vocab()
            allowed = torch.zeros(len(vocab), dtype=torch.bool)
            for token, i in vocab.items():
                if token in ENGLISH_PHONES or i == processor.tokenizer.pad_token_id:
                    allowed[i] = True
            _allowed = allowed
            _processor, _model = processor, model
            log.info("Phoneme model ready.")
        return _processor, _model


def warm_up_async() -> None:
    if not available():
        return

    def run() -> None:
        try:
            get_phonemizer()
            get_model()
            recognize(np.zeros(16000, dtype=np.float32))  # first inference is slow; do it now
        except Exception:  # pragma: no cover
            log.exception("Phoneme model warm-up failed")

    threading.Thread(target=run, name="phoneme-warmup", daemon=True).start()


def recognize(audio: np.ndarray, sample_rate: int = 16000) -> list[Phone]:
    """Greedy CTC decoding of the recording into timed phones with confidences."""
    import torch

    processor, model = get_model()
    tokenizer = processor.tokenizer
    blank = tokenizer.pad_token_id
    audio = np.asarray(audio, dtype=np.float32)
    duration = len(audio) / sample_rate
    pad = np.zeros(int(PAD_SEC * sample_rate), dtype=np.float32)
    inputs = processor(np.concatenate([pad, audio, pad]), sampling_rate=sample_rate, return_tensors="pt")
    with _infer_lock, torch.inference_mode():
        logits = model(inputs.input_values.to(_device)).logits[0].float().cpu()
    if _allowed is not None and len(_allowed) == logits.shape[-1]:
        logits = logits.masked_fill(~_allowed, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    ids = torch.argmax(logits, dim=-1).tolist()

    phones: list[Phone] = []
    prev = blank
    for f, i in enumerate(ids):
        if i == blank:
            prev = blank
            continue
        p = float(probs[f, i])
        start, end = f * FRAME_SEC - PAD_SEC, (f + 1) * FRAME_SEC - PAD_SEC
        if i == prev and phones:  # same phone continues over several frames
            phones[-1].end = end
            phones[-1].prob = max(phones[-1].prob, p)
        else:
            phones.append(Phone(tokenizer.convert_ids_to_tokens(i), start, end, p))
        prev = i
    out = []
    for ph in phones:
        if ph.text in _SPECIAL:
            continue
        ph.start = round(min(max(ph.start, 0.0), duration), 3)
        ph.end = round(min(max(ph.end, ph.start), duration), 3)
        out.append(ph)
    return out
