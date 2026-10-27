"""The sound analysis ships with audio_input: no Parrot Shadowing checkout is needed (no models are loaded)."""
from audio_input.parrot_backend import english, scoring


def test_bundled_backend_imports():
    from audio_input.parrot_backend import analysis, phoneme_engine  # noqa: F401  (torch loads only with the model)

    assert scoring.display_words("Yesterday I can't go, really.") == ["Yesterday", "I", "can't", "go,", "really."]
    assert english.syllabify(["k", "æ", "l", "k", "j", "ʊ", "l", "eɪ", "ɾ", "ɚ"]) == [(0, 3), (3, 6), (6, 8), (8, 10)]
