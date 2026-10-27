"""Parrot Shadowing's sound analysis, bundled so that audio_input needs nothing else.

These modules are unchanged copies from the ``backend`` package of Parrot
Shadowing (a shadowing-practice app by the same author): the phoneme recognizer
(wav2vec2 -> espeak IPA), the alignment of heard and expected sounds, stress,
rhythm and the teacher's checklist (``coach``).  Only this file is audio_input's
own.  ``scripts/sync_parrot.sh`` copies them again from a Parrot Shadowing
checkout; :mod:`audio_input.parrot` is the bridge the app uses.
"""
