# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/ai/transcribe.py -- speech/music transcription adapter.

HYPOTHESIS
    Transcription feeds the karaoke display and the ABC
    transcription aids -- but a transcript is a CLAIM about what
    was said or sung, so a fake one is worse than none. This
    adapter (Whisper tradition) therefore has exactly two
    outputs: a real transcript when torch + weights are present,
    or ModelNotAvailable naming the missing pieces. There is no
    heuristic fallback that pretends to hear words.

METHOD
    WhisperAdapter(LocalModelAdapter): require() first; _run()
    is the real model call, wired when weights land. Segments
    come back as (start_s, end_s, text) triples -- the shape the
    karaoke display consumes.

RESULT
    transcribe() is honest about being unavailable today; the
    segment contract is fixed so the karaoke/ABC consumers can
    be built against it without guessing.
"""
from resonance.ai.adapters import LocalModelAdapter, ModelNotAvailable

WEIGHTS_ENV = "RESONANCE_WHISPER_DIR"
MODEL_ID = "openai/whisper-large-v3"  # adapter target, not a claim


class WhisperAdapter(LocalModelAdapter):
    """Speech-to-text / lyrics transcription (Whisper tradition)."""

    def __init__(self, model_id=MODEL_ID):
        super().__init__("whisper", WEIGHTS_ENV,
                         extra_modules=("transformers",))
        self.model_id = model_id

    def _run(self, audio, sr, language=None):
        """The real model call. Implemented when weights land.

        Unreachable today without require() passing -- the seam
        is exercised, never stubbed around.
        """
        raise ModelNotAvailable(
            f"whisper: model call not wired (target {self.model_id}); "
            f"this path only executes after torch + weights are present.")

    def transcribe(self, audio, sr, language=None):
        """Transcribe audio -> list of (start_s, end_s, text).

        Loud skip until the backend exists. No heuristic word-
        guessing: a transcript is a claim, and claims need the
        model.
        """
        self.require()
        return self._run(audio, sr, language=language)
