# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/ai/__init__.py -- The AI Wing (v0.4.0).

Hypothesis: AI models are guests, not residents -- gigabytes of
  weights, torch/transformers dependencies, a GPU for sane speeds.
  So the wing is adapter-shaped: every model sits behind a probe
  that reports exactly what is present and fails LOUDLY (exit 3)
  when it is not, and every cloud feature takes its credential at
  runtime, never from disk, never silently.
Method:     `gemini` (cloud guest: analysis copilot, NL control,
  composition ideas -- key via env, never stored); `analyze`
  (deterministic measurements: peak/RMS/crest/centroid/width/
  clipping -- the honest numbers the AI reasons about); `master`
  (rule-based auto-chain offline + Gemini advice online);
  `command` (natural-language control: deterministic local parser
  first, Gemini parser on request); `compose` (deterministic
  theory-correct composition helpers + Gemini ideas); `generate`
  / `transcribe` (local-model adapters per the roadmap: MusicGen /
  Whisper tradition -- loud skip until torch + weights exist).
Result:     `from resonance.ai import analyze_track, parse_command`
  just works, offline, with no key and no weights. The CLI entry
  point main() lives in cli.py (`resonance-ai`).
"""
from resonance.ai.adapters import BackendProbe, ModelNotAvailable
from resonance.ai.analyze import AnalysisResult, analyze_track, format_report
from resonance.ai.command import (
    Intent,
    execute_intent,
    parse_command,
)
from resonance.ai.compose import (
    chord_progression,
    drum_pattern,
    melody_from_chords,
)
from resonance.ai.gemini import GeminiClient, GeminiError, GeminiKeyMissing

__all__ = [
    "BackendProbe",
    "ModelNotAvailable",
    "AnalysisResult",
    "analyze_track",
    "format_report",
    "Intent",
    "execute_intent",
    "parse_command",
    "chord_progression",
    "drum_pattern",
    "melody_from_chords",
    "GeminiClient",
    "GeminiError",
    "GeminiKeyMissing",
]
