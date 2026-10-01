# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/core/__init__.py -- core DSP primitives and session configuration.

Public surface:
  buffers : mono/stereo validation, mix, normalize, fades
  io      : WAV read/write via stdlib wave (float32 <-> int16 PCM)
  notes   : note name <-> MIDI <-> frequency (A4 = 440 Hz), Solfeggio data
  config  : SessionConfig dataclass
  edit    : waveform-editing primitives (v0.1.0 deliverable; the full
            multitrack timeline editor UI is a v0.3.0 roadmap item)
"""
from resonance.core import buffers, config, edit, io, notes

__all__ = ["buffers", "config", "edit", "io", "notes"]
