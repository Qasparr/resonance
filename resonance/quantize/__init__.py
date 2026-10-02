# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/quantize/__init__.py -- the timing-quantization subpackage.

Hypothesis: timing correction is moving sound in time, and there are
  exactly two honest ways to move it -- SLICE for percussive
  material (cut at transients, shift slices, crossfade the edges)
  and WARP for legato/vocal material (smooth time-map through
  onset anchors, resample). The honest rule binding them: every hit
  is reported with its detected time, grid target, applied shift,
  confidence, and which mover handled it -- strength 0% returns the
  input bit-identical, and the report says so.
Method:     re-export onset detection (onsets) and the grid engine
  with its QuantizeReport (grid). The CLI entry point main() lives
  in cli.py.
Result:     `from resonance.quantize import quantize, detect_onsets`
  just works.
"""
from resonance.quantize.grid import (
    PERCUSSIVE_CONFIDENCE,
    PERCUSSIVE_MIN_GAP_S,
    GridHit,
    QuantizeReport,
    quantize,
)
from resonance.quantize.onsets import Onset, detect_onsets, spectral_flux

__all__ = [
    "PERCUSSIVE_CONFIDENCE",
    "PERCUSSIVE_MIN_GAP_S",
    "GridHit",
    "QuantizeReport",
    "quantize",
    "Onset",
    "detect_onsets",
    "spectral_flux",
]
