# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/pitch/__init__.py -- the auto-tune subpackage.

Hypothesis: pitch work is three separable jobs -- detect the pitch
  (YIN, detect.py), move the pitch (phase vocoder, shift.py), and
  decide how far toward the scale each voiced segment goes (tune.py).
  The honest rule binding them: every correction is reported
  per-segment with its confidence and its signed cent amount --
  no silent "enhancement," ever.
Method:     re-export the YIN tracker (detect), the phase-vocoder
  pitch shifter (shift), and the two-mode auto-tune engine with its
  TuneReport (tune). The CLI entry point main() lives in cli.py.
Result:     `from resonance.pitch import autotune, track_f0` just works.
"""
from resonance.pitch.detect import FramePitch, track_f0, yin_frame
from resonance.pitch.shift import phase_vocoder, pitch_shift
from resonance.pitch.tune import (
    CHROMATIC,
    MAJOR,
    MINOR_NATURAL,
    SCALES,
    TuneReport,
    TuneSegment,
    autotune,
)

__all__ = [
    "FramePitch",
    "track_f0",
    "yin_frame",
    "phase_vocoder",
    "pitch_shift",
    "CHROMATIC",
    "MAJOR",
    "MINOR_NATURAL",
    "SCALES",
    "TuneReport",
    "TuneSegment",
    "autotune",
]
