# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/stems/__init__.py -- stem separation and the stem mixer (v0.2.0).

Hypothesis: separation and mixing belong together as one importable
  surface: separate.* turns a finished track into named stems through a
  real Demucs backend (or refuses loudly when none is present), and
  mixer.StemMixer turns named stems back into a mix with per-stem
  volume/mute/solo and per-stem tempo.
Method:     re-export the Separator contract, the DemucsAdapter, the
  loud error types, and the mixer surface.  Nothing here imports torch,
  demucs, or rubberband at load time -- numpy stays the only hard
  dependency, and the heavy guests are probed lazily.
Result:     `from resonance.stems import DemucsAdapter, StemMixer` just
  works, with or without any backend installed.

Honest limits (v0.2.0 contract): 4 stems -- vocals/drums/bass/other --
not arbitrary instruments; "isolate the guitar from the piano" is
beyond open tooling.  Absent backends produce loud, specific errors
naming the missing package/tool and how to install it -- never fakes.

No medical claims: this package moves audio samples around.
"""
from resonance.stems.mixer import (
    ResampleStretch,
    RubberBandAdapter,
    StemMixer,
    TimeStretch,
    TimeStretchUnavailable,
)
from resonance.stems.separate import (
    STEMS,
    DemucsAdapter,
    Separator,
    StemSeparationUnavailable,
)

__all__ = [
    "STEMS",
    "DemucsAdapter",
    "ResampleStretch",
    "RubberBandAdapter",
    "Separator",
    "StemMixer",
    "StemSeparationUnavailable",
    "TimeStretch",
    "TimeStretchUnavailable",
]
