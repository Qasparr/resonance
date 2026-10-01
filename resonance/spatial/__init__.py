# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/spatial/__init__.py -- 3D spatial audio (v0.2.0 group D).

Public surface:
  hrtf      : parametric spherical-head binaural panner (ITD + ILD),
              optional measured-HRTF loader
  distance  : 1/r gain with near-field clamp + air-absorption lowpass
  ambisonic : first-order B-format encode/decode (AmbiX: ACN + SN3D)
  orbit     : circular motion around the head -- a CREATIVE
              spatialization effect; no medical or health claims
  cli       : the `resonance-spatial` executable (pan/orbit subcommands)

Audio convention (workspace-wide): numpy float32; mono (N,),
stereo (2, N); sample_rate int, default 44100. Azimuth convention:
0 = front, +90 = hard right, -90 = hard left; elevation 0 = horizon,
+ = up.
"""
from resonance.spatial import ambisonic, cli, distance, hrtf, orbit

__all__ = ["ambisonic", "cli", "distance", "hrtf", "orbit"]
