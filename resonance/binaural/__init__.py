# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/binaural/__init__.py -- entrainment-tone generators and sessions.

Public surface:
  generator : binaural beats, monaural beats, isochronic tones, BANDS table
  session   : timed session scripts with raised-cosine crossfades
  adaptive  : BPM-estimation HEURISTIC + pulse-following session render

Honesty: entrainment and Solfeggio material belong to an experimental
wellness tradition. Nothing here claims medical or therapeutic efficacy.
Adaptive BPM is a documented heuristic, not a measurement.
"""
from resonance.binaural import adaptive, generator, session

__all__ = ["adaptive", "generator", "session"]
