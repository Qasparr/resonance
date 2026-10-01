# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/synth/__init__.py -- the 808-style synthesis subpackage.

Hypothesis: drum voices and the step sequencer belong together as one
  importable surface: voices render the sounds, the sequencer places them
  in time.
Method:     re-export the voice callables, the VOICES registry, and the
  sequencer entry points. Nothing here reaches into other subpackages.
Result:     `from resonance.synth import kick, render_pattern` just works.
"""
from resonance.synth.sequencer import MAX_SWING, STEPS_PER_BAR, render_pattern, step_times
from resonance.synth.voices import (
    VOICES,
    clap,
    closed_hat,
    cowbell,
    kick,
    open_hat,
    rimshot,
    snare,
)

__all__ = [
    "VOICES",
    "STEPS_PER_BAR",
    "MAX_SWING",
    "kick",
    "snare",
    "closed_hat",
    "open_hat",
    "clap",
    "cowbell",
    "rimshot",
    "step_times",
    "render_pattern",
]
