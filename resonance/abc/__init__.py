# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/abc/__init__.py -- the ABC notation subpackage.

Hypothesis: parsing, writing, and rendering ABC tunes belong behind one
  import surface: text in, events out, text back, audio out.
Method:     re-export the parser (parse_abc, Tune, Event, ABCError), the
  writer (write_abc), the renderer (note_events, render_events,
  render_tune), and the built-in test tune. Voices here are defined in
  render.py; this subpackage never imports resonance.synth internals.
Result:     `from resonance.abc import parse_abc, BUILTIN_TUNE` just works.
"""
from resonance.abc.parser import ABCError, Event, Tune, parse_abc
from resonance.abc.render import note_events, render_events, render_tune
from resonance.abc.tunes import (
    BUILTIN_BPM,
    BUILTIN_EVENT_COUNT,
    BUILTIN_FIRST_MIDI,
    BUILTIN_SECONDS,
    BUILTIN_TOTAL_BEATS,
    BUILTIN_TUNE,
)
from resonance.abc.writer import write_abc

__all__ = [
    "ABCError",
    "Event",
    "Tune",
    "parse_abc",
    "write_abc",
    "note_events",
    "render_events",
    "render_tune",
    "BUILTIN_TUNE",
    "BUILTIN_EVENT_COUNT",
    "BUILTIN_FIRST_MIDI",
    "BUILTIN_TOTAL_BEATS",
    "BUILTIN_BPM",
    "BUILTIN_SECONDS",
]
