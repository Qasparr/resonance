# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/abc/tunes.py -- built-in test tunes for the ABC subpackage.

Hypothesis: the ABC parser, writer, and renderer need a fixed,
  known-good tune to test against -- one whose event count, pitches, and
  total duration are hand-verifiable.
Method:     ship one short original tune as a plain Python string
  constant. It was composed for this engine (a simple C-major phrase,
  not copied from any existing melody), so there are no copyright
  questions: it is an original exercise tune.
Result:     BUILTIN_TUNE: 8 bars of 4/4 at Q:1/4=120, L:1/8 -- 31 events
  (30 notes + 1 rest) over exactly 32 beats = 16.0 seconds. The first
  note is middle C (MIDI 60).

No medical or therapeutic claims are made about anything here.
"""

# An original 8-bar phrase in C major, composed for RESONANCE v0.1.0 as a
# deterministic test fixture. Event counts per bar (notes+rests):
#   bar 1: C2 D2 E2 G2            -> 4 events
#   bar 2: A2 G2 E2 D2            -> 4 events
#   bar 3: C2 D2 E E D D          -> 6 events
#   bar 4: C4 z4                  -> 2 events (1 note + 1 rest)
#   bar 5: E2 F2 G2 A2            -> 4 events
#   bar 6: G2 F2 E2 D2            -> 4 events
#   bar 7: E2 D2 C D E2           -> 5 events
#   bar 8: D4 C4                  -> 2 events
# Total: 31 events, 8 bars * 4 beats = 32 beats, at 120 BPM = 16 seconds.
BUILTIN_TUNE = """X:1
T:The North Gate
M:4/4
L:1/8
Q:1/4=120
K:C
C2 D2 E2 G2 | A2 G2 E2 D2 |
C2 D2 E E D D | C4 z4 |
E2 F2 G2 A2 | G2 F2 E2 D2 |
E2 D2 C D E2 | D4 C4 |]
"""

# Hand-computed expectations the test suite asserts against the parse of
# BUILTIN_TUNE. Kept beside the tune so a tune edit forces a re-check.
BUILTIN_EVENT_COUNT = 31
BUILTIN_FIRST_MIDI = 60      # middle C
BUILTIN_TOTAL_BEATS = 32     # 8 bars of 4/4
BUILTIN_BPM = 120.0
BUILTIN_SECONDS = 16.0       # 32 beats at 120 BPM

__all__ = [
    "BUILTIN_TUNE",
    "BUILTIN_EVENT_COUNT",
    "BUILTIN_FIRST_MIDI",
    "BUILTIN_TOTAL_BEATS",
    "BUILTIN_BPM",
    "BUILTIN_SECONDS",
]
