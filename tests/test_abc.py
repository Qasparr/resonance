# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_abc.py -- script-style tests for resonance.abc.

Run:  python3 tests/test_abc.py        (from the repo root)
   or python3 -m pytest tests/test_abc.py

Style: each test prints "  ok: <name>"; the end prints
"<N> abc tests passed." Any failure prints a FAIL line and exits 1 --
the first red line is the diagnosis.
"""
import os
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from resonance.abc import (
    BUILTIN_BPM,
    BUILTIN_EVENT_COUNT,
    BUILTIN_FIRST_MIDI,
    BUILTIN_SECONDS,
    BUILTIN_TOTAL_BEATS,
    BUILTIN_TUNE,
    ABCError,
    parse_abc,
    write_abc,
)
from resonance.abc.render import note_events, render_tune
from resonance.core.io import read_wav, write_wav
from resonance.core.notes import midi_to_freq

PASSED = 0
SR = 44100


def check(name, fn):
    """Run one test; print ok, or FAIL and exit 1."""
    global PASSED
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- report, then stop
        print(f"  FAIL: {name}: {type(exc).__name__}: {exc}")
        sys.exit(1)
    PASSED += 1
    print(f"  ok: {name}")


def events_key(events):
    return [(e.kind, e.midi, e.start, e.dur) for e in events]


# -- parsing the built-in tune -------------------------------------------------
def t_parse_builtin_event_count():
    tune = parse_abc(BUILTIN_TUNE)
    assert len(tune.events) == BUILTIN_EVENT_COUNT, len(tune.events)
check("t_parse_builtin_event_count", t_parse_builtin_event_count)


def t_parse_builtin_first_pitch():
    # The tune opens on middle C: MIDI 60, 261.63 Hz via core.notes.
    tune = parse_abc(BUILTIN_TUNE)
    first = tune.events[0]
    assert first.kind == "note"
    assert first.midi == BUILTIN_FIRST_MIDI == 60, first.midi
    assert abs(midi_to_freq(first.midi) - 261.625565) < 1e-3
check("t_parse_builtin_first_pitch", t_parse_builtin_first_pitch)


def t_parse_builtin_headers_and_timing():
    tune = parse_abc(BUILTIN_TUNE)
    assert tune.headers["X"] == "1"
    assert tune.headers["T"] == "The North Gate"
    assert tune.headers["K"] == "C"
    assert tune.beats_per_bar == Fraction(4, 1)
    assert tune.default_len == Fraction(1, 2)  # L:1/8
    assert tune.bpm == BUILTIN_BPM == 120.0
    # Exact rational timing: the last event ends on beat 32.
    last = tune.events[-1]
    assert last.start + last.dur == Fraction(BUILTIN_TOTAL_BEATS, 1)
    # A rest is present (bar 4) and carries no pitch.
    rests = [e for e in tune.events if e.kind == "rest"]
    assert len(rests) == 1 and rests[0].midi is None
check("t_parse_builtin_headers_and_timing", t_parse_builtin_headers_and_timing)


def t_parse_accidentals_and_octaves():
    # ^F in K:C is F#4 (66); the bar memory holds it; | resets it.
    tune = parse_abc("X:1\nK:C\nM:4/4\nL:1/4\nQ:120\n^F F | F z |\n")
    midis = [e.midi for e in tune.events if e.kind == "note"]
    assert midis == [66, 66, 65], midis
    # Octave marks: C, is C3 (48), c' is C6 (84).
    tune2 = parse_abc("X:1\nK:C\nM:4/4\nL:1/4\nQ:120\nC, c' |\n")
    midis2 = [e.midi for e in tune2.events if e.kind == "note"]
    assert midis2 == [48, 84], midis2
    # Key signature: K:G makes bare F into F#.
    tune3 = parse_abc("X:1\nK:G\nM:4/4\nL:1/4\nQ:120\nF |\n")
    assert tune3.events[0].midi == 66, tune3.events[0].midi
check("t_parse_accidentals_and_octaves", t_parse_accidentals_and_octaves)


# -- round-trip -----------------------------------------------------------------
def t_roundtrip_writer_parser_preserves_events():
    tune = parse_abc(BUILTIN_TUNE)
    text = write_abc(tune)
    tune2 = parse_abc(text)
    assert events_key(tune.events) == events_key(tune2.events)
    # And the text is valid-looking ABC with all six headers.
    for field in ("X:", "T:", "M:", "L:", "Q:", "K:"):
        assert field in text, field
check("t_roundtrip_writer_parser_preserves_events", t_roundtrip_writer_parser_preserves_events)


def t_roundtrip_chromatic_tune():
    # A tune with accidentals, octave jumps, and odd lengths must survive.
    src = ("X:7\nT:Chromatic\nM:3/4\nL:1/8\nQ:1/4=90\nK:F\n"
           "^C, _B, =B c'2 | z/2 D3/2 E/4 F// |\n")
    tune = parse_abc(src)
    tune2 = parse_abc(write_abc(tune))
    assert events_key(tune.events) == events_key(tune2.events)
check("t_roundtrip_chromatic_tune", t_roundtrip_chromatic_tune)


# -- rendering ------------------------------------------------------------------
def t_render_expected_duration():
    # 32 beats at 120 BPM = 16.0 s, plus the 0.25 s release tail.
    tune = parse_abc(BUILTIN_TUNE)
    audio = render_tune(tune, sr=SR, stereo=True)
    assert audio.dtype == np.float32 and audio.ndim == 2 and audio.shape[0] == 2
    expected = int(round(BUILTIN_SECONDS * SR)) + int(round(0.25 * SR))
    assert audio.shape[1] == expected, (audio.shape[1], expected)
check("t_render_expected_duration", t_render_expected_duration)


def t_render_non_silent_and_pitched():
    tune = parse_abc(BUILTIN_TUNE)
    audio = render_tune(tune, sr=SR, stereo=False)
    assert audio.ndim == 1
    assert float(np.max(np.abs(audio))) > 0.1, "rendered silence"
    # The first note's on/off list carries the right pitch from core.notes.
    first = note_events(tune)[0]
    assert abs(first[2] - midi_to_freq(60)) < 1e-9
    # Rests contribute no note events: 31 events, 1 rest -> 30 notes.
    assert len(note_events(tune)) == BUILTIN_EVENT_COUNT - 1
check("t_render_non_silent_and_pitched", t_render_non_silent_and_pitched)


def t_render_wav_roundtrip_via_core_io():
    # render -> write_wav -> read_wav: the shared core.io path, verified.
    tune = parse_abc(BUILTIN_TUNE)
    audio = render_tune(tune, sr=SR, stereo=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tune.wav")
        assert write_wav(path, audio, SR) == path
        back, sr = read_wav(path)
        assert sr == SR
        assert back.shape == audio.shape and back.dtype == np.float32
        # 16-bit quantization: worst case half an LSB of error.
        assert float(np.max(np.abs(back - audio))) < 2.0 / 32767
check("t_render_wav_roundtrip_via_core_io", t_render_wav_roundtrip_via_core_io)


# -- malformed input: loud errors, never silent mis-parses -----------------------
def t_malformed_raises_abc_error():
    bad_cases = {
        "empty string": "",
        "whitespace only": "   \n  ",
        "missing X: header": "T:No X\nK:C\nC D E |\n",
        "bad note letter": "X:1\nK:C\nM:4/4\nL:1/4\nQ:120\nH2 |\n",
        "stray character": "X:1\nK:C\nM:4/4\nL:1/4\nQ:120\nC D E ! |\n",
        "zero denominator": "X:1\nK:C\nM:4/4\nL:1/4\nQ:120\nC/0 |\n",
        "stray length": "X:1\nK:C\nM:4/4\nL:1/4\nQ:120\n2 C |\n",
        "tie unsupported": "X:1\nK:C\nM:4/4\nL:1/4\nQ:120\nC- C |\n",
        "bad key": "X:1\nK:Zz\nM:4/4\nL:1/4\nQ:120\nC |\n",
        "headers but no body": "X:1\nT:Empty\nK:C\nM:4/4\nL:1/4\nQ:120\n",
        "bad meter": "X:1\nK:C\nM:banana\nL:1/4\nQ:120\nC |\n",
    }
    for name, text in bad_cases.items():
        try:
            parse_abc(text)
        except ABCError:
            continue
        raise AssertionError(f"malformed input accepted silently: {name}")
check("t_malformed_raises_abc_error", t_malformed_raises_abc_error)


print(f"\n{PASSED} abc tests passed.")
