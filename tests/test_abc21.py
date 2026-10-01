# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_abc21.py -- script-style tests for resonance.abc.abc21.

Run:  python3 tests/test_abc21.py        (from the repo root)
   or python3 -m pytest tests/test_abc21.py

Style: each test prints "  ok: <name>"; the end prints
"<N> abc21 tests passed." Any failure prints a FAIL line and exits 1 --
the first red line is the diagnosis.
"""
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from resonance.abc.abc21 import (
    Event21,
    Tune21,
    parse_abc21,
    render_voice,
    tune_voice,
    write_abc21,
)
from resonance.abc.parser import ABCError
from resonance.abc.render import note_events, render_tune
from resonance.core.notes import midi_to_freq

PASSED = 0
SR = 44100

HDR = "X:1\nT:TwentyOne\nM:4/4\nL:1/4\nQ:120\nK:C\n"
HDR8 = "X:1\nT:TwentyOne\nM:4/4\nL:1/8\nQ:120\nK:C\n"


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


def evkey(events):
    """Full 2.1 event identity, including the new metadata fields."""
    return [(e.kind, e.midi, e.start, e.dur, e.voice, e.grace, e.tuplet,
             list(e.tie_splits)) for e in events]


def spectrum_peaks(x, sr, freqs, ratio=4.0):
    """Assert each freq in `freqs` is a local spectral peak (magnitude >
    ratio * median magnitude). Returns the peak frequencies found."""
    x = np.asarray(x, dtype=np.float64)
    mag = np.abs(np.fft.rfft(x))
    bins = np.fft.rfftfreq(x.shape[0], d=1.0 / sr)
    floor = float(np.median(mag)) + 1e-12
    found = []
    for f in freqs:
        idx = int(np.argmin(np.abs(bins - f)))
        assert mag[idx] > ratio * floor, (f, mag[idx], floor)
        found.append(float(bins[idx]))
    return found


# -- ties ----------------------------------------------------------------------------
def t_tie_merges_durations():
    tune = parse_abc21(HDR + "C- C |\n")
    assert len(tune.events) == 1, len(tune.events)
    e = tune.events[0]
    assert (e.kind, e.midi, e.start, e.dur) == ("note", 60, Fraction(0), Fraction(2))
    assert e.tie_splits == [Fraction(1)], e.tie_splits
    # Across a bar line: the whole point of a tie.
    tune2 = parse_abc21(HDR + "C- | C |\n")
    assert len(tune2.events) == 1
    assert tune2.events[0].dur == Fraction(2)
    # Chained ties.
    tune3 = parse_abc21(HDR + "C- C- C |\n")
    assert len(tune3.events) == 1
    assert tune3.events[0].dur == Fraction(3)
    assert tune3.events[0].tie_splits == [Fraction(1), Fraction(2)]
check("t_tie_merges_durations", t_tie_merges_durations)


def t_tie_into_chord_tone():
    # A tie may land on a chord tone: that tone's duration grows; the
    # chord's other tones start where the chord starts (beat 1 here).
    tune = parse_abc21(HDR + "C- [CE] |\n")
    assert len(tune.events) == 2, len(tune.events)
    by_midi = {e.midi: e for e in tune.events}
    assert (by_midi[60].start, by_midi[60].dur) == (Fraction(0), Fraction(2))
    assert (by_midi[64].start, by_midi[64].dur) == (Fraction(1), Fraction(1))
check("t_tie_into_chord_tone", t_tie_into_chord_tone)


def t_tie_misuse_raises():
    for bad, why in [
        (HDR + "C- D |\n", "tie across different pitches"),
        (HDR + "C- |\n", "dangling tie"),
        (HDR + "- C |\n", "stray tie"),
        (HDR + "z- C |\n", "tie on a rest"),
        (HDR + "[CE]- D |\n", "tie out of a chord"),
    ]:
        try:
            parse_abc21(bad)
        except ABCError:
            continue
        raise AssertionError(f"accepted bad tie: {why}")
check("t_tie_misuse_raises", t_tie_misuse_raises)


# -- chords ----------------------------------------------------------------------------
def t_chord_simultaneous_events():
    tune = parse_abc21(HDR + "[CEG] |\n")
    assert len(tune.events) == 3, len(tune.events)
    assert [e.midi for e in tune.events] == [60, 64, 67]
    assert all(e.start == Fraction(0) and e.dur == Fraction(1)
               for e in tune.events)
    # The clock advances by the chord (first tone's duration), not per tone.
    tune2 = parse_abc21(HDR + "[CE] D |\n")
    assert tune2.events[-1].midi == 62 and tune2.events[-1].start == Fraction(1)
check("t_chord_simultaneous_events", t_chord_simultaneous_events)


def t_chord_renders_simultaneous_pitches():
    # FFT must show BOTH fundamentals sounding at once.
    tune = parse_abc21(HDR + "[CE] |\n")
    audio = render_tune(tune, sr=SR, stereo=False)
    assert float(np.max(np.abs(audio))) > 0.1, "chord rendered silence"
    seg = audio[: int(0.8 * SR)]
    spectrum_peaks(seg, SR, [midi_to_freq(60), midi_to_freq(64)])
check("t_chord_renders_simultaneous_pitches", t_chord_renders_simultaneous_pitches)


def t_chord_misuse_raises():
    for bad, why in [
        (HDR + "[CE |\n", "unterminated chord"),
        (HDR + "[] |\n", "empty chord"),
        (HDR + "[[CE]] |\n", "nested chord"),
        (HDR + "[Cz] |\n", "rest in chord"),
        (HDR + "C ] |\n", "stray ]"),
    ]:
        try:
            parse_abc21(bad)
        except ABCError:
            continue
        raise AssertionError(f"accepted bad chord: {why}")
check("t_chord_misuse_raises", t_chord_misuse_raises)


# -- tuplets -----------------------------------------------------------------------------
def t_triplet_timing_exact():
    # (3 with L:1/8: three eighth notes in the time of two -> 1/3 beat each.
    tune = parse_abc21(HDR8 + "(3CDE |\n")
    assert len(tune.events) == 3, len(tune.events)
    for e, start in zip(tune.events, (Fraction(0), Fraction(1, 3), Fraction(2, 3))):
        assert e.dur == Fraction(1, 3), (e.midi, e.dur)
        assert e.start == start, (e.midi, e.start)
        assert e.tuplet == (3, 2), e.tuplet
    # They fill exactly one beat: 3 * 1/3.
    assert tune.events[-1].start + tune.events[-1].dur == Fraction(1)
check("t_triplet_timing_exact", t_triplet_timing_exact)


def t_tuplet_explicit_form():
    tune = parse_abc21(HDR8 + "(3:2:2CD |\n")
    assert len(tune.events) == 2
    assert all(e.dur == Fraction(1, 3) and e.tuplet == (3, 2)
               for e in tune.events)
    # (2 = 2 in the time of 3: two eighths (1/2 beat) * 3/2 = 3/4 each.
    tune2 = parse_abc21(HDR8 + "(2CD |\n")
    assert all(e.dur == Fraction(3, 4) and e.tuplet == (2, 3)
               for e in tune2.events)
check("t_tuplet_explicit_form", t_tuplet_explicit_form)


def t_tuplet_misuse_raises():
    for bad, why in [
        (HDR8 + "(5CDEFG |\n", "bare (5 without explicit q"),
        (HDR8 + "(3(3CDE |\n", "nested tuplet"),
        (HDR8 + "(3CD |\n", "tuplet runs out of notes"),
        (HDR8 + "(CD) |\n", "slur paren"),
        (HDR8 + "C D) |\n", "stray slur end"),
    ]:
        try:
            parse_abc21(bad)
        except ABCError:
            continue
        raise AssertionError(f"accepted bad tuplet: {why}")
check("t_tuplet_misuse_raises", t_tuplet_misuse_raises)


# -- grace notes ---------------------------------------------------------------------------
def t_grace_steals_from_following_note():
    # {g}a2 under L:1/8: a2 is 1 beat; the grace steals half (1/2 beat).
    tune = parse_abc21(HDR8 + "{g}a2 |\n")
    assert len(tune.events) == 2, len(tune.events)
    g, a = tune.events
    assert (g.kind, g.midi, g.start, g.dur, g.grace) == (
        "note", 79, Fraction(0), Fraction(1, 2), True)
    assert (a.kind, a.midi, a.start, a.dur, a.grace) == (
        "note", 81, Fraction(1, 2), Fraction(1, 2), False)
    # Two grace notes split the steal equally.
    tune2 = parse_abc21(HDR8 + "{ga}b2 |\n")
    g1, g2, b = tune2.events
    assert g1.dur == g2.dur == Fraction(1, 4), (g1.dur, g2.dur)
    assert (b.start, b.dur) == (Fraction(1, 2), Fraction(1, 2))
check("t_grace_steals_from_following_note", t_grace_steals_from_following_note)


def t_grace_misuse_raises():
    for bad, why in [
        (HDR8 + "{g} |\n", "dangling grace"),
        (HDR8 + "{g}z |\n", "grace before a rest"),
        (HDR8 + "{g |\n", "unterminated grace"),
        (HDR8 + "{}a |\n", "empty grace"),
        (HDR8 + "{g}- a |\n", "tied grace"),
    ]:
        try:
            parse_abc21(bad)
        except ABCError:
            continue
        raise AssertionError(f"accepted bad grace notes: {why}")
check("t_grace_misuse_raises", t_grace_misuse_raises)


# -- repeats ---------------------------------------------------------------------------------
def t_repeat_with_first_second_endings():
    tune = parse_abc21(HDR + "|: C D [1 E :| [2 F |\n")
    midis = [e.midi for e in tune.events if e.kind == "note"]
    assert midis == [60, 62, 64, 60, 62, 65], midis
    starts = [e.start for e in tune.events]
    assert starts == [Fraction(i) for i in range(6)], starts
    # The writer's beat map was recorded.
    assert len(tune.repeats) == 1, tune.repeats
    rep = tune.repeats[0]
    assert (rep["start"], rep["head_end"], rep["ending1_end"],
            rep["head2_end"], rep["ending2_end"]) == (
                Fraction(0), Fraction(2), Fraction(3),
                Fraction(5), Fraction(6)), rep
check("t_repeat_with_first_second_endings", t_repeat_with_first_second_endings)


def t_repeat_simple_doubles():
    tune = parse_abc21(HDR + "|: C D :| G |\n")
    midis = [e.midi for e in tune.events if e.kind == "note"]
    assert midis == [60, 62, 60, 62, 67], midis
check("t_repeat_simple_doubles", t_repeat_simple_doubles)


def t_repeat_misuse_raises():
    for bad, why in [
        (HDR + "C D :| |\n", "stray :|"),
        (HDR + "|: C D |\n", "unmatched |:"),
        (HDR + "[1 C |\n", "variant outside repeat"),
        (HDR + "|: C [1 D :| |\n", "[1 without [2"),
        (HDR + "|: C |: D :| :| |\n", "nested repeats"),
    ]:
        try:
            parse_abc21(bad)
        except ABCError:
            continue
        raise AssertionError(f"accepted bad repeat: {why}")
check("t_repeat_misuse_raises", t_repeat_misuse_raises)


# -- multi-voice -------------------------------------------------------------------------------
def t_multi_voice_independent_timelines():
    src = (HDR + "V:1\nV:2\n"
           "V:1\nC D |\n"
           "V:2\nG2 |\n")
    tune = parse_abc21(src)
    assert tune.voices == ["1", "2"], tune.voices
    v1 = [e for e in tune.events if e.voice == "1"]
    v2 = [e for e in tune.events if e.voice == "2"]
    assert [e.midi for e in v1] == [60, 62], [e.midi for e in v1]
    assert [e.midi for e in v2] == [67], [e.midi for e in v2]
    # Each voice restarts at beat 0: both lines sound simultaneously.
    assert v1[0].start == v2[0].start == Fraction(0)
    # Each voice renders alone, non-silent, at its own pitch.
    for voice, midi in (("1", 60), ("2", 67)):
        audio = render_voice(tune, voice, sr=SR, stereo=False)
        assert float(np.max(np.abs(audio))) > 0.1, f"voice {voice} silent"
        seg = audio[: int(0.4 * SR)]
        spectrum_peaks(seg, SR, [midi_to_freq(midi)])
    # ... and the full tune mixes both lines.
    both = render_tune(tune, sr=SR, stereo=False)
    spectrum_peaks(both[: int(0.4 * SR)], SR,
                   [midi_to_freq(60), midi_to_freq(67)])
check("t_multi_voice_independent_timelines", t_multi_voice_independent_timelines)


def t_voice_change_resets_accidental_memory():
    # ^F in voice 1 must not leak into voice 2.
    src = HDR + "V:1\nV:2\nV:1\n^F |\nV:2\nF |\n"
    tune = parse_abc21(src)
    by_voice = {}
    for e in tune.events:
        by_voice.setdefault(e.voice, []).append(e.midi)
    assert by_voice == {"1": [66], "2": [65]}, by_voice
check("t_voice_change_resets_accidental_memory", t_voice_change_resets_accidental_memory)


# -- loud refusal of the still-unsupported -------------------------------------------------------
def t_still_unsupported_raises_loudly():
    for bad, why in [
        (HDR + "K:G\nC |\n", "inline key change mid-tune"),
        (HDR + "M:3/4\nC |\n", "inline meter change mid-tune"),
        (HDR + "(C D) |\n", "slur"),
        (HDR + "!f!C |\n", "decoration"),
        (HDR + "C D E ! |\n", "stray character"),
    ]:
        try:
            parse_abc21(bad)
        except ABCError:
            continue
        raise AssertionError(f"silently accepted unsupported input: {why}")
check("t_still_unsupported_raises_loudly", t_still_unsupported_raises_loudly)


# -- writer round-trips ------------------------------------------------------------------------------
def t_writer_roundtrip_all_constructs():
    src = (HDR8 + "|: C- C [CEG] (3DEF {g}a2 [1 B :| [2 c' |\n")
    tune = parse_abc21(src)
    text = write_abc21(tune)
    for token in ("|:", "[CEG]", "(3:2:3", "{g}", "[1", ":|", "[2", "C-C"):
        assert token in text, (token, text)
    tune2 = parse_abc21(text)
    assert evkey(tune.events) == evkey(tune2.events), (
        evkey(tune.events), evkey(tune2.events))
    # The repeat beat map round-trips too.
    assert tune.repeats == tune2.repeats, (tune.repeats, tune2.repeats)
check("t_writer_roundtrip_all_constructs", t_writer_roundtrip_all_constructs)


def t_writer_roundtrip_voices():
    src = HDR + "V:1\nV:2\nV:1\nC D |\nV:2\nG A |\n"
    tune = parse_abc21(src)
    text = write_abc21(tune)
    assert "V:1" in text and "V:2" in text, text
    tune2 = parse_abc21(text)
    assert evkey(tune.events) == evkey(tune2.events)
    assert tune2.voices == ["1", "2"], tune2.voices
check("t_writer_roundtrip_voices", t_writer_roundtrip_voices)


def t_writer_plain_tune_no_invented_markup():
    # A v0.1.0 tune written by the 2.1 writer: no '-' is invented between
    # two separate same-pitch notes; no other 2.1 markup appears either.
    from resonance.abc import parse_abc as parse_v1
    tune = parse_v1(HDR + "C C D |\n")
    text = write_abc21(tune)
    assert "-" not in text, text
    assert "[" not in text and "(" not in text and "{" not in text, text
    tune2 = parse_abc21(text)
    assert [(e.kind, e.midi, e.start, e.dur) for e in tune.events] == \
           [(e.kind, e.midi, e.start, e.dur) for e in tune2.events]
check("t_writer_plain_tune_no_invented_markup", t_writer_plain_tune_no_invented_markup)


# -- render integration ----------------------------------------------------------------------------------
def t_render_21_tune_expected_shape():
    # 4/4, L:1/4, Q:120: |: C- C [CE] :| = 3-beat head doubled (6 beats)
    # plus a 1-beat rest. render_tune (v0.1.0, untouched) sizes the
    # buffer from NOTE events only, so the trailing rest does not extend
    # it: 6 beats = 3 s, plus the 0.25 s tail.
    tune = parse_abc21(HDR + "|: C- C [CE] :| z |\n")
    audio = render_tune(tune, sr=SR, stereo=True)
    assert audio.dtype == np.float32 and audio.shape[0] == 2
    expected = int(round(6 * 0.5 * SR)) + int(round(0.25 * SR))
    assert audio.shape[1] == expected, (audio.shape[1], expected)
    assert float(np.max(np.abs(audio))) > 0.1, "rendered silence"
    # Rests still contribute no note events: 2 heads x (1 tied + 2 chord).
    assert len(note_events(tune)) == 6, len(note_events(tune))
check("t_render_21_tune_expected_shape", t_render_21_tune_expected_shape)


print(f"\n{PASSED} abc21 tests passed.")
