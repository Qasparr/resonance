# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_arpeggiator.py -- script-style tests for resonance.synth.arpeggiator.

Run:  python3 tests/test_arpeggiator.py        (from the repo root)
   or python3 -m pytest tests/test_arpeggiator.py

Style: each test prints "  ok: <name>"; the end prints
"<N> arpeggiator tests passed." Any failure prints a FAIL line and
exits 1 -- the first red line is the diagnosis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from resonance.synth import Arpeggiator
from resonance.synth.arpeggiator import Arpeggiator as ArpDirect

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


def midis(events):
    return [m for _, m, _ in events]


def starts(events):
    return [s for s, _, _ in events]


def durs(events):
    return [d for _, _, d in events]


# -- pattern orderings ----------------------------------------------------------
def t_pattern_up():
    arp = Arpeggiator(pattern="up")
    ev = arp.arpeggiate([60, 64, 67], steps=6, step_dur_s=0.125)
    assert midis(ev) == [60, 64, 67, 60, 64, 67], midis(ev)
check("t_pattern_up", t_pattern_up)


def t_pattern_down():
    arp = Arpeggiator(pattern="down")
    ev = arp.arpeggiate([60, 64, 67], steps=6, step_dur_s=0.125)
    assert midis(ev) == [67, 64, 60, 67, 64, 60], midis(ev)
check("t_pattern_down", t_pattern_down)


def t_pattern_up_down():
    # Ping-pong without repeating the endpoints: 60,64,67,64 | 60,64,67,64.
    arp = Arpeggiator(pattern="up_down")
    ev = arp.arpeggiate([60, 64, 67], steps=8, step_dur_s=0.125)
    assert midis(ev) == [60, 64, 67, 64, 60, 64, 67, 64], midis(ev)
check("t_pattern_up_down", t_pattern_up_down)


def t_pattern_played_keeps_finger_order():
    # "played" walks the chord exactly as given -- not sorted.
    arp = Arpeggiator(pattern="played")
    ev = arp.arpeggiate([67, 60, 64], steps=4, step_dur_s=0.125)
    assert midis(ev) == [67, 60, 64, 67], midis(ev)
check("t_pattern_played_keeps_finger_order", t_pattern_played_keeps_finger_order)


def t_octave_range_spans():
    arp = Arpeggiator(pattern="up", range_octaves=2)
    ev = arp.arpeggiate([60, 64], steps=4, step_dur_s=0.125)
    assert midis(ev) == [60, 64, 72, 76], midis(ev)
    arp4 = Arpeggiator(pattern="up", range_octaves=4)
    ev4 = arp4.arpeggiate([60], steps=4, step_dur_s=0.125)
    assert midis(ev4) == [60, 72, 84, 96], midis(ev4)
check("t_octave_range_spans", t_octave_range_spans)


# -- gate / swing / timing -------------------------------------------------------
def t_gate_shortens_durations():
    arp = Arpeggiator(gate=0.5)
    ev = arp.arpeggiate([60, 64], steps=4, step_dur_s=0.2)
    assert all(abs(d - 0.1) < 1e-12 for d in durs(ev)), durs(ev)
    arp_full = Arpeggiator(gate=1.0)
    ev2 = arp_full.arpeggiate([60], steps=2, step_dur_s=0.2)
    assert all(abs(d - 0.2) < 1e-12 for d in durs(ev2))
check("t_gate_shortens_durations", t_gate_shortens_durations)


def t_swing_delays_odd_steps_like_sequencer():
    # Sequencer convention: odd steps arrive late by swing * step_dur.
    arp = Arpeggiator(swing=0.3)
    ev = arp.arpeggiate([60, 64, 67, 60], steps=4, step_dur_s=0.125)
    s = starts(ev)
    assert abs(s[0] - 0.0) < 1e-12, s
    assert abs(s[1] - (0.125 + 0.3 * 0.125)) < 1e-12, s
    assert abs(s[2] - 0.25) < 1e-12, s
    assert abs(s[3] - (0.375 + 0.3 * 0.125)) < 1e-12, s
    arp_straight = Arpeggiator(swing=0.0)
    ev0 = arp_straight.arpeggiate([60, 64, 67, 60], steps=4, step_dur_s=0.125)
    assert starts(ev0) == [0.0, 0.125, 0.25, 0.375], starts(ev0)
check("t_swing_delays_odd_steps_like_sequencer", t_swing_delays_odd_steps_like_sequencer)


def t_random_seeded_reproducible():
    a = Arpeggiator(pattern="random", seed=42)
    b = Arpeggiator(pattern="random", seed=42)
    c = Arpeggiator(pattern="random", seed=7)
    ea = a.arpeggiate([60, 64, 67], steps=16, step_dur_s=0.125)
    eb = b.arpeggiate([60, 64, 67], steps=16, step_dur_s=0.125)
    ec = c.arpeggiate([60, 64, 67], steps=16, step_dur_s=0.125)
    assert midis(ea) == midis(eb), "same seed must give same sequence"
    assert all(m in (60, 64, 67) for m in midis(ea)), "draws outside the pool"
    assert midis(ea) != midis(ec), "different seeds should differ"
check("t_random_seeded_reproducible", t_random_seeded_reproducible)


# -- chord holding ---------------------------------------------------------------
def t_hold_and_override():
    arp = Arpeggiator(pattern="up")
    arp.hold([60, 64, 67])
    assert arp.chord == [60, 64, 67]
    ev = arp.arpeggiate(steps=3, step_dur_s=0.125)  # held chord used
    assert midis(ev) == [60, 64, 67]
    ev2 = arp.arpeggiate([72], steps=2, step_dur_s=0.125)  # one-call override
    assert midis(ev2) == [72, 72]
    assert arp.chord == [60, 64, 67], "override must not clobber the hold"
check("t_hold_and_override", t_hold_and_override)


# -- rendering ---------------------------------------------------------------------
def t_render_nonzero_expected_length():
    arp = Arpeggiator(pattern="up", gate=0.8)
    ev = arp.arpeggiate([60, 64, 67], steps=8, step_dur_s=0.125)
    audio = arp.render(ev, sr=SR, stereo=True)
    assert audio.dtype == np.float32, audio.dtype
    assert audio.ndim == 2 and audio.shape[0] == 2, audio.shape
    assert float(np.max(np.abs(audio))) > 0.1, "arpeggio rendered silence"
    # Last event: start 7*0.125 = 0.875, dur 0.1 -> 0.975 + 0.1 s tail.
    last_end = max(s + d for s, _, d in ev)
    expected = int(round((last_end + 0.10) * SR))
    assert audio.shape[1] == expected, (audio.shape[1], expected)
check("t_render_nonzero_expected_length", t_render_nonzero_expected_length)


def t_render_mono_and_gate_zero_silence():
    arp = Arpeggiator(gate=0.0)
    ev = arp.arpeggiate([60, 64], steps=4, step_dur_s=0.125)
    mono = arp.render(ev, sr=SR, stereo=False)
    assert mono.ndim == 1, mono.shape
    assert float(np.max(np.abs(mono))) == 0.0, "gate 0 must be silence"
    last_end = max(s + d for s, _, d in ev)
    assert mono.shape[0] == int(round((last_end + 0.10) * SR))
check("t_render_mono_and_gate_zero_silence", t_render_mono_and_gate_zero_silence)


def t_render_pitched_at_midi_freq():
    # The rendered note must carry its MIDI pitch: FFT peak at 440 Hz
    # for A4 (MIDI 69).
    from resonance.core.notes import midi_to_freq
    arp = Arpeggiator(pattern="played", gate=1.0)
    ev = arp.arpeggiate([69], steps=1, step_dur_s=0.5)
    mono = arp.render(ev, sr=SR, stereo=False)
    seg = mono[: int(0.4 * SR)]
    mag = np.abs(np.fft.rfft(seg.astype(np.float64)))
    freqs = np.fft.rfftfreq(seg.shape[0], d=1.0 / SR)
    peak_f = freqs[int(np.argmax(mag))]
    assert abs(peak_f - midi_to_freq(69)) < 5.0, (peak_f, midi_to_freq(69))
check("t_render_pitched_at_midi_freq", t_render_pitched_at_midi_freq)


# -- validation: loud errors ---------------------------------------------------------
def t_rejects_bad_input():
    cases = [
        (lambda: Arpeggiator(pattern="sideways"), "bad pattern"),
        (lambda: Arpeggiator(range_octaves=0), "range 0"),
        (lambda: Arpeggiator(range_octaves=5), "range 5"),
        (lambda: Arpeggiator(gate=1.5), "gate > 1"),
        (lambda: Arpeggiator(gate=-0.1), "gate < 0"),
        (lambda: Arpeggiator(swing=0.7), "swing > max"),
        (lambda: Arpeggiator().arpeggiate([60], steps=0), "steps 0"),
        (lambda: Arpeggiator().arpeggiate([], steps=4), "empty chord"),
        (lambda: Arpeggiator().arpeggiate([200], steps=4), "midi > 127"),
        (lambda: Arpeggiator().arpeggiate(steps=4), "no chord at all"),
        (lambda: Arpeggiator().arpeggiate([60], steps=4, step_dur_s=0),
         "step_dur 0"),
        (lambda: Arpeggiator().render([], sr=SR), "render empty events"),
    ]
    for fn, why in cases:
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError(f"accepted bad input: {why}")
check("t_rejects_bad_input", t_rejects_bad_input)


def t_exported_from_package():
    # The package surface exposes Arpeggiator (and it is the same class).
    assert Arpeggiator is ArpDirect
    import resonance.synth as synth_pkg
    assert "Arpeggiator" in synth_pkg.__all__
check("t_exported_from_package", t_exported_from_package)


print(f"\n{PASSED} arpeggiator tests passed.")
