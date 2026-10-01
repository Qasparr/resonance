# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_synth.py -- script-style tests for resonance.synth.

Run:  python3 tests/test_synth.py        (from the repo root)
   or python3 -m pytest tests/test_synth.py

Style: each test prints "  ok: <name>"; the end prints
"<N> synth tests passed." Any failure prints a FAIL line and exits 1 --
the first red line is the diagnosis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from resonance.synth.sequencer import render_pattern, step_times
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


def spectral_centroid(x, sr):
    """Center of mass of the magnitude spectrum, in Hz."""
    x = np.asarray(x, dtype=np.float64)
    mag = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / sr)
    total = float(np.sum(mag))
    assert total > 0.0, "centroid of silence is undefined"
    return float(np.sum(freqs * mag) / total)


def is_non_silent(x, floor=0.1):
    return float(np.max(np.abs(x))) > floor


# -- voices: shape, dtype, audibility ----------------------------------------
def t_voices_render_mono_float32():
    for name, voice in VOICES.items():
        x = voice()
        assert isinstance(x, np.ndarray), name
        assert x.dtype == np.float32, (name, x.dtype)
        assert x.ndim == 1, (name, x.shape)  # mono (N,)
        assert x.shape[0] > 0, name
check("t_voices_render_mono_float32", t_voices_render_mono_float32)


def t_voices_non_silent():
    for name, voice in VOICES.items():
        assert is_non_silent(voice()), f"{name} rendered silence"
check("t_voices_non_silent", t_voices_non_silent)


def t_voice_durations_bounded():
    # Contract: kick < 0.6 s; open hat audibly longer than closed hat.
    assert kick().shape[0] / SR < 0.6, "kick too long"
    assert open_hat().shape[0] > closed_hat().shape[0], "open hat not longer"
    assert rimshot().shape[0] / SR < 0.1, "rimshot should be a blip"
check("t_voice_durations_bounded", t_voice_durations_bounded)


def t_kick_spectral_centroid_low():
    # The kick is a pitch-swept sine: its energy lives in the basement.
    c = spectral_centroid(kick(), SR)
    assert c < 300.0, f"kick centroid {c:.1f} Hz not < 300 Hz"
check("t_kick_spectral_centroid_low", t_kick_spectral_centroid_low)


def t_closed_hat_spectral_centroid_high():
    # The closed hat is highpassed noise: its energy lives in the treble.
    c = spectral_centroid(closed_hat(), SR)
    assert c > 5000.0, f"closed hat centroid {c:.1f} Hz not > 5 kHz"
check("t_closed_hat_spectral_centroid_high", t_closed_hat_spectral_centroid_high)


def t_velocity_scales_and_silences():
    full = kick(velocity=1.0)
    half = kick(velocity=0.5)
    # Same shape, half the amplitude (linear velocity scaling).
    assert half.shape == full.shape
    ratio = float(np.max(np.abs(half))) / float(np.max(np.abs(full)))
    assert abs(ratio - 0.5) < 1e-6, f"velocity ratio {ratio}"
    silent = kick(velocity=0.0)
    assert float(np.max(np.abs(silent))) == 0.0, "velocity 0 not silent"
check("t_velocity_scales_and_silences", t_velocity_scales_and_silences)


def t_voices_deterministic():
    # Same call, same buffer: the seeded RNG makes hits reproducible.
    assert np.array_equal(snare(), snare()), "snare not deterministic"
    assert np.array_equal(clap(), clap()), "clap not deterministic"
check("t_voices_deterministic", t_voices_deterministic)


def t_voices_headroom():
    # Peak-normalized voices stay within [-1, 1] at velocity 1.0.
    for name, voice in VOICES.items():
        peak = float(np.max(np.abs(voice(velocity=1.0))))
        assert peak <= 1.0 + 1e-6, (name, peak)
check("t_voices_headroom", t_voices_headroom)


# -- sequencer ----------------------------------------------------------------
def t_sequencer_exact_duration():
    # 2 bars at 120 BPM: 2 * 4 beats * 0.5 s = 4.0 s, whatever the pattern.
    pattern = {
        "kick": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "closed_hat": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    }
    mix = render_pattern(pattern, bpm=120, bars=2)
    expected = int(round(2 * 4 * (60.0 / 120) * SR))
    assert mix.dtype == np.float32, mix.dtype
    assert mix.shape == (2, expected), (mix.shape, expected)
    assert is_non_silent(mix), "pattern rendered silence"
check("t_sequencer_exact_duration", t_sequencer_exact_duration)


def t_sequencer_step_count():
    # bars * 16 steps of grid: step_times is the sequencer's own clock.
    assert len(step_times(1, 120)) == 16
    assert len(step_times(3, 100)) == 48
check("t_sequencer_step_count", t_sequencer_step_count)


def t_swing_shifts_offbeats():
    # At 120 BPM a 16th lasts 0.125 s = 5512.5 samples; swing 0.3 delays
    # odd steps by 0.3 of that = 1653.75 -> 1654 samples.
    t0 = step_times(1, 120, swing=0.0, sr=SR)
    t1 = step_times(1, 120, swing=0.3, sr=SR)
    step_samples = (60.0 / 120) / 4.0 * SR
    expect = int(round(0.3 * step_samples))
    # Sample-quantized grid: each absolute offset is rounded independently,
    # so a difference of rounded values is exact to within one sample.
    assert abs((t1[1] - t0[1]) - expect) <= 1, (t1[1] - t0[1], expect)
    assert abs((t1[3] - t0[3]) - expect) <= 1, "every odd step must swing"
    assert t1[0] == t0[0] and t1[2] == t0[2], "even steps must not move"
check("t_swing_shifts_offbeats", t_swing_shifts_offbeats)


def t_swing_audible_in_render():
    # One closed-hat hit on step 1 (an off-beat): the rendered onset must
    # move by the same sample offset the grid predicts.
    grid = [0] * 16
    grid[1] = 1
    straight = render_pattern({"closed_hat": grid}, bpm=120, bars=1, swing=0.0)
    swung = render_pattern({"closed_hat": grid}, bpm=120, bars=1, swing=0.3)
    def onset(mix):
        mono = np.abs(mix[0].astype(np.float64))
        idx = np.nonzero(mono > 1e-4)[0]
        assert idx.size, "hit rendered silence"
        return int(idx[0])
    shift = onset(swung) - onset(straight)
    expect = int(round(0.3 * (60.0 / 120) / 4.0 * SR))
    assert abs(shift - expect) <= 2, (shift, expect)
check("t_swing_audible_in_render", t_swing_audible_in_render)


def t_sequencer_rejects_bad_input():
    good = [0] * 16
    for bad_pattern, why in [
        ({"theremin": good}, "unknown voice"),
        ({"kick": [1] * 15}, "short grid"),
        ({"kick": [1] * 16 + [1]}, "long grid"),
        ({"kick": [1 if i else -1 for i in range(16)]}, "negative step"),
    ]:
        try:
            render_pattern(bad_pattern, bpm=120)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad pattern: {why}")
    for bad_swing in (-0.1, 0.7):
        try:
            render_pattern({"kick": good}, bpm=120, swing=bad_swing)
        except ValueError:
            continue
        raise AssertionError(f"accepted swing {bad_swing}")
    try:
        render_pattern({"kick": good}, bpm=0)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted bpm=0")
check("t_sequencer_rejects_bad_input", t_sequencer_rejects_bad_input)


def t_pattern_chaining():
    # A list of patterns chains bar by bar: 2 different bars = 2 bars out.
    bar_a = {"kick": [1] + [0] * 15}
    bar_b = {"snare": [1] + [0] * 15}
    mix = render_pattern([bar_a, bar_b], bpm=120)
    expected = int(round(2 * 4 * (60.0 / 120) * SR))
    assert mix.shape == (2, expected), mix.shape
    # The two bars differ: bar 1 opens with a kick, bar 2 with a snare.
    half = expected // 2
    e_first = float(np.sum(mix[:, :half] ** 2))
    e_second = float(np.sum(mix[:, half:] ** 2))
    assert e_first > 0 and e_second > 0, "chained bars silent"
check("t_pattern_chaining", t_pattern_chaining)


print(f"\n{PASSED} synth tests passed.")
