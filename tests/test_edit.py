# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_edit.py -- script-style tests for resonance.core.edit.

Run:  python3 tests/test_edit.py        (from the repo root)
   or python3 -m pytest tests/test_edit.py

Style: each test prints "  ok: <name>"; the end prints
"<N> edit tests passed." Any failure raises immediately.

These prove sample-accurate behavior: exact counts, exact content,
bit-identical regions outside crossfades, and real filter/limiter
physics (FFT-measured), not vibes.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.core import edit

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


SR = 44100


def _sine(freq, seconds, amp=0.5, sr=SR):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _tone_amp(x, freq, sr=SR):
    """Amplitude of the freq component via quadrature correlation."""
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(len(x), dtype=np.float64) / sr
    c = np.exp(-2j * np.pi * freq * t)
    return float(2.0 * abs(np.mean(x * c)))


# -- trim / split -----------------------------------------------------------
def t_trim_exact_counts_and_content():
    a = _sine(440, 1.0)
    t = edit.trim(a, 100, 1100)
    assert len(t) == 1000
    assert np.array_equal(t, a[100:1100])  # window, not a process
    assert len(edit.trim(a, 0, None)) == len(a)
check("t_trim_exact_counts_and_content", t_trim_exact_counts_and_content)


def t_trim_rejects_empty():
    try:
        edit.trim(_sine(440, 0.1), 500, 100)
    except ValueError:
        return
    raise AssertionError("empty trim accepted")
check("t_trim_rejects_empty", t_trim_rejects_empty)


def t_split_by_samples_and_seconds():
    a = _sine(440, 2.0)
    head, tail = edit.split(a, SR)  # sample index
    assert len(head) == SR and len(tail) == SR
    assert np.array_equal(head, a[:SR]) and np.array_equal(tail, a[SR:])
    head2, tail2 = edit.split(a, 1.0, sample_rate=SR)  # seconds
    assert np.array_equal(head2, head) and np.array_equal(tail2, tail)
    try:
        edit.split(a, 0)
    except ValueError:
        return
    raise AssertionError("degenerate split accepted")
check("t_split_by_samples_and_seconds", t_split_by_samples_and_seconds)


# -- splice -----------------------------------------------------------------
def t_splice_sample_accurate():
    a = _sine(440, 1.0)
    b = _sine(660, 1.0)
    xf = 1000
    out = edit.splice(a, b, crossfade_samples=xf)
    na, nb = len(a), len(b)
    assert len(out) == na + nb - xf
    # Outside the crossfade window: bit-identical to the sources.
    assert np.array_equal(out[: na - xf], a[: na - xf])
    assert np.array_equal(out[na:], b[xf:])
    # The crossfade region is continuous: no sample-to-sample jump larger
    # than a smooth tone could produce (a hard-splice click would be ~1.0).
    assert float(np.max(np.abs(np.diff(out)))) < 0.2
check("t_splice_sample_accurate", t_splice_sample_accurate)


def t_splice_rejects_mismatched_kinds():
    a = _sine(440, 0.1)
    s = np.stack([a, a])
    try:
        edit.splice(a, s, crossfade_samples=10)
    except ValueError:
        return
    raise AssertionError("mono/stereo splice accepted")
check("t_splice_rejects_mismatched_kinds", t_splice_rejects_mismatched_kinds)


# -- mix --------------------------------------------------------------------
def t_mix_clips_sample_additive():
    a = _sine(440, 1.0)
    b = _sine(660, 1.0)
    m = edit.mix_clips([a, b], gains=[0.3, 0.7])
    expect = 0.3 * a + 0.7 * b
    assert m.shape == (2, SR)
    assert np.allclose(m[0], expect, atol=1e-6)
    assert np.allclose(m[1], expect, atol=1e-6)
check("t_mix_clips_sample_additive", t_mix_clips_sample_additive)


# -- fades ------------------------------------------------------------------
def t_fades_exact_endpoints():
    a = np.full(1000, 0.8, dtype=np.float32)
    full = np.float32(0.8)  # float32(0.8) != python 0.8: compare in-kind
    fi = edit.fade_in(a, 200, kind="linear")
    assert fi[0] == 0.0 and fi[199] == full and fi[500] == full
    fo = edit.fade_out(a, 200, kind="linear")
    assert fo[-1] == 0.0 and fo[-200] == full and fo[0] == full
check("t_fades_exact_endpoints", t_fades_exact_endpoints)


def t_exponential_fade_monotonic():
    a = np.ones(1000, dtype=np.float32)
    fe = edit.fade_in(a, 500, kind="exponential")
    region = fe[:500]
    assert np.all(np.diff(region) > 0), "exponential fade not monotonic"
    assert region[0] == 0.0 and region[-1] == 1.0
    # Convex (x^2): starts gentler than linear.
    lin = edit.fade_in(a, 500, kind="linear")[:500]
    assert region[250] < lin[250]
check("t_exponential_fade_monotonic", t_exponential_fade_monotonic)


# -- normalize ---------------------------------------------------------------
def t_normalize_peak_and_silence():
    a = _sine(440, 1.0, amp=0.2)
    n = edit.normalize(a, target=0.9)
    assert abs(float(np.max(np.abs(n))) - 0.9) < 1e-6
    z = np.zeros(500, dtype=np.float32)
    assert np.array_equal(edit.normalize(z), z)  # silent stays silent
check("t_normalize_peak_and_silence", t_normalize_peak_and_silence)


# -- biquad -------------------------------------------------------------------
def t_biquad_lowpass_attenuates_highs():
    # Two tones: 100 Hz (pass) + 8000 Hz (stop); corner at 1000 Hz.
    sig = _sine(100, 3.0, amp=0.4) + _sine(8000, 3.0, amp=0.4)
    out = edit.biquad(sig, "lowpass", 1000.0, sample_rate=SR)
    tail_in, tail_out = sig[SR:], out[SR:]  # skip the start-up transient
    lo_in, lo_out = _tone_amp(tail_in, 100), _tone_amp(tail_out, 100)
    hi_in, hi_out = _tone_amp(tail_in, 8000), _tone_amp(tail_out, 8000)
    assert lo_out / lo_in > 0.9, f"passband damaged: {lo_out/lo_in}"
    assert hi_out / hi_in < 0.05, f"stopband leaked: {hi_out/hi_in}"
check("t_biquad_lowpass_attenuates_highs", t_biquad_lowpass_attenuates_highs)


def t_biquad_highpass_attenuates_lows():
    sig = _sine(100, 3.0, amp=0.4) + _sine(8000, 3.0, amp=0.4)
    out = edit.biquad(sig, "highpass", 1000.0, sample_rate=SR)
    tail_in, tail_out = sig[SR:], out[SR:]
    assert _tone_amp(tail_out, 100) / _tone_amp(tail_in, 100) < 0.05
    assert _tone_amp(tail_out, 8000) / _tone_amp(tail_in, 8000) > 0.9
check("t_biquad_highpass_attenuates_lows", t_biquad_highpass_attenuates_lows)


def t_biquad_peaking_boosts_band():
    sig = _sine(1000, 3.0, amp=0.4)
    out = edit.biquad(sig, "peaking", 1000.0, q=1.0, gain_db=12.0,
                      sample_rate=SR)
    ratio = _tone_amp(out[SR:], 1000) / _tone_amp(sig[SR:], 1000)
    # +12 dB = 3.98x; allow filter-settling tolerance.
    assert 3.4 < ratio < 4.6, f"peaking ratio {ratio}"
check("t_biquad_peaking_boosts_band", t_biquad_peaking_boosts_band)


def t_biquad_rejects_bad_params():
    try:
        edit.biquad(_sine(440, 0.1), "lowpass", SR, sample_rate=SR)  # >= Nyquist
    except ValueError:
        pass
    else:
        raise AssertionError("above-Nyquist corner accepted")
    try:
        edit.biquad(_sine(440, 0.1), "bandpass", 1000.0, sample_rate=SR)
    except ValueError:
        return
    raise AssertionError("bad kind accepted")
check("t_biquad_rejects_bad_params", t_biquad_rejects_bad_params)


# -- soft limiter ---------------------------------------------------------------
def t_soft_limiter_bounds_without_hard_clip():
    hot = _sine(440, 1.0, amp=3.0)  # 3x over the ceiling
    out = edit.soft_limiter(hot, ceiling=0.9)
    assert np.all(np.abs(out) < 0.9), "ceiling exceeded"
    # tanh is asymptotic: no sample is pinned exactly at the ceiling, so
    # there are no flat-top plateaus the way hard clipping makes.
    assert not np.any(np.abs(out) == 0.9), "samples pinned at ceiling"
    # ...yet it genuinely limits: the peak approaches the ceiling.
    assert float(np.max(np.abs(out))) > 0.85
check("t_soft_limiter_bounds_without_hard_clip", t_soft_limiter_bounds_without_hard_clip)


def t_soft_limiter_transparent_when_quiet():
    quiet = _sine(440, 1.0, amp=0.1)
    out = edit.soft_limiter(quiet, ceiling=0.9)
    # Small-signal: tanh(z) ~= z, so quiet audio passes nearly untouched.
    assert np.allclose(out, quiet, rtol=0.01)
check("t_soft_limiter_transparent_when_quiet", t_soft_limiter_transparent_when_quiet)


# -- chain ----------------------------------------------------------------------
def t_apply_chain_runs_and_rejects_unknown():
    a = _sine(440, 1.0, amp=0.8)
    out = edit.apply_chain(a, [
        {"type": "gain", "amount": -6},
        {"type": "lowpass", "freq": 5000},
        {"type": "limiter", "ceiling": 0.9},
        {"type": "normalize", "target": 0.8},
    ], sample_rate=SR)
    assert np.all(np.isfinite(out))
    assert abs(float(np.max(np.abs(out))) - 0.8) < 1e-6
    try:
        edit.apply_chain(a, [{"type": "phaser"}])
    except ValueError:
        return
    raise AssertionError("unknown effect accepted")
check("t_apply_chain_runs_and_rejects_unknown", t_apply_chain_runs_and_rejects_unknown)


def t_gain_db_math():
    a = _sine(440, 0.5)
    # -6.0206 dB is exactly half amplitude.
    assert np.allclose(edit.gain(a, -6.0206), a * 0.5, atol=1e-6)
    assert np.allclose(edit.gain(a, 2.0, unit="linear"), a * 2.0, atol=1e-6)
check("t_gain_db_math", t_gain_db_math)


print(f"\n{PASSED} edit tests passed.")
