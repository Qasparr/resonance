# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_core.py -- script-style tests for resonance.core.

Run:  python3 tests/test_core.py        (from the repo root)
   or python3 -m pytest tests/test_core.py

Style: each test prints "  ok: <name>"; the end prints
"<N> core tests passed." Any failure raises immediately.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.core import buffers, config, io, notes

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


SR = 44100


def _sine(freq, seconds, sr=SR):
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# -- buffers ---------------------------------------------------------------
def t_validate_mono_ok():
    a = _sine(440, 1.0)
    out = buffers.validate(a)
    assert out.dtype == np.float32 and out.shape == a.shape
check("t_validate_mono_ok", t_validate_mono_ok)


def t_validate_stereo_ok():
    a = np.stack([_sine(440, 1.0), _sine(660, 1.0)])
    out = buffers.validate(a)
    assert out.shape == (2, SR)
check("t_validate_stereo_ok", t_validate_stereo_ok)


def t_validate_rejects_bad_shapes():
    for bad in (np.zeros((3, 100), dtype=np.float32),   # 3 channels
                np.zeros((2, 2, 10), dtype=np.float32),  # 3-D
                np.zeros((0,), dtype=np.float32)):       # empty
        try:
            buffers.validate(bad)
        except ValueError:
            continue
        raise AssertionError(f"bad shape accepted: {bad.shape}")
check("t_validate_rejects_bad_shapes", t_validate_rejects_bad_shapes)


def t_validate_rejects_nan():
    bad = _sine(440, 0.1)
    bad[5] = np.nan
    try:
        buffers.validate(bad)
    except ValueError:
        return
    raise AssertionError("NaN accepted")
check("t_validate_rejects_nan", t_validate_rejects_nan)


def t_mix_is_gain_weighted_sum():
    a = _sine(440, 1.0)
    b = _sine(660, 1.0)
    m = buffers.mix(a, b, gains=[0.5, 2.0])
    assert m.shape == (2, SR)
    expect = 0.5 * a + 2.0 * b
    assert np.allclose(m[0], expect, atol=1e-6)
    assert np.allclose(m[1], expect, atol=1e-6)
check("t_mix_is_gain_weighted_sum", t_mix_is_gain_weighted_sum)


def t_mix_rejects_gain_mismatch():
    try:
        buffers.mix(_sine(440, 0.1), gains=[1.0, 2.0])
    except ValueError:
        return
    raise AssertionError("gain mismatch accepted")
check("t_mix_rejects_gain_mismatch", t_mix_rejects_gain_mismatch)


def t_normalize_peak_target():
    a = _sine(440, 1.0) * 0.3
    n = buffers.normalize(a, target=0.9)
    assert abs(float(np.max(np.abs(n))) - 0.9) < 1e-6
check("t_normalize_peak_target", t_normalize_peak_target)


def t_normalize_silent_stays_silent():
    z = np.zeros(1000, dtype=np.float32)
    n = buffers.normalize(z)
    assert np.array_equal(n, z)  # never divides by zero
check("t_normalize_silent_stays_silent", t_normalize_silent_stays_silent)


def t_fades_hit_endpoints():
    a = np.ones(1000, dtype=np.float32)
    fi = buffers.fade_in(a, 100, kind="linear")
    assert fi[0] == 0.0 and fi[99] == 1.0 and fi[500] == 1.0
    fo = buffers.fade_out(a, 100, kind="linear")
    assert fo[-1] == 0.0 and fo[-100] == 1.0 and fo[0] == 1.0
    fe = buffers.fade_in(a, 100, kind="exponential")
    assert fe[0] == 0.0 and fe[99] == 1.0
check("t_fades_hit_endpoints", t_fades_hit_endpoints)


# -- io --------------------------------------------------------------------
def t_wav_roundtrip_mono_sample_exact():
    a = _sine(440, 0.5)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "m.wav")
        io.write_wav(p, a, SR)
        back, sr = io.read_wav(p)
    assert sr == SR and back.shape == a.shape and back.dtype == np.float32
    # Sample-exact: both sides use the same int16 quantization, so the
    # round-trip must equal the quantized original bit-for-bit.
    expect = np.round(np.clip(a, -1, 1) * 32767) / 32767
    assert np.array_equal(back, expect.astype(np.float32))
check("t_wav_roundtrip_mono_sample_exact", t_wav_roundtrip_mono_sample_exact)


def t_wav_roundtrip_stereo_sample_exact():
    s = np.stack([_sine(440, 0.5), _sine(660, 0.5)])
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "s.wav")
        io.write_wav(p, s, SR)
        back, sr = io.read_wav(p)
    assert sr == SR and back.shape == (2, SR // 2)
    expect = np.round(np.clip(s, -1, 1) * 32767) / 32767
    assert np.array_equal(back, expect.astype(np.float32))
check("t_wav_roundtrip_stereo_sample_exact", t_wav_roundtrip_stereo_sample_exact)


def t_wav_clips_hot_signal():
    hot = np.full(100, 2.0, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "h.wav")
        io.write_wav(p, hot, SR)
        back, _ = io.read_wav(p)
    assert np.all(back == 1.0)  # clipped, never wrapped
check("t_wav_clips_hot_signal", t_wav_clips_hot_signal)


# -- notes -----------------------------------------------------------------
def t_note_freq_known_values():
    assert notes.note_to_freq("A4") == 440.0
    assert notes.note_to_midi("A4") == 69
    assert notes.note_to_midi("C4") == 60
    assert abs(notes.note_to_freq("C4") - 261.63) < 0.01
    assert notes.midi_to_note(69) == "A4"
    assert notes.freq_to_midi(440.0) == 69
    assert notes.note_to_midi("Bb3") == notes.note_to_midi("A#3") == 58
    # round-trips
    assert notes.midi_to_note(notes.note_to_midi("F#5")) == "F#5"
check("t_note_freq_known_values", t_note_freq_known_values)


def t_note_rejects_garbage():
    for bad in ("H4", "A", "", "C##4"):
        try:
            notes.note_to_midi(bad)
        except ValueError:
            continue
        raise AssertionError(f"bad note accepted: {bad!r}")
check("t_note_rejects_garbage", t_note_rejects_garbage)


def t_solfeggio_data_integrity():
    assert set(notes.SOLFEGGIO_FREQS) == {174, 285, 396, 417, 528, 639, 741,
                                          852, 963}
    assert len(notes.SOLFEGGIO) == 9
    assert notes.solfeggio_name(528) == "Transformation"
    assert notes.solfeggio_name(174) == "Foundation"
    assert all(isinstance(v, str) and v for v in notes.SOLFEGGIO.values())
    assert notes.nearest_solfeggio(530) == 528
    try:
        notes.SOLFEGGIO[440] = "nope"
    except TypeError:
        pass
    else:
        raise AssertionError("Solfeggio table is mutable")
check("t_solfeggio_data_integrity", t_solfeggio_data_integrity)


# -- config ----------------------------------------------------------------
def t_config_defaults_and_frames():
    c = config.SessionConfig()
    assert c.sample_rate == 44100 and c.duration == 60.0
    assert c.n_frames == 44100 * 60
    assert c.fade_samples("in") == 2 * 44100
check("t_config_defaults_and_frames", t_config_defaults_and_frames)


def t_config_rejects_bad_values():
    for kw in ({"sample_rate": 0}, {"duration": -1.0},
               {"fade_in": -0.5}, {"duration": 1.0, "fade_in": 1.0,
                                   "fade_out": 1.0}):
        try:
            config.SessionConfig(**kw)
        except ValueError:
            continue
        raise AssertionError(f"bad config accepted: {kw}")
check("t_config_rejects_bad_values", t_config_rejects_bad_values)


print(f"\n{PASSED} core tests passed.")
