# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_pitch.py -- script-style tests for resonance.pitch.

Run:  python3 tests/test_pitch.py        (from the repo root)
   or python3 -m pytest tests/test_pitch.py

Style: each test prints "  ok: <name>"; the end prints
"<N> pitch tests passed." Any failure raises immediately.
Ground truth is synthetic (sines with known frequencies) -- the
only honest ground truth without a labeled corpus.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.core.notes import midi_to_freq
from resonance.pitch import (
    autotune,
    phase_vocoder,
    pitch_shift,
    track_f0,
    yin_frame,
)

PASSED = 0
SR = 44100


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


def sine(freq, dur_s, sr=SR, amp=0.5):
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def dominant_freq(x, sr):
    """Peak of the magnitude spectrum (for shift verification)."""
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return freqs[int(np.argmax(mag[1:])) + 1]


def t_yin_pure_sine():
    f0, conf, voiced = yin_frame(sine(440.0, 0.2) * np.hanning(int(0.2 * SR)),
                                 SR)
    assert voiced and f0 is not None
    assert abs(f0 - 440.0) < 2.0, f"yin off: {f0}"
    assert conf > 0.8, f"low confidence on clean sine: {conf}"


def t_yin_silence_unvoiced():
    f0, conf, voiced = yin_frame(np.zeros(2048), SR)
    assert not voiced and f0 is None and conf == 0.0


def t_yin_melody_two_notes():
    x = np.concatenate([sine(440.0, 0.5), sine(523.25, 0.5)])
    frames = track_f0(x, SR)
    voiced = [f for f in frames if f.voiced]
    assert len(voiced) > 10
    first = np.median([f.f0_hz for f in voiced[:len(voiced) // 3]])
    last = np.median([f.f0_hz for f in voiced[-len(voiced) // 3:]])
    assert abs(first - 440.0) < 5.0, f"first note: {first}"
    assert abs(last - 523.25) < 6.0, f"second note: {last}"


def t_yin_confidence_noisy():
    rng = np.random.RandomState(7)
    noisy = sine(440.0, 0.5) + 0.8 * rng.randn(int(0.5 * SR))
    frames = track_f0(noisy, SR)
    voiced = [f for f in frames if f.voiced]
    # Noise must not produce HIGH-confidence voicing everywhere.
    mean_conf = np.mean([f.confidence for f in frames])
    assert mean_conf < 0.95, f"suspiciously certain on noise: {mean_conf}"
    assert len(frames) > 0  # frames exist; pitches not invented


def t_shift_zero_is_identity():
    x = sine(220.0, 0.5)
    y = pitch_shift(x, SR, 0.0)
    assert np.allclose(x, y), "0-semitone shift must be bit-identical"


def t_shift_octave_up():
    x = sine(220.0, 1.0)
    y = pitch_shift(x, SR, 12.0)
    assert len(y) == len(x), "duration must be preserved"
    f = dominant_freq(y, SR)
    assert abs(f - 440.0) < 8.0, f"octave shift landed at {f}"


def t_shift_fifth_down():
    x = sine(440.0, 1.0)
    y = pitch_shift(x, SR, -7.0)
    f = dominant_freq(y, SR)
    expect = 440.0 * 2 ** (-7 / 12)
    assert abs(f - expect) < 8.0, f"fifth-down landed at {f}, want {expect}"


def t_phase_vocoder_rate_one():
    x = sine(330.0, 0.5)
    y = phase_vocoder(x, 1.0)
    # Rate 1.0 must be near-transparent (phase-vocoder round trip).
    assert len(y) == len(x)
    corr = np.corrcoef(x, y)[0, 1]
    assert corr > 0.95, f"rate-1.0 round trip degraded: r={corr}"


def t_phase_vocoder_bad_rate():
    try:
        phase_vocoder(sine(440.0, 0.2), 0.0)
    except ValueError:
        return
    raise AssertionError("rate<=0 must raise ValueError")


def t_autotune_corrective_pulls_flat_note():
    # A4 sung 40 cents flat: 440 * 2**(-40/1200) ~= 429.9 Hz.
    flat = sine(429.9, 1.0)
    out, report = autotune(flat, SR, scale="chromatic", mode="corrective")
    assert report.n_tuned == 1, f"expected 1 segment, got {report.n_tuned}"
    seg = report.segments[0]
    assert seg.correction_cents > 0, "flat note must correct upward"
    assert abs(seg.correction_cents - 40 * 0.7) < 12.0, \
        f"corrective amount wrong: {seg.correction_cents}"
    assert seg.confidence > 0.5
    # The output pitch must be closer to 440 than the input was.
    f_in = dominant_freq(flat, SR)
    f_out = dominant_freq(out, SR)
    assert abs(f_out - 440.0) < abs(f_in - 440.0), \
        f"not corrected: in={f_in:.1f} out={f_out:.1f}"


def t_autotune_effect_hard_snaps():
    flat = sine(429.9, 1.0)
    out, report = autotune(flat, SR, scale="chromatic", mode="effect")
    assert report.n_tuned == 1
    seg = report.segments[0]
    assert abs(seg.correction_cents - 40.0) < 12.0, \
        f"effect must apply the full correction: {seg.correction_cents}"
    f_out = dominant_freq(out, SR)
    assert abs(f_out - 440.0) < 6.0, f"effect snap missed: {f_out}"


def t_autotune_report_never_silent():
    # In-tune note: correction ~0 but the segment is still REPORTED.
    x = sine(440.0, 1.0)
    out, report = autotune(x, SR, mode="corrective")
    assert report.n_tuned == 1
    assert abs(report.segments[0].correction_cents) < 8.0


def t_autotune_silence_untouched():
    x = np.zeros(int(0.5 * SR))
    out, report = autotune(x, SR)
    assert np.allclose(out, x), "silence must pass through untouched"
    assert report.n_tuned == 0, "no segments on silence, honestly"


def t_autotune_bad_mode():
    try:
        autotune(sine(440.0, 0.3), SR, mode="vibey")
    except ValueError:
        return
    raise AssertionError("bad mode must raise ValueError")


def t_autotune_bad_scale():
    try:
        autotune(sine(440.0, 0.3), SR, scale="dorian#4")
    except ValueError:
        return
    raise AssertionError("unknown scale must raise ValueError")


def t_autotune_major_scale_target():
    # F4 (349.23) is not in C major... it is (F). Use F#4 (369.99):
    # nearest C-major degrees are F (349.23) and G (392.00).
    x = sine(369.99, 1.0)
    out, report = autotune(x, SR, scale="major", mode="effect")
    assert report.n_tuned == 1
    tgt = report.segments[0].target_midi
    assert tgt in (65.0, 67.0), f"F# must snap to F or G, got {tgt}"


def t_cli_help():
    r = subprocess.run(
        [sys.executable, "-m", "resonance.pitch.cli", "--help"],
        capture_output=True, text=True, timeout=60,
        env=_env(), cwd=str(Path.cwd()))
    assert r.returncode == 0 and "resonance-tune" in r.stdout


def t_cli_detect_smoke():
    with tempfile.TemporaryDirectory() as td:
        from resonance.core.io import write_wav
        wav = os.path.join(td, "a.wav")
        write_wav(wav, sine(440.0, 0.5).astype("float32"), SR)
        r = subprocess.run(
            [sys.executable, "-m", "resonance.pitch.cli", "detect",
             "--input", wav],
            capture_output=True, text=True, timeout=120,
            env=_env(), cwd=str(Path.cwd()))
        assert r.returncode == 0, r.stderr
        assert "440" in r.stdout, r.stdout[:200]


def t_cli_tune_smoke():
    with tempfile.TemporaryDirectory() as td:
        from resonance.core.io import read_wav, write_wav
        wav = os.path.join(td, "flat.wav")
        write_wav(wav, sine(429.9, 0.8).astype("float32"), SR)
        out = os.path.join(td, "fixed.wav")
        r = subprocess.run(
            [sys.executable, "-m", "resonance.pitch.cli", "tune",
             "--input", wav, "--output", out, "--mode", "effect"],
            capture_output=True, text=True, timeout=300,
            env=_env(), cwd=str(Path.cwd()))
        assert r.returncode == 0, r.stderr
        assert "segment(s) tuned" in r.stdout, r.stdout[:300]
        audio, sr = read_wav(out)
        assert sr == SR and len(audio) > 0


def _env():
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


check("yin_pure_sine", t_yin_pure_sine)
check("yin_silence_unvoiced", t_yin_silence_unvoiced)
check("yin_melody_two_notes", t_yin_melody_two_notes)
check("yin_confidence_noisy", t_yin_confidence_noisy)
check("shift_zero_is_identity", t_shift_zero_is_identity)
check("shift_octave_up", t_shift_octave_up)
check("shift_fifth_down", t_shift_fifth_down)
check("phase_vocoder_rate_one", t_phase_vocoder_rate_one)
check("phase_vocoder_bad_rate", t_phase_vocoder_bad_rate)
check("autotune_corrective_pulls_flat_note", t_autotune_corrective_pulls_flat_note)
check("autotune_effect_hard_snaps", t_autotune_effect_hard_snaps)
check("autotune_report_never_silent", t_autotune_report_never_silent)
check("autotune_silence_untouched", t_autotune_silence_untouched)
check("autotune_bad_mode", t_autotune_bad_mode)
check("autotune_bad_scale", t_autotune_bad_scale)
check("autotune_major_scale_target", t_autotune_major_scale_target)
check("cli_help", t_cli_help)
check("cli_detect_smoke", t_cli_detect_smoke)
check("cli_tune_smoke", t_cli_tune_smoke)

print(f"\n{PASSED} pitch tests passed.")
