# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_quantize.py -- script-style tests for resonance.quantize.

Run:  python3 tests/test_quantize.py      (from the repo root)
   or python3 -m pytest tests/test_quantize.py

Style: each test prints "  ok: <name>"; the end prints
"<N> quantize tests passed." Any failure raises immediately.
Ground truth is synthetic (drum patterns with known hit times) --
the only honest ground truth without a labeled corpus.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.quantize import detect_onsets, quantize

PASSED = 0
SR = 44100


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


def drum_hit(sr=SR, dur_s=0.25, freq=180.0):
    """Synthetic drum hit: decaying sine burst + noise click."""
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    env = np.exp(-t * 30.0)
    tone = np.sin(2 * np.pi * freq * t) * env
    click = np.zeros(n)
    click[:int(0.005 * sr)] = np.random.RandomState(1).randn(int(0.005 * sr))
    return 0.8 * tone + 0.5 * click * np.exp(-t * 200.0)


def drum_pattern(hit_times, sr=SR, dur_s=2.0):
    """Assemble hits at exact times -> mono audio."""
    x = np.zeros(int(dur_s * sr))
    hit = drum_hit(sr)
    for ht in hit_times:
        i = int(ht * sr)
        x[i:i + len(hit)] += hit[:max(0, min(len(hit), len(x) - i))]
    return x / max(1e-9, np.abs(x).max()) * 0.9


def t_onsets_find_drum_hits():
    hits = [0.25, 0.75, 1.25, 1.75]
    x = drum_pattern(hits)
    onsets = detect_onsets(x, SR)
    assert len(onsets) == 4, f"want 4 onsets, got {len(onsets)}"
    for o, ht in zip(onsets, hits):
        assert abs(o.time_s - ht) < 0.020, \
            f"onset {o.time_s:.3f}s vs true {ht:.3f}s"
        assert o.confidence > 0.3


def t_onsets_silence_empty():
    assert detect_onsets(np.zeros(SR), SR) == []
    assert detect_onsets(np.zeros(0), SR) == []


def t_onsets_confidence_ordering():
    # Loud hit + quiet hit: the loud one must be more confident.
    x = np.zeros(2 * SR)
    loud = drum_hit() * 1.0
    quiet = drum_hit() * 0.2
    x[int(0.3 * SR):int(0.3 * SR) + len(loud)] += loud
    x[int(1.0 * SR):int(1.0 * SR) + len(quiet)] += quiet
    onsets = detect_onsets(x, SR, sensitivity=0.8)
    assert len(onsets) >= 2, f"want >=2 onsets, got {len(onsets)}"
    assert onsets[0].confidence >= onsets[1].confidence


def t_strength_zero_untouched():
    hits = [0.25, 0.75, 1.25]
    x = drum_pattern(hits, dur_s=2.0)
    out, report = quantize(x, SR, bpm=120.0, strength=0.0)
    assert np.array_equal(out, x), "strength=0 must be bit-identical"
    assert report.n_hits == 0  # and the report says nothing moved


def t_slice_lands_on_grid():
    # 120 BPM -> grid every 0.5 s. Drag hits +/-40 ms off the grid.
    grid = [0.5, 1.0, 1.5]
    dragged = [0.54, 0.96, 1.53]
    x = drum_pattern(dragged, dur_s=2.0)
    out, report = quantize(x, SR, bpm=120.0, strength=1.0, mode="slice")
    assert report.n_hits == 3
    assert all(h.method == "slice" for h in report.hits)
    # Re-detect onsets on the output: they must sit on the grid now.
    onsets = detect_onsets(out, SR)
    assert len(onsets) == 3, f"want 3 onsets after, got {len(onsets)}"
    for o, g in zip(onsets, grid):
        assert abs(o.time_s - g) < 0.010, \
            f"hit at {o.time_s:.3f}s not on grid {g:.3f}s"


def t_strength_half_moves_halfway():
    dragged = [0.54, 1.04]
    x = drum_pattern(dragged, dur_s=2.0)
    out, report = quantize(x, SR, bpm=120.0, strength=0.5, mode="slice")
    # 50% of the 40 ms drag = ~20 ms residual offset from grid.
    onsets = detect_onsets(out, SR)
    assert len(onsets) == 2
    for o, g in zip(onsets, [0.5, 1.0]):
        assert abs(abs(o.time_s - g) - 0.020) < 0.010, \
            f"50% strength should leave ~20ms, got {o.time_s - g:+.3f}s"


def t_warp_mode_legato():
    # Warp on percussive material must also converge (different mover).
    dragged = [0.54, 0.96]
    x = drum_pattern(dragged, dur_s=2.0)
    out, report = quantize(x, SR, bpm=120.0, strength=1.0, mode="warp")
    assert all(h.method == "warp" for h in report.hits)
    onsets = detect_onsets(out, SR)
    assert len(onsets) == 2
    for o, g in zip(onsets, [0.5, 1.0]):
        assert abs(o.time_s - g) < 0.015, \
            f"warp missed grid: {o.time_s:.3f}s vs {g:.3f}s"


def t_auto_picks_slice_for_drums():
    x = drum_pattern([0.5, 1.0, 1.5], dur_s=2.0)
    out, report = quantize(x, SR, bpm=120.0, strength=1.0, mode="auto")
    assert report.mode == "slice", \
        f"auto should pick slice for drums, got {report.mode}"


def t_auto_picks_warp_for_legato():
    # Smooth vibrato-ish tone with soft amplitude swells: weak flux.
    t = np.arange(2 * SR) / SR
    vib = np.sin(2 * np.pi * 220 * t + 2 * np.sin(2 * np.pi * 5 * t))
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)  # 2 Hz swells, no transients
    x = (0.4 * vib * swell).astype(np.float64)
    out, report = quantize(x, SR, bpm=120.0, strength=1.0, mode="auto")
    # Either no onsets (honest: nothing to move) or warp mode.
    assert report.mode in ("auto", "warp"), \
        f"legato must not be sliced, got {report.mode}"
    if report.n_hits:
        assert all(h.method == "warp" for h in report.hits)


def t_report_never_silent():
    x = drum_pattern([0.54, 1.04], dur_s=2.0)
    out, report = quantize(x, SR, bpm=120.0, strength=1.0, mode="slice")
    assert report.n_hits == 2
    for h in report.hits:
        assert h.grid_s >= 0 and abs(h.shift_s) > 0
        assert 0.0 <= h.confidence <= 1.0
    assert report.max_shift_ms > 0
    assert "slice" in report.summary()


def t_bad_bpm_raises():
    try:
        quantize(drum_pattern([0.5]), SR, bpm=0.0)
    except ValueError:
        return
    raise AssertionError("bpm<=0 must raise ValueError")


def t_bad_mode_raises():
    try:
        quantize(drum_pattern([0.5]), SR, bpm=120.0, mode="swing")
    except ValueError:
        return
    raise AssertionError("bad mode must raise ValueError")


def t_stereo_channel_coherent():
    x = drum_pattern([0.54, 1.04], dur_s=2.0)
    stereo = np.stack([x, x * 0.8])
    out, report = quantize(stereo, SR, bpm=120.0, strength=1.0,
                           mode="slice")
    assert out.shape == stereo.shape
    # Both channels moved identically (ratio preserved sample-wise
    # where both are nonzero): timing moves are channel-coherent.
    nz = np.abs(out[0]) > 1e-4
    ratio = out[1][nz] / out[0][nz]
    assert np.allclose(ratio, 0.8, atol=0.05), \
        "channels diverged: timing moves must be coherent"


def t_empty_input():
    out, report = quantize(np.zeros(0), SR, bpm=120.0)
    assert out.size == 0 and report.n_hits == 0


def t_cli_help():
    r = subprocess.run(
        [sys.executable, "-m", "resonance.quantize.cli", "--help"],
        capture_output=True, text=True, timeout=60, env=_env(),
        cwd=str(Path.cwd()))
    assert r.returncode == 0 and "resonance-quantize" in r.stdout


def t_cli_onsets_smoke():
    with tempfile.TemporaryDirectory() as td:
        from resonance.core.io import write_wav
        wav = os.path.join(td, "drums.wav")
        write_wav(wav, drum_pattern([0.25, 0.75]).astype("float32"), SR)
        r = subprocess.run(
            [sys.executable, "-m", "resonance.quantize.cli", "onsets",
             "--input", wav],
            capture_output=True, text=True, timeout=120, env=_env(),
            cwd=str(Path.cwd()))
        assert r.returncode == 0, r.stderr
        assert "onsets" in r.stdout


def t_cli_quantize_smoke():
    with tempfile.TemporaryDirectory() as td:
        from resonance.core.io import read_wav, write_wav
        wav = os.path.join(td, "drums.wav")
        write_wav(wav, drum_pattern([0.54, 1.04], dur_s=2.0).astype("float32"),
                  SR)
        out = os.path.join(td, "q.wav")
        r = subprocess.run(
            [sys.executable, "-m", "resonance.quantize.cli", "quantize",
             "--input", wav, "--output", out, "--bpm", "120",
             "--strength", "100", "--mode", "slice"],
            capture_output=True, text=True, timeout=300, env=_env(),
            cwd=str(Path.cwd()))
        assert r.returncode == 0, r.stderr
        assert "hit(s)" in r.stdout, r.stdout[:300]
        audio, sr = read_wav(out)
        assert sr == SR and len(audio) > 0


def _env():
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


check("onsets_find_drum_hits", t_onsets_find_drum_hits)
check("onsets_silence_empty", t_onsets_silence_empty)
check("onsets_confidence_ordering", t_onsets_confidence_ordering)
check("strength_zero_untouched", t_strength_zero_untouched)
check("slice_lands_on_grid", t_slice_lands_on_grid)
check("strength_half_moves_halfway", t_strength_half_moves_halfway)
check("warp_mode_legato", t_warp_mode_legato)
check("auto_picks_slice_for_drums", t_auto_picks_slice_for_drums)
check("auto_picks_warp_for_legato", t_auto_picks_warp_for_legato)
check("report_never_silent", t_report_never_silent)
check("bad_bpm_raises", t_bad_bpm_raises)
check("bad_mode_raises", t_bad_mode_raises)
check("stereo_channel_coherent", t_stereo_channel_coherent)
check("empty_input", t_empty_input)
check("cli_help", t_cli_help)
check("cli_onsets_smoke", t_cli_onsets_smoke)
check("cli_quantize_smoke", t_cli_quantize_smoke)

print(f"\n{PASSED} quantize tests passed.")
