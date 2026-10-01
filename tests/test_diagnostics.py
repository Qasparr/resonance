# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_diagnostics.py -- script-style tests for resonance.diagnostics.

Run:  python3 tests/test_diagnostics.py   (from the repo root)
   or python3 -m pytest tests/test_diagnostics.py

Style: each test prints "  ok: <name>"; the end prints
"<N> diagnostics tests passed." Any failure raises immediately.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.binaural import generator
from resonance.diagnostics import measure, verify

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


SR = 44100


def _render_10hz_528():
    return generator.binaural_beat(10.0, carrier=528, duration=4.0,
                                   sample_rate=SR)


# -- measure -------------------------------------------------------------------
def t_throughput_sane_positive_finite():
    stats = measure.measure_throughput(_render_10hz_528, n_runs=2)
    for key in ("samples_per_second", "wall_seconds", "seconds_per_run"):
        v = stats[key]
        assert isinstance(v, float), (key, v)
        assert math.isfinite(v) and v > 0, (key, v)
    assert stats["runs"] == 2
    assert stats["frames_per_run"] == 4 * SR
    assert stats["channels"] == 2
    # Cross-check: samples/sec == total samples / wall time, by definition.
    expect = (stats["samples_per_run"] * 2) / stats["wall_seconds"]
    assert abs(stats["samples_per_second"] - expect) < 1e-6 * expect
check("t_throughput_sane_positive_finite", t_throughput_sane_positive_finite)


def t_throughput_rejects_garbage_result():
    try:
        measure.measure_throughput(lambda: np.full(10, np.nan), n_runs=1)
    except AssertionError:
        pass
    else:
        raise AssertionError("NaN render accepted")
    try:
        measure.measure_throughput(lambda: np.array([]), n_runs=1)
    except AssertionError:
        return
    raise AssertionError("empty render accepted")
check("t_throughput_rejects_garbage_result", t_throughput_rejects_garbage_result)


# -- verify --------------------------------------------------------------------
def t_verify_passes_good_render():
    v = verify.verify_binaural(_render_10hz_528(), 10.0, 528,
                               sample_rate=SR, tol_hz=0.5)
    assert v["passed"] is True, v["errors"]
    assert abs(v["peak_l_hz"] - 523.0) <= 0.5
    assert abs(v["peak_r_hz"] - 533.0) <= 0.5
    assert abs(v["beat_measured_hz"] - 10.0) <= 0.5
    assert v["errors"] == []
check("t_verify_passes_good_render", t_verify_passes_good_render)


def t_verify_fails_mistuned_render():
    # Deliberately mistuned: rendered at 12 Hz, labeled as 10 Hz.
    bad = generator.binaural_beat(12.0, carrier=528, duration=4.0,
                                  sample_rate=SR)
    v = verify.verify_binaural(bad, 10.0, 528, sample_rate=SR, tol_hz=0.5)
    assert v["passed"] is False, "mistuned render passed verification"
    assert len(v["errors"]) == 3, v["errors"]  # L, R, and beat all off
    assert abs(v["beat_measured_hz"] - 12.0) <= 0.5
check("t_verify_fails_mistuned_render", t_verify_fails_mistuned_render)


def t_verify_tolerance_boundary():
    # A 10.4 Hz render labeled 10 Hz: within a 0.5 Hz tolerance it passes,
    # at 0.3 Hz it fails. The tolerance knob does what it says.
    near = generator.binaural_beat(10.4, carrier=528, duration=4.0,
                                   sample_rate=SR)
    v_loose = verify.verify_binaural(near, 10.0, 528, sample_rate=SR,
                                     tol_hz=0.5)
    v_tight = verify.verify_binaural(near, 10.0, 528, sample_rate=SR,
                                     tol_hz=0.3)
    assert v_loose["passed"] is True, v_loose["errors"]
    assert v_tight["passed"] is False, "tight tolerance passed a near-miss"
    assert any("beat" in e for e in v_tight["errors"])
check("t_verify_tolerance_boundary", t_verify_tolerance_boundary)


print(f"\n{PASSED} diagnostics tests passed.")
