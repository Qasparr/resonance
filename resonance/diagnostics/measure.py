# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/diagnostics/measure.py -- real throughput benchmarking.

Hypothesis: render functions can be honestly benchmarked by timing them
  with time.perf_counter and dividing real samples rendered by real
  seconds elapsed -- no simulated numbers, no estimates.
Method:     run render_fn(*args, **kwargs) n_runs times, time each with
  perf_counter, count result.size samples per run.
Observation: samples_per_second is finite and positive for any real
  renderer; a function returning garbage (non-array, empty) fails loudly.
Result:     a dict with wall time and throughput, asserted sane.

Uses only the render function's PUBLIC call signature -- measure never
reaches into module internals.
"""
import time

import numpy as np


def measure_throughput(render_fn, *args, n_runs=3, **kwargs):
    """Time render_fn and return real throughput numbers.

    Returns dict:
      runs               : n_runs
      frames_per_run     : samples per channel per run
      channels           : channel count
      samples_per_run    : frames * channels
      wall_seconds       : total elapsed seconds across all runs
      seconds_per_run    : wall_seconds / runs
      samples_per_second : total samples / wall_seconds (the headline)

    Asserts: the result is a finite non-empty float array, wall time is
    positive and finite, throughput is finite and positive. These are
    assertions, not warnings -- a benchmark that cannot produce a real
    number is a failed benchmark.
    """
    if n_runs < 1:
        raise ValueError(f"measure_throughput: n_runs must be >= 1, got {n_runs}")
    if not callable(render_fn):
        raise TypeError("measure_throughput: render_fn must be callable")
    walls = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = render_fn(*args, **kwargs)
        t1 = time.perf_counter()
        walls.append(t1 - t0)
    wall = float(sum(walls))
    # The result must be real audio: a finite, non-empty array. Anything
    # else means the render function is broken, and the benchmark says so.
    arr = np.asarray(result)
    assert arr.size > 0, "measure_throughput: render_fn returned empty result"
    assert np.all(np.isfinite(arr)), \
        "measure_throughput: render_fn returned non-finite samples"
    assert np.issubdtype(arr.dtype, np.floating), \
        f"measure_throughput: expected float audio, got {arr.dtype}"
    assert np.isfinite(wall) and wall > 0, \
        f"measure_throughput: non-positive wall time {wall}"
    frames = int(arr.shape[-1])
    channels = int(arr.shape[0]) if arr.ndim == 2 else 1
    samples_per_run = int(arr.size)
    sps = (samples_per_run * n_runs) / wall
    assert np.isfinite(sps) and sps > 0, \
        f"measure_throughput: insane throughput {sps}"
    return {
        "runs": int(n_runs),
        "frames_per_run": frames,
        "channels": channels,
        "samples_per_run": samples_per_run,
        "wall_seconds": wall,
        "seconds_per_run": wall / n_runs,
        "samples_per_second": float(sps),
    }
