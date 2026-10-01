# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
demo/resonance_demo.py -- end-to-end RESONANCE v0.1.0 demonstration.

Run:  python3 demo/resonance_demo.py        (from the repo root)

Script-style (matching tests/test_telemetry.py): each step prints
"  ok: <name>"; any failure raises immediately with a clear message
and the process exits NONZERO. Nothing is swallowed: a failed step
kills the demo, because a demo that lies about success is worse than
no demo at all.

Steps (every one exercises a real claim the README makes):
  1. render a 10 Hz alpha binaural tone on a 528 Hz carrier, then
     FFT-verify the render with diagnostics.verify_binaural
  2. render an 808 pattern (kick/snare/closed_hat) to a buffer
  3. render the built-in ABC tune "The North Gate" to a WAV file
  4. render mandala frames for the beat phase and assert each frame's
     rotation == 2*pi*phase exactly (the viz timeline contract)
  5. load the sample plugin (demo/plugins/echo_plugin.py) through the
     PluginManager and fire its hooks
  6. submit a render job through the API (FastAPI TestClient when
     fastapi is importable; a printed skip line otherwise -- the demo
     never fails for a missing optional dependency)
  7. print real benchmark numbers (perf_counter, not stubs)

Honesty: no medical or therapeutic claims appear here. Adaptive BPM
is not exercised because it is a labeled heuristic, and this demo
makes no claim about it.
"""
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEMO_DIR = Path(__file__).resolve().parent

import resonance  # noqa: E402
from resonance.binaural.generator import binaural_beat  # noqa: E402
from resonance.diagnostics import measure, verify  # noqa: E402
from resonance.plugins import PluginManager  # noqa: E402
from resonance.viz.engine import TAU, phase_at, render_rosette_frame  # noqa: E402

PASSED = 0


def step(name, fn):
    """Run one demo step; print ok or raise loudly and exit nonzero."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


# -- 1. binaural render + FFT verification --------------------------------
def d_binaural_verified():
    audio = binaural_beat(10.0, carrier=528, duration=4.0, sample_rate=44100)
    assert audio.shape == (2, 4 * 44100), f"unexpected shape {audio.shape}"
    verdict = verify.verify_binaural(audio, 10.0, 528, sample_rate=44100,
                                     tol_hz=0.5)
    assert verdict["passed"], f"FFT verification failed: {verdict}"
    print(f"     L peak {verdict['peak_l_hz']:.2f} Hz, R peak "
          f"{verdict['peak_r_hz']:.2f} Hz, beat measured "
          f"{verdict['beat_measured_hz']:.2f} Hz (target 10.00 Hz)")


step("d_binaural_verified", d_binaural_verified)


# -- 2. 808 pattern ----------------------------------------------------------
def d_pattern():
    from resonance.synth import render_pattern
    kick = [0] * 16
    kick[0] = kick[8] = 1.0                      # four on the floor-ish
    kick[4] = kick[12] = 1.0
    snare = [0] * 16
    snare[4] = snare[12] = 1.0                   # backbeat
    hat = [1.0] * 16                             # driving 16ths
    audio = render_pattern(
        {"kick": kick, "snare": snare, "closed_hat": hat},
        bpm=128.0, bars=1, swing=0.0, sr=44100,
    )
    assert audio.ndim == 2 and audio.shape[0] == 2, \
        f"pattern must be stereo (2, N), got {audio.shape}"
    assert audio.shape[1] > 0 and (abs(audio).max() > 0.01), \
        "pattern rendered silence"
    print(f"     {audio.shape[1]} frames stereo, peak {abs(audio).max():.3f}")


step("d_pattern", d_pattern)


# -- 3. ABC built-in tune -> WAV ---------------------------------------------
def d_abc_tune():
    from resonance.abc import parse_abc, render_tune
    from resonance.abc.tunes import BUILTIN_TUNE
    from resonance.core.io import read_wav
    tune = parse_abc(BUILTIN_TUNE)
    assert tune.headers.get("T") == "The North Gate", \
        f"title drift: {tune.headers.get('T')!r}"
    tmp = tempfile.mkdtemp(prefix="resonance-demo-abc-")
    wav = os.path.join(tmp, "north-gate.wav")
    audio = render_tune(tune, path=wav, sr=44100, stereo=True)
    assert audio.shape[0] == 2, f"ABC render must be stereo, got {audio.shape}"
    back, back_sr = read_wav(wav)
    assert back.shape == audio.shape, "WAV round-trip changed the buffer"
    assert back_sr == 44100, f"WAV round-trip sample rate drift: {back_sr}"
    print(f"     '{tune.headers['T']}' -> {wav} ({back.shape[1]} frames)")


step("d_abc_tune", d_abc_tune)


# -- 4. mandala frames: rotation == 2*pi*phase --------------------------------
def d_mandala_phases():
    beat = 10.0
    for t in (0.0, 0.025, 0.05, 0.075):
        phase = phase_at(beat, t)
        svg = render_rosette_frame(phase, petals=12, rings=3, beat_hz=beat)
        m = re.search(r'data-rotation-rad="([^"]+)"', svg)
        assert m, "frame missing data-rotation-rad stamp"
        rot = float(m.group(1))
        expected = TAU * phase
        assert rot == expected, \
            f"t={t}: frame rotation {rot!r} != 2*pi*phase {expected!r}"
        assert phase == (beat * t) % 1.0
    print("     4 frames: rotation == 2*pi*phase bit-for-bit (TAU = 2*pi)")


step("d_mandala_phases", d_mandala_phases)


# -- 5. sample plugin through the manager --------------------------------------
def d_plugin():
    mgr = PluginManager()
    found = mgr.discover(str(DEMO_DIR / "plugins"))
    assert "echo_plugin" in found, f"echo_plugin not discovered: {found}"
    meta = mgr.load("echo_plugin")
    assert meta["name"] == "echo", f"plugin name drift: {meta['name']!r}"
    replies = mgr.fire("ping", "hello")
    assert replies == ["pong:hello"], f"ping hook misfired: {replies}"
    doubled = mgr.fire("transform", 21)
    assert doubled == [42], f"transform hook misfired: {doubled}"
    print(f"     ping -> {replies[0]!r}, transform(21) -> {doubled[0]}")
    mgr.unload("echo_plugin")


step("d_plugin", d_plugin)


# -- 6. API job lifecycle ------------------------------------------------------
def d_api_job():
    from resonance.api.service import fastapi_available
    if not fastapi_available():
        print("     skip: fastapi not installed (optional dependency)")
        return
    from fastapi.testclient import TestClient
    from resonance.api.service import create_app

    app = create_app(jobs_dir=tempfile.mkdtemp(prefix="resonance-demo-jobs-"))
    client = TestClient(app)
    headers = {"Authorization": "Bearer dev-token-001"}

    health = client.get("/health").json()
    assert health["ok"] and health["resonance"] == resonance.__version__

    # Bad token is rejected: the prototype auth gate really gates.
    bad = client.post("/jobs/render", json={"kind": "binaural", "params": {}})
    assert bad.status_code == 401, f"bad token accepted: {bad.status_code}"

    # Submit a short binaural job, poll until done, download the WAV.
    sub = client.post(
        "/jobs/render",
        json={"kind": "binaural",
              "params": {"beat_hz": 10.0, "carrier": 440.0,
                         "duration": 2.0, "sample_rate": 44100}},
        headers=headers,
    ).json()
    job_id = sub["job_id"]
    assert sub["status"] == "rendering"
    deadline = time.time() + 60
    info = {}
    while time.time() < deadline:
        info = client.get(f"/jobs/{job_id}", headers=headers).json()
        if info["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert info["status"] == "done", f"job failed: {info}"
    assert info["wav_bytes"] and info["wav_bytes"] > 44
    dl = client.get(f"/jobs/{job_id}/download", headers=headers)
    assert dl.status_code == 200 and dl.content[:4] == b"RIFF", \
        "download did not return a valid WAV"
    print(f"     job {job_id[:8]}... rendered {info['wav_bytes']} bytes, "
          f"downloaded as RIFF/WAV")


step("d_api_job", d_api_job)


# -- 7. real benchmark numbers -------------------------------------------------
def d_benchmark():
    result = measure.measure_throughput(
        binaural_beat, 10.0,
        n_runs=1, carrier=440.0, duration=2.0, sample_rate=44100,
    )
    sps = result["samples_per_second"]
    assert sps > 0 and math.isfinite(sps)
    realtime = sps / (2 * 44100)
    print(f"     {sps:,.0f} samples/s ({realtime:.1f}x realtime, "
          f"{result['wall_seconds']*1000:.1f} ms wall for 2 s stereo)")


step("d_benchmark", d_benchmark)


print(f"\nresonance v{resonance.__version__}: {PASSED} demo steps passed.")
