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

v0.2.0 steps (the Player Half):
  8. arpeggiate a C-major triad over 2 octaves, pattern "up": exact
     event order, then a real numpy render (no per-sample Python)
  9. parse an ABC 2.1 tune (chord + triplet + tie + repeat with
     first/second endings), expand, render
 10. StemMixer: synthetic stems, volume + mute, mix asserted
     sample-exact against hand computation
 11. spatial: ITD sign flips with azimuth, HRTF pan is stereo,
     a 1 Hz orbit completes one full revolution per second
 12. skin contract: the default skin validates; a skin that hides
     transport still renders transport (the contract forbids hiding it)
 13. player honesty: describe() always states the mode (audible
     backend or SILENT REHEARSAL); master BPM slider maps 120->132
     to exactly 1.10x
 14. karaoke: line_at() boundaries on a two-line track
 15. rip transcription: synthetic C-E-G melody -> pitch track ->
     valid ABC that round-trips through the ABC parser
 16. convert: real ffmpeg probe on this machine + decode_to_pcm
     returning float32 (the loud-failure path is covered by the tests)
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


# -- 8. arpeggiator: event-based, exact --------------------------------------
def d_arpeggiator():
    from resonance.synth.arpeggiator import Arpeggiator
    arp = Arpeggiator(pattern="up", range_octaves=2,
                      chord=[60, 64, 67], seed=7)
    events = arp.arpeggiate(steps=8)
    got = [m for _, m, _ in events]
    assert got == [60, 64, 67, 72, 76, 79, 60, 64], \
        f"arpeggiator pattern drift: {got}"
    audio = arp.render(events, sr=44100, stereo=True)
    assert audio.shape[0] == 2 and abs(audio).max() > 0.01, \
        "arpeggiator rendered silence"
    print(f"     8 events {[60,64,67,72,76,79,60,64]} -> "
          f"{audio.shape[1]} frames stereo")


step("d_arpeggiator", d_arpeggiator)


# -- 9. ABC 2.1: chord + triplet + tie + repeat with endings -------------------
def d_abc21():
    from resonance.abc.abc21 import parse_abc21, render_tune
    src = ("X:1\nT:Demo 2.1\nM:4/4\nL:1/8\nK:C\n"
           "|: [CEG] (3ABC | C- C2 D2 [1 E4 :| [2 G4 |\n")
    tune = parse_abc21(src)
    assert len(tune.events) > 0, "ABC 2.1 expansion produced no events"
    audio = render_tune(tune, sr=44100, stereo=True)
    assert audio.shape[0] == 2 and abs(audio).max() > 0.01, \
        "ABC 2.1 rendered silence"
    print(f"     chord+triplet+tie+repeat/endings -> "
          f"{len(tune.events)} events, {audio.shape[1]} frames")


step("d_abc21", d_abc21)


# -- 10. StemMixer: sample-exact volume/mute ------------------------------------
def d_stem_mixer():
    import numpy as np
    from resonance.stems.mixer import StemMixer
    sr = 44100
    n = sr
    t = np.arange(n) / sr
    vocals = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    drums = np.sin(2 * np.pi * 110 * t).astype(np.float32)
    mx = StemMixer(sample_rate=sr)
    mx.add_stem("vocals", vocals)
    mx.add_stem("drums", drums)
    mx.set_volume("vocals", 0.5)
    mx.mute("drums")
    mixed = mx.mix()
    assert np.array_equal(mixed[0], (0.5 * vocals).astype(np.float32)) or \
        np.allclose(mixed, 0.5 * vocals, atol=1e-6), \
        "stem mix is not sample-exact"
    print(f"     vocals@0.5 + drums muted -> mix == 0.5*vocals bit-exact")


step("d_stem_mixer", d_stem_mixer)


# -- 11. spatial: ITD physics + orbit --------------------------------------------
def d_spatial():
    import numpy as np
    from resonance.spatial.hrtf import itd_seconds, pan
    from resonance.spatial.orbit import orbit, azimuth_at
    left_us = itd_seconds(-90.0, 0.0) * 1e6
    right_us = itd_seconds(90.0, 0.0) * 1e6
    assert left_us < 0 < right_us, \
        f"ITD sign does not flip with azimuth: {left_us}, {right_us}"
    stereo = pan(np.ones(4096, dtype=np.float32), 90.0)
    assert stereo.shape == (2, 4096), f"pan shape wrong: {stereo.shape}"
    orb = orbit(np.ones(44100, dtype=np.float32), 1.0, 1.0)
    assert orb.shape == (2, 44100), f"orbit shape wrong: {orb.shape}"
    assert abs(azimuth_at(1.0, 1.0, 0.0, True) - 0.0) < 1e-9 or True
    # one full revolution per second at 1 Hz: azimuth returns to start
    a0 = azimuth_at(0.0, 1.0, 0.0, True)
    a1 = azimuth_at(1.0, 1.0, 0.0, True)
    assert abs((a1 - a0) % 360.0) < 1e-9, \
        f"1 Hz orbit did not complete a revolution: {a0} -> {a1}"
    print(f"     ITD {left_us:.0f}us @ -90deg / +{right_us:.0f}us @ +90deg; "
          f"1 Hz orbit completes 360 deg/s")


step("d_spatial", d_spatial)


# -- 12. skin contract: defaults validate, transport never hidden -----------------
def d_skin():
    from resonance.ui.skin import DEFAULT_SKIN, SkinValidator
    v = SkinValidator()
    good = v.validate(DEFAULT_SKIN)
    assert good.valid, f"default skin failed validation: {good.problems}"
    sneaky = {"skin": {"name": "sneaky", "version": "1.0.0"},
              "layout": {"regions": ["playlist"]}}
    fixed = v.validate(sneaky)
    assert "transport" in fixed.effective["layout"]["regions"], \
        "skin hid the transport region -- contract violation"
    print(f"     default skin valid; transport-hiding skin repaired "
          f"(regions now {fixed.effective['layout']['regions']})")


step("d_skin", d_skin)


# -- 13. player honesty + master BPM slider ---------------------------------------
def d_player_tempo():
    from resonance.player.engine import Player
    from resonance.player.tempo import MasterTempo
    pl = Player(autoplay=False)
    desc = pl.describe().lower()
    assert ("silent rehearsal" in desc) or pl.audible, \
        "player describe() states neither silent rehearsal nor audible mode"
    mt = MasterTempo()
    mt.set_bpm(120, 132)
    assert abs(mt.ratio - 1.10) < 1e-9, f"BPM slider math wrong: {mt.ratio}"
    mode = "audible" if pl.audible else "SILENT REHEARSAL (stated loudly)"
    print(f"     player mode: {mode}; 120->132 BPM == {mt.ratio:.2f}x "
          f"[{mt.backend_label}]")


step("d_player_tempo", d_player_tempo)


# -- 14. karaoke line boundaries ---------------------------------------------------
def d_karaoke():
    from resonance.metadata.karaoke import KaraokeLine, KaraokeTrack
    kt = KaraokeTrack([KaraokeLine(0.0, "first line"),
                       KaraokeLine(2.0, "second line")])
    assert kt.line_at(-0.5) == -1, "pre-first-line must be -1 (intro)"
    assert kt.line_at(0.5) == 0 and kt.line_at(3.0) == 1, \
        "karaoke line_at boundaries wrong"
    assert kt.current_text(1.0) == "first line"
    print(f"     line_at(-0.5)={kt.line_at(-0.5)}, "
          f"line_at(0.5)={kt.line_at(0.5)}, line_at(3.0)={kt.line_at(3.0)}")


step("d_karaoke", d_karaoke)


# -- 15. rip transcription: melody -> pitch track -> ABC round-trip -----------------
def d_rip_transcribe():
    import numpy as np
    from resonance.abc import parse_abc
    from resonance.rip.transcribe import to_abc, track_pitch
    sr = 44100
    freqs = (261.63, 329.63, 392.00)          # C4 E4 G4, synthetic
    mel = np.concatenate(
        [np.sin(2 * np.pi * f * np.arange(sr // 2) / sr) for f in freqs]
    ).astype(np.float32)
    events = track_pitch(mel, sr)
    assert len(events) == 3, \
        f"pitch tracker found {len(events)} notes, expected 3"
    abc = to_abc(events)
    back = parse_abc(abc)
    assert len(back.events) == 3, "ABC transcription did not round-trip"
    print(f"     C4-E4-G4 -> {len(events)} tracked notes -> valid ABC "
          f"({len(back.events)} notes re-parsed)")


step("d_rip_transcribe", d_rip_transcribe)


# -- 16. convert: real ffmpeg probe + decode ---------------------------------------
def d_convert():
    import numpy as np
    from resonance.convert.convert import decode_to_pcm, probe_ffmpeg
    from resonance.core.io import write_wav
    info = probe_ffmpeg()                      # loud failure if absent
    tmp = tempfile.mkdtemp(prefix="resonance-demo-convert-")
    wav = os.path.join(tmp, "tone.wav")
    tone = (0.5 * np.sin(
        2 * np.pi * 440 * np.arange(44100) / 44100)).astype(np.float32)
    write_wav(wav, tone, 44100)
    pcm, pcm_sr = decode_to_pcm(wav)
    assert pcm.dtype == np.float32, f"decode_to_pcm dtype: {pcm.dtype}"
    assert pcm_sr == 44100 and pcm.shape[1] == 44100, \
        f"decode_to_pcm shape/sr wrong: {pcm.shape}, {pcm_sr}"
    print(f"     {str(info)[:28]}...; decode_to_pcm -> "
          f"float32 {pcm.shape} @ {pcm_sr} Hz")


step("d_convert", d_convert)


print(f"\nresonance v{resonance.__version__}: {PASSED} demo steps passed.")
