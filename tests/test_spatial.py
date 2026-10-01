# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_spatial.py -- script-style tests for resonance.spatial.

Run:  python3 tests/test_spatial.py      (from the repo root)
   or python3 -m pytest tests/test_spatial.py

Style: each test prints "  ok: <name>"; the end prints
"N spatial tests passed." Any failure raises immediately.

Coverage contract (v0.2.0 group D):
  * ITD monotonic in azimuth; left/right delay sign flips at 0
  * ILD sign (far ear attenuated)
  * distance gain decreases with distance
  * ambisonic encode->decode round-trip identity for a centered source
  * orbit completes a full revolution (azimuth wraps 2*pi)
  * no-claim checks: no therapeutic language in the orbit docstrings
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.spatial import ambisonic, distance, hrtf, orbit

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


SR = 44100


def _impulse(n=SR):
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1.0
    return x


# -- HRTF / ITD -------------------------------------------------------------
def t_itd_zero_at_center():
    assert abs(float(hrtf.itd_seconds(0.0, 0.0))) < 1e-12


def t_itd_sign_flips_at_zero():
    # Source right (+az): right ear first -> tL - tR > 0.
    assert float(hrtf.itd_seconds(90.0, 0.0)) > 0.0
    assert float(hrtf.itd_seconds(-90.0, 0.0)) < 0.0
    # Antisymmetric: itd(-a) == -itd(a).
    for a in (15.0, 30.0, 45.0, 60.0, 90.0, 135.0):
        l = float(hrtf.itd_seconds(-a, 0.0))
        r = float(hrtf.itd_seconds(a, 0.0))
        assert abs(l + r) < 1e-12, (a, l, r)


def t_itd_monotonic_to_90():
    vals = [abs(float(hrtf.itd_seconds(a, 0.0)))
            for a in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)]
    for a, b in zip(vals, vals[1:]):
        assert b > a, (vals,)
    # Sanity scale: ~0.65 ms max for the default head.
    assert 0.0005 < vals[-1] < 0.0008, vals[-1]


def t_itd_pan_delay_side():
    # Impulse at +90 deg (hard right): right channel peaks first.
    stereo = hrtf.pan(_impulse(), 90.0, 0.0, sample_rate=SR)
    assert stereo.shape == (2, SR) and stereo.dtype == np.float32
    peak_l = int(np.argmax(np.abs(stereo[0])))
    peak_r = int(np.argmax(np.abs(stereo[1])))
    assert peak_r < peak_l, (peak_l, peak_r)
    # Mirror: impulse at -90 deg -> left channel peaks first.
    stereo = hrtf.pan(_impulse(), -90.0, 0.0, sample_rate=SR)
    peak_l = int(np.argmax(np.abs(stereo[0])))
    peak_r = int(np.argmax(np.abs(stereo[1])))
    assert peak_l < peak_r, (peak_l, peak_r)


def t_itd_pan_center_identical():
    stereo = hrtf.pan(_impulse(), 0.0, 0.0, sample_rate=SR)
    assert np.array_equal(stereo[0], stereo[1])


# -- HRTF / ILD -------------------------------------------------------------
def t_ild_zero_at_center_max_at_side():
    assert float(hrtf.ild_db(0.0, 0.0)) == 0.0
    side = float(hrtf.ild_db(90.0, 0.0))
    assert side > 0.0
    assert abs(side - float(hrtf.ild_db(-90.0, 0.0))) < 1e-12
    assert float(hrtf.ild_db(45.0, 0.0)) < side


def t_ild_pan_far_ear_quieter():
    # 1 kHz-ish tone burst; far ear must be quieter at +/-90.
    t = np.arange(SR, dtype=np.float64) / SR
    tone = (np.sin(2 * np.pi * 440.0 * t) *
            np.hanning(SR)).astype(np.float32)
    stereo = hrtf.pan(tone, 90.0, 0.0, sample_rate=SR)
    rms_l = float(np.sqrt(np.mean(stereo[0] ** 2)))
    rms_r = float(np.sqrt(np.mean(stereo[1] ** 2)))
    assert rms_r > rms_l, (rms_l, rms_r)
    stereo = hrtf.pan(tone, -90.0, 0.0, sample_rate=SR)
    rms_l = float(np.sqrt(np.mean(stereo[0] ** 2)))
    rms_r = float(np.sqrt(np.mean(stereo[1] ** 2)))
    assert rms_l > rms_r, (rms_l, rms_r)


def t_head_model_validation():
    for bad in (dict(head_radius_m=0.0), dict(head_radius_m=-1.0),
                dict(sample_rate=0), dict(ild_max_db=-1.0)):
        try:
            hrtf.HeadModel(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"HeadModel accepted {bad}")


# -- measured HRTF loader ----------------------------------------------------
def t_measured_loader_roundtrip():
    m, n = 4, 64
    az = np.array([0.0, 90.0, 180.0, -90.0])
    el = np.zeros(4)
    h = np.zeros((m, 2, n), dtype=np.float32)
    h[:, 0, 0] = 1.0
    h[:, 1, 1] = 0.5
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "hrtf.npz")
        np.savez(p, azimuths_deg=az, elevations_deg=el, hrirs=h,
                 sample_rate=SR)
        loaded = hrtf.load_hrtf_set(p)
    assert loaded.hrirs.shape == (m, 2, n)
    assert loaded.sample_rate == SR
    out = hrtf.pan_measured(_impulse(1024), 90.0, 0.0, loaded)
    assert out.shape == (2, 1024) and out.dtype == np.float32
    # Nearest grid point is az=90 -> hrir [1,0,...] left, [0,.5,...] right.
    assert out[0, 0] == np.float32(1.0)
    assert out[1, 1] == np.float32(0.5)


def t_measured_loader_loud_errors():
    with tempfile.TemporaryDirectory() as td:
        # Missing keys.
        p1 = str(Path(td) / "bad1.npz")
        np.savez(p1, azimuths_deg=np.zeros(2))
        try:
            hrtf.load_hrtf_set(p1)
        except ValueError as e:
            assert "missing required key" in str(e)
        else:
            raise AssertionError("missing keys not rejected")
        # Shape mismatch.
        p2 = str(Path(td) / "bad2.npz")
        np.savez(p2, azimuths_deg=np.zeros(3), elevations_deg=np.zeros(3),
                 hrirs=np.zeros((2, 2, 8), dtype=np.float32),
                 sample_rate=SR)
        try:
            hrtf.load_hrtf_set(p2)
        except ValueError as e:
            assert "M mismatch" in str(e)
        else:
            raise AssertionError("M mismatch not rejected")
        # Not even an NPZ.
        p3 = str(Path(td) / "bad3.txt")
        Path(p3).write_text("hello")
        try:
            hrtf.load_hrtf_set(p3)
        except ValueError as e:
            assert "cannot read" in str(e)
        else:
            raise AssertionError("non-NPZ not rejected")


# -- distance -----------------------------------------------------------------
def t_distance_gain_decreases():
    g1 = float(distance.gain(1.0))
    g2 = float(distance.gain(2.0))
    g10 = float(distance.gain(10.0))
    assert abs(g1 - 1.0) < 1e-12
    assert g1 > g2 > g10 > 0.0
    assert abs(g2 - 0.5) < 1e-12  # exact 1/r halving per doubling


def t_distance_near_clamp():
    assert float(distance.gain(0.0)) == float(distance.gain(0.05))
    assert np.isfinite(float(distance.gain(0.0)))
    try:
        distance.gain(-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative distance not rejected")


def t_distance_apply_shape_and_darkening():
    t = np.arange(SR, dtype=np.float64) / SR
    tone = np.sin(2 * np.pi * 8000.0 * t).astype(np.float32)
    near = distance.apply(tone, 1.0, sample_rate=SR)
    far = distance.apply(tone, 50.0, sample_rate=SR)
    assert near.shape == tone.shape and near.dtype == np.float32
    assert far.shape == tone.shape
    assert float(np.max(np.abs(far))) < float(np.max(np.abs(near)))
    # Stereo in -> stereo out.
    st = np.stack([tone, tone])
    out = distance.apply(st, 5.0, sample_rate=SR)
    assert out.shape == (2, SR) and out.dtype == np.float32


# -- ambisonic ------------------------------------------------------------------
def t_ambisonic_encode_shapes():
    b = ambisonic.encode(_impulse(512), 30.0, 10.0)
    assert b.shape == (4, 512) and b.dtype == np.float32


def t_ambisonic_center_encode():
    # Centered source: W = p/sqrt(2), X = p, Y = Z = 0 (AmbiX ACN/SN3D).
    b = ambisonic.encode(np.ones(8, dtype=np.float32), 0.0, 0.0)
    assert np.allclose(b[0], 1.0 / np.sqrt(2.0))
    assert np.allclose(b[1], 0.0, atol=1e-7)
    assert np.allclose(b[2], 0.0, atol=1e-7)
    assert np.allclose(b[3], 1.0)


def t_ambisonic_roundtrip_centered_source():
    # A source at the array CENTER has no direction: only W is
    # energized (X = Y = Z = 0). Decoding must then feed every speaker
    # of a symmetric array identically -- round-trip identity.
    mono = (np.sin(2 * np.pi * 440.0 *
                   np.arange(SR, dtype=np.float64) / SR)).astype(np.float32)
    b = np.zeros((4, SR), dtype=np.float32)
    b[0] = mono / np.sqrt(2.0)  # W-only, SN3D
    feeds = ambisonic.decode(b, [0.0, 90.0, 180.0, -90.0])
    assert feeds.shape == (4, SR) and feeds.dtype == np.float32
    for i in (1, 2, 3):
        assert np.allclose(feeds[i], feeds[0], rtol=1e-5, atol=1e-7), i
    assert float(np.max(np.abs(feeds[0]))) > 0.0
    assert np.allclose(feeds[0], mono / 2.0, rtol=1e-5, atol=1e-6)
    # Directionality preserved: a FRONT source (az 0) puts more energy
    # on the front speaker than the rear one.
    b2 = ambisonic.encode(mono, 0.0, 0.0)
    f2 = ambisonic.decode(b2, [0.0, 180.0])
    e_front = float(np.mean(f2[0] ** 2))
    e_back = float(np.mean(f2[1] ** 2))
    assert e_front > e_back, (e_front, e_back)


def t_ambisonic_validation():
    try:
        ambisonic.encode(np.zeros((2, 8), dtype=np.float32), 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-mono encode not rejected")
    try:
        ambisonic.decode(np.zeros((3, 8), dtype=np.float32), [0.0])
    except ValueError:
        pass
    else:
        raise AssertionError("non-4ch decode not rejected")
    try:
        ambisonic.decode(np.zeros((4, 8), dtype=np.float32), [])
    except ValueError:
        pass
    else:
        raise AssertionError("empty speaker list not rejected")


# -- orbit ----------------------------------------------------------------------
def t_orbit_full_revolution():
    # 0.25 Hz over 4 s = exactly one revolution: the unwrapped
    # trajectory must sweep 2*pi and end where it started.
    traj = orbit.trajectory(0.25, 4.0, n_points=4097)
    unwrapped = np.unwrap(np.deg2rad(traj))
    swept = unwrapped[-1] - unwrapped[0]
    assert abs(abs(swept) - 2 * np.pi) < 1e-9, swept
    assert abs(traj[-1] - traj[0]) < 1e-9


def t_orbit_output_shape_and_motion():
    from resonance.binaural.generator import binaural_beat
    # A real entrainment array from resonance.binaural, orbited.
    sess = binaural_beat(10.0, carrier=440, duration=4.0, sample_rate=SR)
    assert sess.shape == (2, 4 * SR)
    out = orbit.orbit(sess, 0.25, 4.0, sample_rate=SR)
    assert out.shape == (2, 4 * SR) and out.dtype == np.float32
    # Motion check: the L/R energy balance in the first half-second
    # must differ from the balance in the third half-second block --
    # the source has moved to the other side.
    def balance(seg):
        l = float(np.mean(seg[0] ** 2)) + 1e-12
        r = float(np.mean(seg[1] ** 2)) + 1e-12
        return l / r
    b0 = balance(out[:, 0:SR // 2])
    b2 = balance(out[:, SR:3 * SR // 2])
    assert abs(b0 - b2) > 0.02, (b0, b2)


def t_orbit_mono_and_validation():
    out = orbit.orbit(_impulse(SR), 0.5, 1.0, sample_rate=SR)
    assert out.shape == (2, SR) and out.dtype == np.float32
    for bad in (dict(angular_rate_hz=0.0), dict(angular_rate_hz=-1.0)):
        try:
            orbit.orbit(_impulse(64), duration_s=1.0, sample_rate=SR, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"orbit accepted {bad}")
    try:
        orbit.orbit(np.zeros((3, 64), dtype=np.float32), 0.5, 1.0,
                    sample_rate=SR)
    except ValueError:
        pass
    else:
        raise AssertionError("non (N,)/(2,N) orbit input not rejected")


# -- no-claim docstring checks ----------------------------------------------------
def t_no_therapeutic_claims():
    for mod in (orbit, hrtf, distance, ambisonic):
        doc = (mod.__doc__ or "")
        assert "therap" not in doc.lower(), mod.__name__
    # The orbit function docstring itself must say creative; no health-claim language.
    doc = orbit.orbit.__doc__ or ""
    assert "therap" not in doc.lower()
    assert "creative" in doc.lower()


# -- cli --------------------------------------------------------------------------
def t_cli_help_and_pan(tmp_wav=None):
    from resonance.spatial import cli
    import io as _io
    # --help exits 0 via SystemExit code 0.
    try:
        cli.main(["--help"])
    except SystemExit as e:
        assert e.code == 0, e.code
    else:
        raise AssertionError("--help did not exit")
    # Pan a real mono WAV end to end.
    from resonance.core.io import read_wav, write_wav
    with tempfile.TemporaryDirectory() as td:
        inp = str(Path(td) / "in.wav")
        outp = str(Path(td) / "out.wav")
        write_wav(inp, _impulse(4096), SR)
        rc = cli.main(["pan", inp, "-o", outp, "--azimuth", "45"])
        assert rc == 0
        audio, sr = read_wav(outp)
        assert audio.shape == (2, 4096) and sr == SR


TESTS = [
    ("itd zero at center", t_itd_zero_at_center),
    ("itd sign flips at zero", t_itd_sign_flips_at_zero),
    ("itd monotonic to 90 deg", t_itd_monotonic_to_90),
    ("itd pan delay side", t_itd_pan_delay_side),
    ("itd pan center identical", t_itd_pan_center_identical),
    ("ild zero at center, max at side", t_ild_zero_at_center_max_at_side),
    ("ild pan far ear quieter", t_ild_pan_far_ear_quieter),
    ("head model validation", t_head_model_validation),
    ("measured loader roundtrip", t_measured_loader_roundtrip),
    ("measured loader loud errors", t_measured_loader_loud_errors),
    ("distance gain decreases", t_distance_gain_decreases),
    ("distance near clamp", t_distance_near_clamp),
    ("distance apply shape/darkening", t_distance_apply_shape_and_darkening),
    ("ambisonic encode shapes", t_ambisonic_encode_shapes),
    ("ambisonic center encode", t_ambisonic_center_encode),
    ("ambisonic roundtrip centered source", t_ambisonic_roundtrip_centered_source),
    ("ambisonic validation", t_ambisonic_validation),
    ("orbit full revolution", t_orbit_full_revolution),
    ("orbit output shape and motion", t_orbit_output_shape_and_motion),
    ("orbit mono and validation", t_orbit_mono_and_validation),
    ("no therapeutic claims", t_no_therapeutic_claims),
    ("cli help and pan", t_cli_help_and_pan),
]


def main():
    for name, fn in TESTS:
        check(name, fn)
    print(f"{PASSED} spatial tests passed.")


if __name__ == "__main__":
    main()
