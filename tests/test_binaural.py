# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_binaural.py -- script-style tests for resonance.binaural.

Run:  python3 tests/test_binaural.py      (from the repo root)
   or python3 -m pytest tests/test_binaural.py

Style: each test prints "  ok: <name>"; the end prints
"<N> binaural tests passed." Any failure raises immediately.

Honesty: the adaptive BPM test is labeled a HEURISTIC check -- it runs on
a synthesized metronomic click track, the one signal the heuristic is
actually designed for. No therapeutic claims anywhere.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.binaural import adaptive, generator, session

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


SR = 44100


def _peak_hz(channel, sr):
    """Independent FFT peak estimate (parabolic interpolation)."""
    x = np.asarray(channel, dtype=np.float64)
    n = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    k = int(np.argmax(spec))
    if 0 < k < len(spec) - 1:
        y0, y1, y2 = spec[k - 1], spec[k], spec[k + 1]
        d = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / d if d != 0 else 0.0
    else:
        shift = 0.0
    return (k + shift) * sr / n


# -- generator --------------------------------------------------------------
def t_binaural_alpha_on_528():
    # 10 Hz beat on a 528 Hz carrier: L must peak ~523, R ~533.
    stereo = generator.binaural_beat(10.0, carrier=528, duration=6.0,
                                     sample_rate=SR)
    assert stereo.shape == (2, 6 * SR) and stereo.dtype == np.float32
    pl = _peak_hz(stereo[0], SR)
    pr = _peak_hz(stereo[1], SR)
    assert abs(pl - 523.0) <= 0.5, f"L peak {pl}"
    assert abs(pr - 533.0) <= 0.5, f"R peak {pr}"
    assert abs((pr - pl) - 10.0) <= 0.5, f"beat {pr - pl}"
check("t_binaural_alpha_on_528", t_binaural_alpha_on_528)


def t_binaural_bands_table():
    assert generator.BANDS["alpha"] == (8.0, 13.0)
    assert generator.BANDS["delta"] == (0.5, 4.0)
    assert generator.band_for(10.0) == "alpha"
    assert generator.band_for(6.0) == "theta"
    assert generator.band_for(200.0) is None
check("t_binaural_bands_table", t_binaural_bands_table)


def t_binaural_rejects_bad_beat():
    for beat in (0.0, -5.0, 500.0):  # 500 Hz beat >= 440 carrier
        try:
            generator.binaural_beat(beat, carrier=440.0, duration=0.1)
        except ValueError:
            continue
        raise AssertionError(f"bad beat accepted: {beat}")
check("t_binaural_rejects_bad_beat", t_binaural_rejects_bad_beat)


def t_monaural_beat_renders():
    m = generator.monaural_beat(10.0, carrier=440.0, duration=2.0,
                                sample_rate=SR)
    assert m.shape == (2, 2 * SR) and np.all(np.isfinite(m))
    # Dual mono: identical channels.
    assert np.array_equal(m[0], m[1])
    # The beat exists acoustically: envelope of |signal| pulses at 10 Hz.
    # (Smooth the envelope first: raw |sin| rectifies the 440 Hz carrier
    # and its 880 Hz harmonic would dominate the spectrum. A 2000-sample
    # moving average nulls ~880 Hz while passing 10 Hz.)
    env = np.abs(m[0].astype(np.float64))
    env = np.convolve(env, np.ones(2000) / 2000.0, mode="same")
    spec = np.abs(np.fft.rfft(env - env.mean()))
    freqs = np.fft.rfftfreq(len(env), 1.0 / SR)
    peak = freqs[int(np.argmax(spec[1:])) + 1]
    assert abs(peak - 10.0) < 0.5, f"envelope peak {peak}"
check("t_monaural_beat_renders", t_monaural_beat_renders)


def t_isochronic_renders():
    iso = generator.isochronic_tone(10.0, carrier=440.0, duration=2.0,
                                    sample_rate=SR)
    assert iso.shape == (2, 2 * SR) and np.all(np.isfinite(iso))
    assert np.array_equal(iso[0], iso[1])
    # Gated: there must be near-silent gaps between pulses.
    assert np.min(np.abs(iso[0][:SR])) < 0.01
    try:
        generator.isochronic_tone(10.0, duty=1.5, duration=0.1)
    except ValueError:
        return
    raise AssertionError("bad duty accepted")
check("t_isochronic_renders", t_isochronic_renders)


def t_carrier_note_name():
    s = generator.binaural_beat(10.0, carrier="A4", duration=1.0,
                                 sample_rate=SR)
    pl = _peak_hz(s[0], SR)
    assert abs(pl - 435.0) <= 0.5, f"L peak {pl}"  # 440 - 5
check("t_carrier_note_name", t_carrier_note_name)


# -- session -----------------------------------------------------------------
def t_session_length_and_boundaries():
    sess = session.Session([(10.0, 0.1), ("theta", 0.1)], carrier=528,
                           sample_rate=SR, crossfade_seconds=1.0)
    assert sess.total_minutes == 0.2
    xf = SR  # 1.0 s crossfade
    n0 = n1 = int(round(SR * 0.1 * 60.0))
    bounds = sess.segment_boundaries()
    assert bounds == [(0, n0), (n0 - xf, n0 - xf + n1)], bounds
    out = sess.render()
    assert out.shape == (2, n0 + n1 - xf), out.shape
    assert np.all(np.isfinite(out))
check("t_session_length_and_boundaries", t_session_length_and_boundaries)


def t_session_single_segment_no_crossfade():
    sess = session.Session([(10.0, 0.05)], carrier=440.0, sample_rate=SR)
    out = sess.render()
    expect = generator.binaural_beat(10.0, carrier=440.0, duration=3.0,
                                     sample_rate=SR, amplitude=0.5)
    assert out.shape == expect.shape
    assert np.array_equal(out, expect)  # no crossfade: bit-identical
check("t_session_single_segment_no_crossfade", t_session_single_segment_no_crossfade)


def t_session_rejects_bad_band():
    try:
        session.Session([("ultraviolet", 5)], carrier=440.0)
    except ValueError:
        return
    raise AssertionError("bad band accepted")
check("t_session_rejects_bad_band", t_session_rejects_bad_band)


# -- adaptive (HEURISTIC) ------------------------------------------------------
def _click_track(bpm, seconds, sr=SR):
    """Synthesized metronomic click track: the heuristic's home turf."""
    n = int(sr * seconds)
    out = np.zeros(n, dtype=np.float32)
    period = int(round(sr * 60.0 / bpm))
    # A click: short exponentially-decaying burst, 5 ms long.
    decay = np.exp(-np.arange(220, dtype=np.float32) / 40.0)
    for start in range(0, n - 220, period):
        out[start:start + 220] += decay
    peak = np.max(np.abs(out))
    return (out / peak * 0.9).astype(np.float32)


def t_adaptive_bpm_heuristic_on_click_track():
    # HEURISTIC check on a synthesized 120 BPM click track.
    clicks = _click_track(120.0, 12.0)
    bpm = adaptive.estimate_bpm(clicks, SR)
    assert abs(bpm - 120.0) <= 2.0, f"detected {bpm}"
check("t_adaptive_bpm_heuristic_on_click_track",
      t_adaptive_bpm_heuristic_on_click_track)


def t_adaptive_render_follows_detected_bpm():
    # HEURISTIC: the rendered session pulses at detected_bpm / 60 Hz.
    clicks = _click_track(120.0, 12.0)
    rendered, bpm = adaptive.render_following(clicks, duration=4.0,
                                              carrier=528, sample_rate=SR)
    assert abs(bpm - 120.0) <= 2.0, f"detected {bpm}"
    assert rendered.shape == (2, 4 * SR)
    # The isochronic envelope must pulse at ~2 Hz (120/60).
    env = np.abs(rendered[0].astype(np.float64))
    spec = np.abs(np.fft.rfft(env - env.mean()))
    freqs = np.fft.rfftfreq(len(env), 1.0 / SR)
    peak = freqs[int(np.argmax(spec[1:])) + 1]
    assert abs(peak - bpm / 60.0) < 0.25, f"pulse peak {peak}"
check("t_adaptive_render_follows_detected_bpm",
      t_adaptive_render_follows_detected_bpm)


def t_adaptive_rejects_non_solfeggio_carrier():
    clicks = _click_track(120.0, 4.0)
    try:
        adaptive.render_following(clicks, duration=1.0, carrier=440,
                                  sample_rate=SR)
    except ValueError:
        return
    raise AssertionError("non-Solfeggio carrier accepted")
check("t_adaptive_rejects_non_solfeggio_carrier",
      t_adaptive_rejects_non_solfeggio_carrier)


print(f"\n{PASSED} binaural tests passed.")
