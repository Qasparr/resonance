# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_viz.py -- script-style tests for resonance.viz.

Run:  python3 tests/test_viz.py      (from the repo root)
   or python3 -m pytest tests/test_viz.py

Style (matching bleat/trvvth): each test prints "  ok: <name>"; the end
prints "<N> viz tests passed." Any failure raises immediately -- the
first red line is the diagnosis.

Covers: the rotation-angle contract (2*pi*phase, parsed back out of the
SVG transform), the phase timeline, SVG well-formedness, petal/ring
counts, the petal "breath" pulse, band palettes, and the graceful
PNG/MP4 degradation paths (real files when matplotlib/ffmpeg are
present, None-without-exception when they are not).
"""
import math
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.viz.engine import (
    BREATH_AMPLITUDE,
    TAU,
    band_for,
    check_svg_wellformed,
    phase_at,
    render_rosette_frame,
    rosette_geometry,
    session_frames,
)
from resonance.viz import render as render_mod
from resonance.viz.render import render_mp4, render_png

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


def parse_svg(svg):
    """Parse SVG text; return (root_element, rotating_group)."""
    root = ET.fromstring(svg)
    ns = {"s": "http://www.w3.org/2000/svg"}
    group = root.find("s:g", ns)
    assert group is not None, "no rotating <g> found in SVG"
    return root, group


# -- the rotation contract -----------------------------------------------
def t_rotation_angle_equals_two_pi_phase():
    # THE core contract: the whole rosette's rotation angle is exactly
    # 2*pi*phase. The exact radian value is parsed back out of the
    # SVG's data-rotation-rad attribute (repr round-trips bit-for-bit).
    phase = 0.25
    svg = render_rosette_frame(phase, petals=8, rings=2)
    _, group = parse_svg(svg)
    back = float(group.get("data-rotation-rad"))
    assert back == TAU * phase, (back, TAU * phase)  # exact, not approx
    # And the SVG rotate() transform agrees in degrees: 360 * phase.
    m = re.search(r"rotate\(([-\d.eE+]+)", group.get("transform"))
    assert m, "transform has no rotate()"
    assert math.isclose(float(m.group(1)), 360.0 * phase, rel_tol=1e-6)
check("t_rotation_angle_equals_two_pi_phase",
      t_rotation_angle_equals_two_pi_phase)


def t_rotation_zero_phase_is_identity():
    svg = render_rosette_frame(0.0, petals=6, rings=1)
    _, group = parse_svg(svg)
    assert float(group.get("data-rotation-rad")) == 0.0
    assert float(group.get("data-phase")) == 0.0
check("t_rotation_zero_phase_is_identity", t_rotation_zero_phase_is_identity)


def t_rotation_deterministic():
    # Same inputs -> byte-identical SVG. A session recorder depends on
    # this: any frame can be re-rendered later and match exactly.
    a = render_rosette_frame(0.37, petals=12, rings=3)
    b = render_rosette_frame(0.37, petals=12, rings=3)
    assert a == b
check("t_rotation_deterministic", t_rotation_deterministic)


# -- the phase timeline ---------------------------------------------------
def t_phase_at_quarter_second_ten_hz():
    # The contract's worked example: 10 Hz beat, t = 0.25 s -> phase 0.5.
    assert phase_at(10.0, 0.25) == 0.5
    frames = session_frames(10.0, duration=1.0, fps=4)
    assert len(frames) == 4
    t, phase, _svg = frames[1]
    assert t == 0.25
    assert phase == 0.5
check("t_phase_at_quarter_second_ten_hz", t_phase_at_quarter_second_ten_hz)


def t_session_timeline_shape():
    frames = session_frames(10.0, duration=2.0, fps=4)
    assert len(frames) == 8  # int(2.0 * 4)
    for i, (t, phase, svg) in enumerate(frames):
        assert t == i / 4.0
        assert phase == (10.0 * t) % 1.0  # the contract, verbatim
        assert 0.0 <= phase < 1.0
        assert svg.startswith("<svg")
    # t = 0 is the downbeat: phase 0.
    assert frames[0][1] == 0.0
check("t_session_timeline_shape", t_session_timeline_shape)


def t_phase_wraps():
    # 10 Hz at t = 1.05 s is 10.5 cycles -> phase 0.5 (wraps, never 1.0+).
    assert phase_at(10.0, 1.05) == 0.5
check("t_phase_wraps", t_phase_wraps)


# -- SVG structure ---------------------------------------------------------
def t_svg_wellformed():
    svg = render_rosette_frame(0.1, petals=5, rings=2)
    assert svg.startswith("<svg")
    assert check_svg_wellformed(svg)
    # Balanced tags: the parse succeeding already proves it, but assert
    # the document closes properly too.
    assert svg.rstrip().endswith("</svg>")
check("t_svg_wellformed", t_svg_wellformed)


def t_petal_count_matches():
    for petals in (1, 7, 12):
        svg = render_rosette_frame(0.3, petals=petals, rings=2)
        count = len(re.findall(r"<ellipse\b", svg))
        assert count == petals, (petals, count)
check("t_petal_count_matches", t_petal_count_matches)


def t_ring_count_matches():
    for rings in (1, 3, 5):
        svg = render_rosette_frame(0.3, petals=4, rings=rings)
        count = len(re.findall(r"<circle\b", svg))
        # rings + 1: the accent seed at the centre is also a <circle>.
        assert count == rings + 1, (rings, count)
check("t_ring_count_matches", t_ring_count_matches)


def t_petal_breath_pulses_with_phase():
    # The visual "breath": petal radii scale with sin(2*pi*phase).
    # At phase 0.25 the sine peaks -> petals at maximum swell;
    # at phase 0.0 the sine is 0 -> petals at base size.
    geo_rest = rosette_geometry(0.0, petals=8, rings=2)
    geo_peak = rosette_geometry(0.25, petals=8, rings=2)
    assert geo_rest["breath"] == 1.0
    assert geo_peak["breath"] == 1.0 + BREATH_AMPLITUDE
    rx_rest = geo_rest["petals"][0]["rx"]
    rx_peak = geo_peak["petals"][0]["rx"]
    assert rx_peak > rx_rest
    assert math.isclose(rx_peak, rx_rest * (1.0 + BREATH_AMPLITUDE),
                        rel_tol=1e-9)
    # And the trough at 0.75 mirrors it.
    geo_trough = rosette_geometry(0.75, petals=8, rings=2)
    assert geo_trough["breath"] == 1.0 - BREATH_AMPLITUDE
check("t_petal_breath_pulses_with_phase", t_petal_breath_pulses_with_phase)


def t_band_palettes():
    # Band edges follow the conventional entrainment bands.
    assert band_for(2.0)[0] == "delta"
    assert band_for(6.0)[0] == "theta"
    assert band_for(10.0)[0] == "alpha"
    assert band_for(20.0)[0] == "beta"
    assert band_for(40.0)[0] == "gamma"
    name, palette = band_for(10.0)
    assert name == "alpha"
    assert set(palette) == {"background", "ring", "petal", "accent"}
    for colour in palette.values():
        assert re.fullmatch(r"#[0-9a-f]{6}", colour), colour
    # The band is stamped into the frame so a player can read it back.
    _, group = parse_svg(render_rosette_frame(0.2, 8, 2, beat_hz=40.0))
    assert group.get("data-band") == "gamma"
check("t_band_palettes", t_band_palettes)


def t_rejects_bad_inputs():
    for bad in (dict(petals=0), dict(rings=0), dict(petals=-3)):
        try:
            render_rosette_frame(0.1, **{"petals": 8, "rings": 2, **bad})
        except ValueError:
            continue
        raise AssertionError(f"bad input accepted: {bad}")
    try:
        band_for(-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative beat_hz accepted")
check("t_rejects_bad_inputs", t_rejects_bad_inputs)


# -- graceful rasterizer degradation ---------------------------------------
def _matplotlib_present():
    import importlib.util
    return importlib.util.find_spec("matplotlib") is not None


def _ffmpeg_present():
    import shutil
    return shutil.which("ffmpeg") is not None


def t_png_path():
    # If matplotlib is importable: a real PNG file appears.
    # If not: None, and crucially NO exception (graceful degradation).
    frames = session_frames(10.0, duration=0.5, fps=4,
                            petals=8, rings=2)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "rosette.png")
        result = render_png(frames, out, width=320, height=240,
                            petals=8, rings=2, beat_hz=10.0)
        if _matplotlib_present():
            assert result == out, result
            assert os.path.isfile(out) and os.path.getsize(out) > 0
            # A PNG's magic bytes: the file is really a PNG.
            with open(out, "rb") as fh:
                assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
        else:
            assert result is None
            assert render_mod.RENDER_LOG, "reason must be logged"
check("t_png_path", t_png_path)


def t_mp4_path():
    # If matplotlib AND ffmpeg are present: a real MP4 appears.
    # Otherwise: None, no exception.
    frames = session_frames(10.0, duration=0.5, fps=4,
                            petals=8, rings=2)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "session.mp4")
        result = render_mp4(frames, out, fps=4, width=320, height=240,
                            petals=8, rings=2, beat_hz=10.0)
        if _matplotlib_present() and _ffmpeg_present():
            assert result == out, result
            assert os.path.isfile(out) and os.path.getsize(out) > 0
        else:
            assert result is None
            assert render_mod.RENDER_LOG, "reason must be logged"
check("t_mp4_path", t_mp4_path)


def t_render_never_raises_on_empty():
    # Degenerate input is also graceful: None, never an exception.
    assert render_png([], "/tmp/never_written_xyz.png") is None
    assert render_mp4([], "/tmp/never_written_xyz.mp4") is None
check("t_render_never_raises_on_empty", t_render_never_raises_on_empty)


print(f"\n{PASSED} viz tests passed.")
