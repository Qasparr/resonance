# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance.viz.render -- optional rasterizers for the frame engine.

HYPOTHESIS
    The SVG frames from resonance.viz.engine are exact and portable,
    but some players want pixels (a PNG still) or a container (an MP4
    session video). Those need matplotlib and ffmpeg respectively --
    heavy, optional, and possibly absent -- so the rasterizers must
    degrade gracefully instead of dragging in hard dependencies.

METHOD
    1. Probe, don't import: importlib.util.find_spec("matplotlib") and
       shutil.which("ffmpeg") decide availability. Neither module is
       imported at this file's load time -- there is no hard
       dependency on either, by construction.
    2. render_png re-draws the frame from engine.rosette_geometry --
       the SAME geometry dict the SVG backend uses. matplotlib cannot
       rasterize our SVG without extra libraries, and re-deriving the
       math by hand in two places would let the backends drift, so one
       source of truth feeds both. The PNG matches the SVG by
       construction, not by coincidence.
    3. render_mp4 rasterizes every frame to a temp-dir PNG sequence
       (via render_png) and shells out to the ffmpeg binary with
       libx264. If any step fails, it cleans up and returns None.
    4. Every failure path returns None and records the reason in
       RENDER_LOG (plus a stderr note). These functions NEVER raise --
       a missing optional backend is an expected condition, not an
       exception. The test suite asserts exactly this.

OBSERVATION
    On this machine matplotlib is importable and ffmpeg exists at
    /usr/bin/ffmpeg, so both paths produce real files here; on a bare
    interpreter both return None without raising. Either way the caller
    gets a plain answer: a path, or None with the reason logged.

RESULT
    render_png(frames, path, ...) -> path str | None
    render_mp4(frames, path, ...) -> path str | None
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

from .engine import rosette_geometry

# Every graceful degradation records its reason here (and on stderr),
# so a caller that gets None can find out *why* without guessing.
RENDER_LOG = []


def _note(reason):
    """Record a degradation reason; never raises."""
    RENDER_LOG.append(reason)
    print(f"[resonance.viz.render] {reason}", file=sys.stderr)


def _have_matplotlib():
    """True iff matplotlib is importable right now (probe, not import)."""
    try:
        return importlib.util.find_spec("matplotlib") is not None
    except Exception:
        return False


def _have_ffmpeg():
    """True iff an ffmpeg binary is on PATH right now."""
    return shutil.which("ffmpeg") is not None


def render_png(frames, path, width=640, height=480, dpi=100,
               petals=12, rings=3, beat_hz=10.0):
    """Rasterize frames to a single PNG (the first frame) via matplotlib.

    frames   -- session_frames() output: [(t, phase, svg), ...]; only
                the first frame's phase is drawn (a still, not a film).
    path     -- destination filename, e.g. "/tmp/rosette.png".
    width/height/dpi -- output raster size.
    petals/rings/beat_hz -- must match the frames' construction, since
                the PNG is re-drawn from rosette_geometry, not from the
                SVG text. (Rationale: matplotlib has no SVG rasterizer
                in stdlib reach; the geometry dict is the shared source
                of truth, so both backends agree by construction.)

    Returns the path on success, or None (+ logged reason) when
    matplotlib is missing or anything goes wrong. Never raises.
    """
    # -- graceful gate 1: no matplotlib, no PNG. Expected, not an error.
    if not _have_matplotlib():
        _note("render_png: matplotlib not importable; returning None")
        return None
    if not frames:
        _note("render_png: no frames supplied; returning None")
        return None
    try:
        # Local imports only: nothing here may leak into module load,
        # keeping the "no hard dependency" promise literally true.
        import matplotlib
        matplotlib.use("Agg")  # headless: no display server needed
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Ellipse

        _t, phase, _svg = frames[0]
        geo = rosette_geometry(phase, petals, rings, size=400)
        palette = {
            "background": "#0a0e1a", "ring": "#3b4a9e",
            "petal": "#5f74d8", "accent": "#8fa2ff",
        }
        # Re-derive the band palette the honest way: ask the engine.
        from .engine import band_for
        _band, palette = band_for(beat_hz)

        c = geo["center"]
        size = geo["size"]
        # In the SVG the petals live inside the rotating <g>, so each
        # petal's absolute orientation is spoke angle + rosette rotation.
        total_rot = geo["rotation_deg"]

        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor(palette["background"])
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect("equal")
        ax.set_facecolor(palette["background"])
        ax.axis("off")

        for ring in geo["rings"]:
            ax.add_patch(Circle((c, c), ring["r"], fill=False,
                                ec=palette["ring"], lw=1.5))
        for petal in geo["petals"]:
            # matplotlib's Ellipse angle is in degrees, counter-clockwise,
            # matching the SVG rotate() convention used in the markup.
            ax.add_patch(Ellipse((petal["cx"], petal["cy"]),
                                 width=2 * petal["rx"],
                                 height=2 * petal["ry"],
                                 angle=petal["angle_deg"] + total_rot,
                                 fc=palette["petal"], alpha=0.55, ec="none"))
        ax.add_patch(Circle((c, c), 0.03 * size, fc=palette["accent"],
                            ec="none"))
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)

        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
        _note(f"render_png: {path!r} missing or empty after save")
        return None
    except Exception as exc:  # never let a rasterizer take down the app
        _note(f"render_png failed ({type(exc).__name__}): {exc}")
        return None


def render_mp4(frames, path, fps=30, width=640, height=480,
               petals=12, rings=3, beat_hz=10.0):
    """Encode frames to an MP4 via the ffmpeg binary (libx264).

    The pipeline: rasterize each frame to a temp PNG with render_png,
    then run ffmpeg on the numbered sequence. Needs BOTH backends:
    matplotlib to rasterize, ffmpeg to encode.

    Returns the path on success, or None (+ logged reason) when either
    backend is missing or encoding fails. Never raises.
    """
    # -- graceful gates: missing backends are expected conditions --
    if not _have_ffmpeg():
        _note("render_mp4: no ffmpeg binary on PATH; returning None")
        return None
    if not _have_matplotlib():
        _note("render_mp4: matplotlib not importable (needed to rasterize "
              "frames); returning None")
        return None
    if not frames:
        _note("render_mp4: no frames supplied; returning None")
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="resonance_mp4_") as tmp:
            # 1. Rasterize: one PNG per frame, ffmpeg's %04d pattern.
            for i, frame in enumerate(frames):
                png = os.path.join(tmp, f"frame_{i:04d}.png")
                if render_png([frame], png, width=width, height=height,
                              petals=petals, rings=rings,
                              beat_hz=beat_hz) is None:
                    _note(f"render_mp4: frame {i} failed to rasterize")
                    return None
            # 2. Encode: yuv420p for maximum player compatibility.
            cmd = [
                shutil.which("ffmpeg"), "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmp, "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-loglevel", "error",
                path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=120)
            if proc.returncode != 0:
                _note(f"render_mp4: ffmpeg exited {proc.returncode}: "
                      f"{proc.stderr.strip()[:200]}")
                return None
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
        _note(f"render_mp4: {path!r} missing or empty after encode")
        return None
    except Exception as exc:  # the encoder never takes down the app
        _note(f"render_mp4 failed ({type(exc).__name__}): {exc}")
        return None
