# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance.viz.engine -- headless sacred-geometry / mandala frame engine.

HYPOTHESIS
    A binaural-style beat frequency (the difference tone between two
    carriers, in Hz) can be given a *visual correlate*: a slowly turning
    geometric rosette whose rotation angle advances exactly one full
    revolution per beat cycle, and whose petals "breathe" (grow and
    shrink) in step with the same phase. If the mapping from
    (beat_hz, time) -> (phase, SVG) is deterministic, then any frame can
    be re-rendered byte-identically later, which is the property a
    session recorder needs.

METHOD
    1. Convert time to beat phase with the one-line contract:
           phase = (beat_hz * t) % 1.0
       Phase lives on [0, 1); one beat cycle is one unit of phase.
    2. Map phase to a rosette: `rings` concentric circles plus `petals`
       ellipse petals arranged on spokes. The *whole rosette group* is
       rotated by exactly 2*pi*phase radians (emitted into the SVG as a
       `rotate(<degrees>)` transform on a top-level <g>, plus exact
       `data-rotation-rad` / `data-phase` attributes so the angle can be
       parsed back out of the markup bit-for-bit).
    3. Petal "breath": every petal's radii are scaled by
           breath = 1 + BREATH_AMPLITUDE * sin(2*pi*phase)
       so the petals swell to maximum at phase 0.25 and shrink to
       minimum at phase 0.75 -- the visual pulse correlated to the beat.
    4. Colour comes from the beat *band* (delta/theta/alpha/beta/gamma,
       standard entrainment band edges), each band owning a documented
       palette. Colours are aesthetic choices, not encodings of data.

OBSERVATION
    The engine is stdlib-only (math, xml). It emits SVG text, which is
    resolution-independent and parseable, so tests can assert on the
    geometry itself rather than on pixels. The same geometry dict also
    feeds resonance.viz.render, which rasterizes it with matplotlib --
    one source of truth, two backends, no drift between them.

RESULT
    render_rosette_frame(phase, petals, rings) -> str (SVG)
    session_frames(beat_hz, duration, fps)   -> [(t, phase, svg), ...]
    Both deterministic: identical inputs always produce identical SVG.

HONESTY NOTE (read before showing these frames to anyone)
    These frames are *visual correlates of the beat phase* -- moving
    mandala art that turns in step with an audio beat. They make no
    medical or therapeutic claim, and nothing about the geometry makes
    entrainment "work better". Aesthetic correlate, full stop.
"""

import math
import xml.etree.ElementTree as ET

# -- the one constant everything turns on ---------------------------------
# TAU = 2*pi. One beat cycle is one unit of phase, so the rosette's
# rotation angle in radians is simply TAU * phase: a full turn per beat.
TAU = 2.0 * math.pi

# How far the petals swell past their base size at the breath's peak.
# breath = 1 + BREATH_AMPLITUDE * sin(TAU * phase), so with 0.20 the
# petals range from 80% to 120% of base size across one beat cycle.
# Kept well below 1.0 so the scale factor can never go negative.
BREATH_AMPLITUDE = 0.20

# -- beat bands and their documented palettes ------------------------------
# Band edges (Hz) follow the conventional entrainment bands: delta < 4,
# theta 4-8, alpha 8-13, beta 13-30, gamma 30+. Each band gets a
# background, a ring colour, a petal colour, and an accent. These are
# aesthetic choices -- dusk indigo for the slow delta band, bright
# violet for the fast gamma band -- and carry no medical meaning.
_BANDS = (
    # (name, low_hz_inclusive, high_hz_exclusive, palette)
    ("delta", 0.0, 4.0, {
        "background": "#0a0e1a",  # deep night indigo
        "ring":       "#3b4a9e",  # muted indigo
        "petal":      "#5f74d8",  # brighter indigo
        "accent":     "#8fa2ff",  # pale periwinkle
    }),
    ("theta", 4.0, 8.0, {
        "background": "#0a1512",  # dark teal night
        "ring":       "#1f7a6b",  # deep teal
        "petal":      "#35b596",  # sea green
        "accent":     "#6fe3c1",  # mint
    }),
    ("alpha", 8.0, 13.0, {
        "background": "#0d140a",  # dark forest
        "ring":       "#4d8f3a",  # leaf green
        "petal":      "#7cc24a",  # bright leaf
        "accent":     "#c9e265",  # pale gold-green
    }),
    ("beta", 13.0, 30.0, {
        "background": "#150f08",  # dark umber
        "ring":       "#a86a1f",  # bronze
        "petal":      "#e09a2b",  # amber
        "accent":     "#ffcf6e",  # pale gold
    }),
    ("gamma", 30.0, float("inf"), {
        "background": "#120a18",  # dark plum
        "ring":       "#7a2fa0",  # violet
        "petal":      "#b04fd8",  # bright violet
        "accent":     "#e08fff",  # pale orchid
    }),
)


def band_for(beat_hz):
    """Return (band_name, palette_dict) for a beat frequency in Hz.

    Bands are the conventional entrainment edges: delta [0,4),
    theta [4,8), alpha [8,13), beta [13,30), gamma [30,inf).
    Raises ValueError on a negative frequency -- a beat cannot run
    backwards in this engine.
    """
    if beat_hz < 0:
        raise ValueError(f"beat_hz must be >= 0, got {beat_hz!r}")
    for name, low, high, palette in _BANDS:
        if low <= beat_hz < high:
            # Return a copy so callers cannot mutate the module table.
            return name, dict(palette)
    raise AssertionError(f"unreachable: no band for beat_hz={beat_hz!r}")


def phase_at(beat_hz, t):
    """Beat phase at time t (seconds): (beat_hz * t) % 1.0.

    Phase is dimensionless on [0, 1): 0.0 is the beat's downbeat,
    0.5 is half a cycle later, and 1.0 wraps back to 0.0. This is the
    single formula the whole timeline contract rests on.
    """
    return (beat_hz * t) % 1.0


def rosette_geometry(phase, petals, rings, size=400):
    """Compute the rosette's geometry as plain data (no SVG, no pixels).

    This is the single source of truth shared by the SVG backend
    (render_rosette_frame) and the matplotlib backend
    (resonance.viz.render.render_png): both draw from this dict, so a
    PNG can never drift out of agreement with its SVG frame.

    Returns a dict with:
      size, center, rotation_rad (= TAU*phase, the rosette's whole-turn
        angle), rotation_deg, breath (= 1 + A*sin(TAU*phase), the petal
        pulse factor), rings (list of {"r"}), petals (list of
        {"cx","cy","rx","ry","angle_deg"} where angle_deg is the spoke
        angle *before* the whole-rosette rotation is applied), and
      the band name + palette (filled in by render_rosette_frame).
    """
    # -- validate early, loudly: a malformed rosette is a bug, not art --
    if not isinstance(petals, int) or petals < 1:
        raise ValueError(f"petals must be a positive int, got {petals!r}")
    if not isinstance(rings, int) or rings < 1:
        raise ValueError(f"rings must be a positive int, got {rings!r}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size!r}")
    phase = float(phase) % 1.0  # tolerate callers passing e.g. 1.25

    center = size / 2.0
    max_r = 0.46 * size          # outermost ring stays inside the canvas

    # The documented contract: the whole rosette turns exactly one full
    # revolution per beat cycle. rotation_rad is TAU * phase -- parse it
    # back out of the SVG's data-rotation-rad attribute bit-for-bit.
    rotation_rad = TAU * phase
    rotation_deg = 360.0 * phase  # what SVG's rotate() actually consumes

    # The visual "breath": petals swell and shrink with sin(TAU*phase),
    # peaking at phase 0.25, bottoming at 0.75, neutral at 0/0.5.
    breath = 1.0 + BREATH_AMPLITUDE * math.sin(TAU * phase)

    # Concentric rings, evenly spaced from the centre outward.
    ring_list = [
        {"r": max_r * (i + 1) / rings}
        for i in range(rings)
    ]

    # Petals ride on spokes: petal k sits at angle TAU*k/petals, at 62%
    # of the max radius, drawn as an ellipse whose own long axis points
    # along the spoke (angle_deg), scaled by the breath.
    petal_cx_r = max_r * 0.62
    rx_base = max_r * 0.22
    ry_base = max_r * 0.11
    petal_list = []
    for k in range(petals):
        spoke = TAU * k / petals
        cx = center + petal_cx_r * math.cos(spoke)
        cy = center + petal_cx_r * math.sin(spoke)
        petal_list.append({
            "cx": cx,
            "cy": cy,
            "rx": rx_base * breath,
            "ry": ry_base * breath,
            "angle_deg": math.degrees(spoke),
        })

    return {
        "size": size,
        "center": center,
        "rotation_rad": rotation_rad,
        "rotation_deg": rotation_deg,
        "breath": breath,
        "rings": ring_list,
        "petals": petal_list,
    }


def render_rosette_frame(phase, petals, rings, beat_hz=10.0, size=400):
    """Render one rosette frame as an SVG string (stdlib only).

    phase      -- beat phase on [0, 1) (wraps if outside).
    petals     -- number of ellipse petals (positive int).
    rings      -- number of concentric rings (positive int).
    beat_hz    -- beat frequency; selects the band palette only.
    size       -- canvas width/height in user units.

    The returned SVG holds the whole rosette inside one top-level <g>
    whose transform is rotate(<degrees>): rotation_deg = 360 * phase,
    i.e. rotation_rad = 2*pi*phase exactly. The exact radian value is
    also stamped as data-rotation-rad (repr, so it round-trips
    bit-for-bit) for tests to parse back out.
    """
    band, palette = band_for(beat_hz)
    geo = rosette_geometry(phase, petals, rings, size=size)
    geo["band"] = band
    geo["palette"] = palette
    c = geo["center"]

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
    )
    # Canvas background: the band's night colour.
    parts.append(
        f'<rect x="0" y="0" width="{size}" height="{size}" '
        f'fill="{palette["background"]}"/>'
    )
    # The rotating group. rotation_deg is printed at 6 decimals for the
    # renderer; the exact value lives in data-rotation-rad via repr().
    # This group is what "the whole rosette's rotation angle = 2*pi*phase"
    # means in the contract.
    parts.append(
        f'<g transform="rotate({geo["rotation_deg"]:.6f} {c:.3f} {c:.3f})" '
        f'data-rotation-rad="{geo["rotation_rad"]!r}" '
        f'data-phase="{float(phase) % 1.0!r}" '
        f'data-band="{band}" data-beat-hz="{beat_hz!r}">'
    )
    # Concentric rings, stroked, unfilled.
    for ring in geo["rings"]:
        parts.append(
            f'<circle cx="{c:.3f}" cy="{c:.3f}" r="{ring["r"]:.3f}" '
            f'fill="none" stroke="{palette["ring"]}" stroke-width="1.5"/>'
        )
    # Petals: one <ellipse> per petal, each rotated about its own centre
    # to lie along its spoke. Petal count in the markup == petals, which
    # the tests assert by counting <ellipse> elements.
    for petal in geo["petals"]:
        parts.append(
            f'<ellipse cx="{petal["cx"]:.3f}" cy="{petal["cy"]:.3f}" '
            f'rx="{petal["rx"]:.3f}" ry="{petal["ry"]:.3f}" '
            f'fill="{palette["petal"]}" fill-opacity="0.55" '
            f'transform="rotate({petal["angle_deg"]:.6f} '
            f'{petal["cx"]:.3f} {petal["cy"]:.3f})"/>'
        )
    # A bright accent seed at the still centre of the turning rosette.
    parts.append(
        f'<circle cx="{c:.3f}" cy="{c:.3f}" r="{0.03 * size:.3f}" '
        f'fill="{palette["accent"]}"/>'
    )
    parts.append('</g>')
    parts.append('</svg>')
    return "".join(parts)


def session_frames(beat_hz, duration, fps, petals=12, rings=3, size=400):
    """Build a session timeline: list of (time_s, phase, svg) tuples.

    beat_hz  -- beat frequency in Hz.
    duration -- session length in seconds.
    fps      -- frames per second; frame i sits at t = i / fps.
    petals / rings / size -- passed through to render_rosette_frame.

    Phase follows the contract literally: phase = (beat_hz * t) % 1.0.
    So with beat_hz=10 and fps=4, frame 1 lands at t=0.25 s with phase
    exactly 0.5 -- the test asserts this, because it is the property a
    beat-synced player depends on.
    """
    if duration < 0:
        raise ValueError(f"duration must be >= 0, got {duration!r}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    n = int(duration * fps)
    frames = []
    for i in range(n):
        t = i / fps
        phase = (beat_hz * t) % 1.0  # the contract, verbatim
        svg = render_rosette_frame(phase, petals, rings,
                                   beat_hz=beat_hz, size=size)
        frames.append((t, phase, svg))
    return frames


def check_svg_wellformed(svg):
    """Parse-check helper: returns True iff the SVG is well-formed XML
    starting with an <svg> root. Used by the test suite."""
    if not svg.startswith("<svg"):
        return False
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return False
    # Strip the namespace before comparing the tag name.
    tag = root.tag.rsplit("}", 1)[-1]
    return tag == "svg"
