# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance.viz -- headless mandala frame engine + optional rasterizers.

engine: stdlib-only SVG rosette frames, the visual correlates of beat
        phase (see engine.py's honesty note -- aesthetic, not medical).
render: PNG via matplotlib and MP4 via ffmpeg, both strictly optional;
        each degrades to None + a logged reason when its backend is
        missing, and neither is ever imported at module load.
"""

from .engine import (  # noqa: F401  (re-exported public surface)
    BREATH_AMPLITUDE,
    TAU,
    band_for,
    check_svg_wellformed,
    phase_at,
    render_rosette_frame,
    rosette_geometry,
    session_frames,
)
from .render import RENDER_LOG, render_mp4, render_png  # noqa: F401
