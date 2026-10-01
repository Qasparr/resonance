# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/viz/cli.py -- the `resonance-viz` executable.

Hypothesis: the headless mandala engine's zero-argument story is a
  directory of SVG frames: run the tool, get one SVG per frame, each
  stamped with its beat phase. No rasterizers needed -- stdlib only.
Method:     session_frames(beat, duration, fps, petals, rings) ->
  (t, phase, svg) triples -> frame-NNNN.svg files in the output dir.
  --png/--mp4 are refused loudly when their backends are missing (the
  honest-degradation contract in viz/render.py); SVG always works.
Observation: `resonance-viz --help` exits 0 and names the tool;
  `resonance-viz -o frames/` writes exactly duration*fps SVG files.
Result:     the [project.scripts] `resonance-viz` entry point.

Exit codes: 0 on success, 2 on bad arguments, 1 on render/write failure.

Honesty: the frames are aesthetic visual correlates of beat phase, not
medical or therapeutic anything -- the viz engine's standing note.
"""
import argparse
import os
import sys


def build_parser():
    """The argparse contract: frames with zero arguments."""
    p = argparse.ArgumentParser(
        prog="resonance-viz",
        description="Render mandala SVG frames for a beat phase timeline.",
    )
    p.add_argument("--beat", type=float, default=10.0,
                   help="beat frequency in Hz (default: 10.0)")
    p.add_argument("--duration", type=float, default=2.0,
                   help="seconds of frames (default: 2.0)")
    p.add_argument("--fps", type=float, default=4.0,
                   help="frames per second (default: 4.0)")
    p.add_argument("--petals", type=int, default=12,
                   help="petals per rosette (default: 12)")
    p.add_argument("--rings", type=int, default=3,
                   help="concentric rings (default: 3)")
    p.add_argument("--size", type=int, default=400,
                   help="canvas width/height in user units (default: 400)")
    p.add_argument("-o", "--output", default="frames",
                   help="output directory for frame-*.svg (default: frames/)")
    return p


def main(argv=None):
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    from resonance.viz.engine import session_frames

    try:
        frames = session_frames(args.beat, args.duration, args.fps,
                                petals=args.petals, rings=args.rings,
                                size=args.size)
        os.makedirs(args.output, exist_ok=True)
        for i, (t, phase, svg) in enumerate(frames):
            path = os.path.join(args.output, f"frame-{i:04d}.svg")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(svg)
    except Exception as exc:
        print(f"resonance-viz: error: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    print(f"resonance-viz: wrote {len(frames)} SVG frames to {args.output}/ "
          f"(beat {args.beat} Hz, {args.duration} s @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
