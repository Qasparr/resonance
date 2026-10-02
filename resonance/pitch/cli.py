# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/pitch/cli.py -- `resonance-tune` command line.

HYPOTHESIS
    An auto-tune CLI that prints nothing about what it changed is a
    silent enhancer -- the exact thing the roadmap's honest-limits
    section forbids. So `tune` always prints the per-segment report:
    what was detected, what it was moved to, by how many cents, and
    with what confidence. The audio output and the report are one
    action; you cannot get the first without seeing the second.

METHOD
    argparse with subcommands:
      detect  -- YIN pitch track of a 16-bit PCM WAV; prints
                 time/f0/confidence per voiced frame (numpy only).
      tune    -- auto-tune to a scale in corrective or effect mode;
                 writes the corrected WAV and prints the TuneReport.
                 NO formant preservation (stated on every surface);
                 large corrections will chipmunk -- that is physics,
                 not a bug report.

OBSERVATION
    Both paths run on numpy alone -- no torch, no CREPE, no network.
    CREPE-neural remains the documented optional upgrade (roadmap),
    not a hidden dependency.

RESULT
    main() -> int exit code; wired as the `resonance-tune` console
    script (pyproject registration is the packaging group's file --
    this module only provides the entry point).

No medical or therapeutic claims are made by any subcommand here.
"""

import argparse
import sys

from .detect import track_f0
from .tune import SCALES, autotune


def cmd_detect(args):
    """YIN pitch track -> printed frame table (numpy only)."""
    from resonance.core.io import read_wav
    try:
        audio, sr = read_wav(args.input)
    except Exception as exc:
        print(f"resonance-tune: cannot read {args.input}: {exc}",
              file=sys.stderr)
        return 2
    frames = track_f0(audio, sr, fmin=args.fmin, fmax=args.fmax)
    voiced = [f for f in frames if f.voiced]
    print(f"# YIN pitch track: {len(frames)} frames, {len(voiced)} voiced "
          f"(heuristic -- verify by ear)")
    print("# time_s\tf0_hz\tconfidence")
    for f in voiced:
        print(f"{f.time_s:.3f}\t{f.f0_hz:.1f}\t{f.confidence:.2f}")
    return 0


def cmd_tune(args):
    """Auto-tune a WAV -> corrected WAV + printed per-segment report."""
    from resonance.core.io import read_wav, write_wav
    try:
        audio, sr = read_wav(args.input)
    except Exception as exc:
        print(f"resonance-tune: cannot read {args.input}: {exc}",
              file=sys.stderr)
        return 2
    scale = args.scale
    if scale not in SCALES:
        print(f"resonance-tune: unknown scale {scale!r}; known: "
              f"{sorted(SCALES)}", file=sys.stderr)
        return 2
    corrected, report = autotune(
        audio, sr, scale=scale, mode=args.mode, amount=args.amount)
    out_path = args.output or args.input.replace(".wav", f".tuned-{args.mode}.wav")
    write_wav(out_path, corrected.astype("float32"), sr)
    print(f"wrote {out_path}")
    print(report.summary())
    print("# honest limit: no formant preservation -- large corrections "
          "shift the spectral envelope (chipmunk); pushed too far, the "
          "phase vocoder warbles. The cents above are what moved.")
    return 0


def build_parser():
    """Assemble the resonance-tune argument parser."""
    parser = argparse.ArgumentParser(
        prog="resonance-tune",
        description=(
            "YIN pitch detection and two-mode auto-tune (corrective / "
            "effect). Every correction is reported per segment with "
            "confidence and signed cent amount -- no silent enhancement. "
            "numpy only; no torch/CREPE required."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_det = sub.add_parser("detect", help="YIN pitch track of a WAV file")
    p_det.add_argument("--input", required=True, help="16-bit PCM WAV file")
    p_det.add_argument("--fmin", type=float, default=55.0)
    p_det.add_argument("--fmax", type=float, default=2000.0)
    p_det.set_defaults(func=cmd_detect)

    p_tune = sub.add_parser("tune", help="auto-tune a WAV file")
    p_tune.add_argument("--input", required=True, help="16-bit PCM WAV file")
    p_tune.add_argument("--output", default=None,
                        help="output WAV (default: <input>.tuned-<mode>.wav)")
    p_tune.add_argument("--scale", default="chromatic",
                        choices=sorted(SCALES),
                        help="target scale (default: chromatic)")
    p_tune.add_argument("--mode", default="corrective",
                        choices=["corrective", "effect"],
                        help="corrective = gentle/transparent, effect = "
                             "hard snap (default: corrective)")
    p_tune.add_argument("--amount", type=float, default=0.7,
                        help="corrective-mode pull toward target, 0..1 "
                             "(default: 0.7)")
    p_tune.set_defaults(func=cmd_tune)
    return parser


def main(argv=None):
    """resonance-tune entry point -> int exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
