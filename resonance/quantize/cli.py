# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/quantize/cli.py -- `resonance-quantize` command line.

HYPOTHESIS
    A quantization CLI that hides what it moved is a silent
    surgeon -- the timing twin of the silent enhancer the pitch
    module refuses to be. So `quantize` always prints the per-hit
    report: detected time, grid target, applied shift, confidence,
    and whether each hit was sliced or warped. The audio output and
    the report are one action.

METHOD
    argparse with subcommands:
      onsets    -- spectral-flux transient detection on a 16-bit PCM
                   WAV; prints time/strength/confidence per onset
                   (numpy only).
      quantize  -- snap to the beat grid at --bpm with --strength
                   0..100%; writes the corrected WAV and prints the
                   QuantizeReport. strength 0 = untouched,
                   bit-identical (tested, not promised).

OBSERVATION
    Both paths run on numpy alone. Percussive material gets slices;
    legato/vocal material gets onset-nudging warps with confidence-
    scaled shifts -- the honest-limits section of the roadmap,
    implemented, not quoted.

RESULT
    main() -> int exit code; wired as the `resonance-quantize`
    console script (pyproject registration is the packaging group's
    file -- this module only provides the entry point).

No medical or therapeutic claims are made by any subcommand here.
"""

import argparse
import sys

from .grid import quantize
from .onsets import detect_onsets


def cmd_onsets(args):
    """Spectral-flux onset detection -> printed onset table."""
    from resonance.core.io import read_wav
    try:
        audio, sr = read_wav(args.input)
    except Exception as exc:
        print(f"resonance-quantize: cannot read {args.input}: {exc}",
              file=sys.stderr)
        return 2
    onsets = detect_onsets(audio, sr, sensitivity=args.sensitivity)
    print(f"# spectral-flux onsets: {len(onsets)} detected "
          f"(heuristic -- verify by ear)")
    print("# time_s\tstrength\tconfidence")
    for o in onsets:
        print(f"{o.time_s:.3f}\t{o.strength:.4f}\t{o.confidence:.2f}")
    return 0


def cmd_quantize(args):
    """Snap a WAV to the beat grid -> corrected WAV + hit report."""
    from resonance.core.io import read_wav, write_wav
    try:
        audio, sr = read_wav(args.input)
    except Exception as exc:
        print(f"resonance-quantize: cannot read {args.input}: {exc}",
              file=sys.stderr)
        return 2
    out, report = quantize(audio, sr, bpm=args.bpm,
                           strength=args.strength / 100.0,
                           mode=args.mode)
    out_path = args.output or args.input.replace(".wav", ".quantized.wav")
    write_wav(out_path, out.astype("float32"), sr)
    print(f"wrote {out_path}")
    print(report.summary())
    print("# honest limit: percussive material slices cleanly; "
          "vocal/legato gets onset-nudging, not surgery -- the "
          "confidences above are where the tracker was unsure.")
    return 0


def build_parser():
    """Assemble the resonance-quantize argument parser."""
    parser = argparse.ArgumentParser(
        prog="resonance-quantize",
        description=(
            "Transient/onset detection and beat-grid timing "
            "quantization with adjustable strength (0-100%%). Every "
            "moved hit is reported with its shift and confidence -- "
            "no silent surgery. numpy only."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_on = sub.add_parser("onsets", help="detect transients in a WAV file")
    p_on.add_argument("--input", required=True, help="16-bit PCM WAV file")
    p_on.add_argument("--sensitivity", type=float, default=1.5,
                      help="peak threshold factor (default: 1.5; higher "
                           "= fewer, surer onsets)")
    p_on.set_defaults(func=cmd_onsets)

    p_q = sub.add_parser("quantize", help="snap a WAV file to the beat grid")
    p_q.add_argument("--input", required=True, help="16-bit PCM WAV file")
    p_q.add_argument("--output", default=None,
                     help="output WAV (default: <input>.quantized.wav)")
    p_q.add_argument("--bpm", type=float, required=True,
                     help="grid tempo in BPM")
    p_q.add_argument("--strength", type=float, default=100.0,
                     help="quantization strength 0-100%% (0 = untouched, "
                          "default: 100)")
    p_q.add_argument("--mode", default="auto",
                     choices=["auto", "slice", "warp"],
                     help="slice = percussive, warp = legato/vocal, "
                          "auto = documented heuristic (default: auto)")
    p_q.set_defaults(func=cmd_quantize)
    return parser


def main(argv=None):
    """resonance-quantize entry point -> int exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
