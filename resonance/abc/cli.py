# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/abc/cli.py -- the `resonance-abc` executable.

Hypothesis: the built-in tune "The North Gate" is the honest zero-
  argument render: run the tool, get the tune as a WAV. The parser and
  renderer do the real work; the CLI only wires them to a filename.
Method:     parse_abc(BUILTIN_TUNE) -> render_tune(path=...) -> a stereo
  WAV. --sr and --mono are the only knobs; tempo comes from the tune's
  own Q: header, never from the CLI (the tune is the score).
Observation: `resonance-abc --help` exits 0 and names the tool;
  `resonance-abc -o gate.wav` produces a WAV whose frame count matches
  the tune's duration.
Result:     the [project.scripts] `resonance-abc` entry point.

Exit codes: 0 on success, 2 on bad arguments, 1 on parse/render failure.
"""
import argparse
import sys


def build_parser():
    """The argparse contract: the tune, with zero arguments."""
    p = argparse.ArgumentParser(
        prog="resonance-abc",
        description='Render the built-in ABC tune "The North Gate" to WAV.',
    )
    p.add_argument("--sr", type=int, default=44100,
                   help="sample rate in Hz (default: 44100)")
    p.add_argument("--mono", action="store_true",
                   help="render mono instead of stereo")
    p.add_argument("-o", "--output", default="north-gate.wav",
                   help="output WAV path (default: north-gate.wav)")
    return p


def main(argv=None):
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    from resonance.abc import parse_abc, render_tune
    from resonance.abc.tunes import BUILTIN_TUNE

    try:
        tune = parse_abc(BUILTIN_TUNE)
        audio = render_tune(tune, path=args.output, sr=args.sr,
                            stereo=not args.mono)
    except Exception as exc:
        print(f"resonance-abc: error: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    title = tune.headers.get("T", "(untitled)")
    print(f"resonance-abc: wrote {args.output} ({title!r}, "
          f"{audio.shape[-1]} frames @ {args.sr} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
