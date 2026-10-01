# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/synth/cli.py -- the `resonance-808` executable.

Hypothesis: the step sequencer deserves a zero-argument executable: run
  it, get a drum-loop WAV. The default pattern is a plain house/groove
  skeleton -- kick four-on-the-floor, snare backbeat, driving 16th hats.
Method:     build the default grid (16 numeric steps per voice, the
  sequencer's contract), render_pattern at --bpm/--bars/--swing, write
  with core.io.write_wav.
Observation: `resonance-808 --help` exits 0 and names the tool;
  `resonance-808 -o loop.wav` produces a stereo WAV with the backbeat
  audible at the sequencer's own step_times offsets.
Result:     the [project.scripts] `resonance-808` entry point.

Exit codes: 0 on success, 2 on bad arguments, 1 on render/write failure.
"""


def default_pattern():
    """The built-in groove: four-on-the-floor kick, snare backbeat, 16th hats.

    Kept here (not in synth/) because it is a CLI default, not library
    doctrine -- the library's contract is grids, not this groove.
    """
    kick = [1.0 if i in (0, 4, 8, 12) else 0.0 for i in range(16)]
    snare = [1.0 if i in (4, 12) else 0.0 for i in range(16)]
    closed_hat = [1.0] * 16
    return {"kick": kick, "snare": snare, "closed_hat": closed_hat}


def build_parser():
    """The argparse contract: a loop with zero arguments."""
    import argparse

    p = argparse.ArgumentParser(
        prog="resonance-808",
        description="Render the built-in 808 pattern to a stereo WAV file.",
    )
    p.add_argument("--bpm", type=float, default=128.0,
                   help="tempo in beats per minute (default: 128.0)")
    p.add_argument("--bars", type=int, default=1,
                   help="bars to render (default: 1)")
    p.add_argument("--swing", type=float, default=0.0,
                   help="swing 0.0-0.75, delays odd 16ths (default: 0.0)")
    p.add_argument("--sr", type=int, default=44100,
                   help="sample rate in Hz (default: 44100)")
    p.add_argument("-o", "--output", default="pattern.wav",
                   help="output WAV path (default: pattern.wav)")
    return p


def main(argv=None):
    """Entry point. Returns the process exit code."""
    import sys

    args = build_parser().parse_args(argv)

    from resonance.core.io import write_wav
    from resonance.synth import render_pattern

    try:
        audio = render_pattern(default_pattern(), args.bpm, bars=args.bars,
                               swing=args.swing, sr=args.sr)
        write_wav(args.output, audio, args.sr)
    except Exception as exc:
        print(f"resonance-808: error: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    print(f"resonance-808: wrote {args.output} "
          f"({args.bpm} BPM, {args.bars} bar(s), {audio.shape[1]} frames)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
