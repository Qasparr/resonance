# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/core/cli.py -- the `resonance-edit` executable.

Hypothesis: the editing primitives deserve a zero-argument executable:
  with no input file, synthesize a short demo tone and run it through
  the default chain (fade-in, fade-out, normalize) so the tool always
  demonstrates its own machinery on a real buffer.
Method:     -i/--input reads a WAV when given; otherwise a 3 s 440 Hz
  sine is synthesized. The chain applies in fixed order -- trim, gain,
  fade-in, fade-out, soft limiter, normalize -- with each step logged
  to stdout. Output written with core.io.write_wav.
Observation: `resonance-edit --help` exits 0 and names the tool;
  `resonance-edit -o out.wav` writes a processed WAV.
Result:     the [project.scripts] `resonance-edit` entry point.

Exit codes: 0 on success, 2 on bad arguments, 1 on read/process/write
failure. Only 16-bit PCM WAV input is supported (the core.io
contract) -- anything else is a loud error, never a silent guess.
"""
import argparse
import sys


def _demo_tone(sr):
    """The zero-argument input: 3 s of 440 Hz, honest and boring.

    A generated tone -- not claimed to be anyone's recording -- so the
    tool's chain is exercisable without a file on disk.
    """
    import numpy as np

    t = np.arange(3 * sr, dtype=np.float64) / float(sr)
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def build_parser():
    """The argparse contract: works with zero arguments."""
    p = argparse.ArgumentParser(
        prog="resonance-edit",
        description="Apply waveform edits (trim/gain/fades/limiter/"
                    "normalize) to a WAV file.",
    )
    p.add_argument("-i", "--input", default=None,
                   help="input WAV path (default: a 3 s 440 Hz demo tone)")
    p.add_argument("--trim-start", type=float, default=0.0,
                   help="trim start in seconds (default: 0.0)")
    p.add_argument("--trim-end", type=float, default=None,
                   help="trim end in seconds (default: end of buffer)")
    p.add_argument("--gain-db", type=float, default=0.0,
                   help="gain in dB (default: 0.0)")
    p.add_argument("--fade-in", type=float, default=0.5,
                   help="fade-in length in seconds (default: 0.5)")
    p.add_argument("--fade-out", type=float, default=0.5,
                   help="fade-out length in seconds (default: 0.5)")
    p.add_argument("--limiter", action="store_true",
                   help="apply the soft limiter (default: off)")
    p.add_argument("--normalize", type=float, default=0.95,
                   help="peak-normalize target; 0 disables (default: 0.95)")
    p.add_argument("-o", "--output", default="edited.wav",
                   help="output WAV path (default: edited.wav)")
    return p


def main(argv=None):
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    from resonance.core import edit
    from resonance.core.io import read_wav, write_wav

    try:
        if args.input is None:
            sr = 44100
            audio = _demo_tone(sr)
            print("resonance-edit: no input -- using the 3 s 440 Hz demo tone")
        else:
            audio, sr = read_wav(args.input)
            print(f"resonance-edit: read {args.input} ({audio.shape}, "
                  f"{sr} Hz)")

        # Fixed chain order; every applied step is logged, so the tool
        # teaches while it works.
        n = audio.shape[-1]
        start = max(0, int(round(args.trim_start * sr)))
        end = n if args.trim_end is None else min(n, int(round(args.trim_end * sr)))
        if (start, end) != (0, n):
            audio = edit.trim(audio, start, end)
            print(f"resonance-edit: trim -> {start}:{end} samples")
        if args.gain_db:
            audio = edit.gain(audio, args.gain_db, unit="db")
            print(f"resonance-edit: gain {args.gain_db:+.1f} dB")
        if args.fade_in > 0:
            audio = edit.fade_in(audio, int(round(args.fade_in * sr)))
            print(f"resonance-edit: fade-in {args.fade_in} s")
        if args.fade_out > 0:
            audio = edit.fade_out(audio, int(round(args.fade_out * sr)))
            print(f"resonance-edit: fade-out {args.fade_out} s")
        if args.limiter:
            audio = edit.soft_limiter(audio)
            print("resonance-edit: soft limiter")
        if args.normalize > 0:
            audio = edit.normalize(audio, target=args.normalize)
            print(f"resonance-edit: normalized to {args.normalize}")

        write_wav(args.output, audio, sr)
    except Exception as exc:
        print(f"resonance-edit: error: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    print(f"resonance-edit: wrote {args.output} "
          f"({audio.shape[-1]} frames @ {sr} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
