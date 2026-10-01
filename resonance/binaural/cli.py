# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/binaural/cli.py -- the `resonance-binaural` executable.

Hypothesis: the binaural generator deserves a zero-argument executable:
  run it, get a WAV. No session plumbing, no API -- a file on disk.
Method:     argparse with useful defaults (10 Hz alpha on the 528 Hz
  Solfeggio carrier, 10 s, 44100 Hz) plus --beat/--carrier/--duration/
  --kind knobs; render with the same functions the library tests
  exercise, write with core.io.write_wav.
Observation: `resonance-binaural --help` exits 0 and names the tool;
  `resonance-binaural -o alpha.wav` produces a valid stereo WAV.
Result:     the [project.scripts] `resonance-binaural` entry point.

Exit codes: 0 on success, 2 on bad arguments (argparse's own), 1 on a
render/write failure (with the error printed, never swallowed).

Honesty: no medical or therapeutic claims -- this CLI renders tones
and verifies nothing about what they do to a listener.
"""
import argparse
import sys

KINDS = ("binaural", "monaural", "isochronic")

DEFAULT_BEAT = 10.0      # alpha band -- the classic default session rate
DEFAULT_CARRIER = 528    # Solfeggio MI: cultural data, not a claim
DEFAULT_DURATION = 10.0
DEFAULT_SR = 44100


def build_parser():
    """The argparse contract: defaults that work with zero arguments."""
    p = argparse.ArgumentParser(
        prog="resonance-binaural",
        description="Render an entrainment tone to a stereo WAV file.",
    )
    p.add_argument("--kind", choices=KINDS, default="binaural",
                   help="tone family (default: binaural)")
    p.add_argument("--beat", type=float, default=DEFAULT_BEAT,
                   help=f"beat/pulse rate in Hz (default: {DEFAULT_BEAT})")
    p.add_argument("--carrier", type=float, default=DEFAULT_CARRIER,
                   help=f"carrier frequency in Hz (default: {DEFAULT_CARRIER})")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                   help=f"seconds to render (default: {DEFAULT_DURATION})")
    p.add_argument("--sr", type=int, default=DEFAULT_SR,
                   help=f"sample rate in Hz (default: {DEFAULT_SR})")
    p.add_argument("-o", "--output", default="binaural.wav",
                   help="output WAV path (default: binaural.wav)")
    return p


def main(argv=None):
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    from resonance.binaural.generator import (
        binaural_beat, isochronic_tone, monaural_beat,
    )
    from resonance.core.io import write_wav

    renderers = {
        "binaural": binaural_beat,
        "monaural": monaural_beat,
        "isochronic": isochronic_tone,
    }
    render = renderers[args.kind]
    try:
        # Signature notes: binaural_beat/monaural_beat take beat_hz;
        # isochronic_tone takes pulse_hz -- same position, different name.
        if args.kind == "isochronic":
            audio = render(args.beat, carrier=args.carrier,
                           duration=args.duration, sample_rate=args.sr)
        else:
            audio = render(args.beat, carrier=args.carrier,
                           duration=args.duration, sample_rate=args.sr)
        write_wav(args.output, audio, args.sr)
    except Exception as exc:
        print(f"resonance-binaural: error: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    frames = audio.shape[1] if audio.ndim == 2 else audio.shape[0]
    print(f"resonance-binaural: wrote {args.output} "
          f"({args.kind}, beat {args.beat} Hz, carrier {args.carrier} Hz, "
          f"{frames} frames @ {args.sr} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
