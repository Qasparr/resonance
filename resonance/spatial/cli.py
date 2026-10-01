# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/spatial/cli.py -- the `resonance-spatial` executable.

Hypothesis: the spatial module deserves the same zero-plumbing
  treatment as resonance-binaural: a WAV in, a WAV out, no API.
Method:     argparse subcommands -- `pan` (mono WAV -> binaurally
  panned stereo WAV) and `orbit` (mono/stereo WAV -> orbiting stereo
  WAV); rendering through the library functions the tests exercise;
  WAV I/O through core.io.
Observation: `resonance-spatial --help` exits 0; `pan` of a mono
  file yields a stereo WAV; `orbit` of a binaural entrainment WAV
  yields a stereo WAV of the requested duration.
Result:     the [project.scripts] `resonance-spatial` entry point.

Exit codes: 0 on success, 2 on bad arguments, 1 on render/write
failure. Honesty: no medical or health claims of any kind -- the orbit
subcommand's help text says creative spatialization, and nothing
else.
"""
import argparse
import sys


def build_parser():
    """Subcommand layout: pan | orbit."""
    p = argparse.ArgumentParser(
        prog="resonance-spatial",
        description="3D spatial audio: binaural panning and orbital "
                    "motion (creative spatialization effects).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("pan", help="binaurally pan a mono WAV to stereo")
    pp.add_argument("input", help="input mono WAV path")
    pp.add_argument("-o", "--output", default="panned.wav",
                    help="output stereo WAV path (default: panned.wav)")
    pp.add_argument("--azimuth", type=float, default=0.0,
                    help="azimuth in degrees: 0 front, +90 hard right, "
                         "-90 hard left (default: 0)")
    pp.add_argument("--elevation", type=float, default=0.0,
                    help="elevation in degrees: 0 horizon, +90 zenith "
                         "(default: 0)")
    pp.add_argument("--head-radius", type=float, default=None,
                    help="head radius in meters (default: 0.0875)")

    po = sub.add_parser("orbit", help="orbit audio around the head "
                                      "(creative spatialization effect)")
    po.add_argument("input", help="input mono or stereo WAV path")
    po.add_argument("-o", "--output", default="orbit.wav",
                    help="output stereo WAV path (default: orbit.wav)")
    po.add_argument("--rate", type=float, default=0.25,
                    help="revolutions per second (default: 0.25)")
    po.add_argument("--duration", type=float, default=10.0,
                    help="seconds to render; input loops to fit "
                         "(default: 10)")
    po.add_argument("--elevation", type=float, default=0.0,
                    help="orbit elevation in degrees (default: 0)")
    po.add_argument("--head-radius", type=float, default=None,
                    help="head radius in meters (default: 0.0875)")
    return p


def main(argv=None):
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    from resonance.core.io import read_wav, write_wav
    from resonance.spatial import hrtf, orbit as orbit_mod

    head = None
    if args.head_radius is not None:
        head = hrtf.HeadModel(head_radius_m=args.head_radius)

    try:
        audio, sr = read_wav(args.input)
        if args.command == "pan":
            if not (audio.ndim == 1):
                print("resonance-spatial: error: pan needs a mono WAV "
                      f"(got shape {audio.shape})", file=sys.stderr)
                return 1
            out = hrtf.pan(audio, args.azimuth, args.elevation, head=head,
                            sample_rate=sr)
            detail = (f"azimuth {args.azimuth} deg, elevation "
                      f"{args.elevation} deg")
        else:  # orbit
            out = orbit_mod.orbit(audio, args.rate, args.duration,
                                  sample_rate=sr,
                                  elevation_deg=args.elevation,
                                  head=head)
            detail = (f"rate {args.rate} rev/s, duration {args.duration} s")
        write_wav(args.output, out, sr)
    except Exception as exc:
        print(f"resonance-spatial: error: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    print(f"resonance-spatial: wrote {args.output} "
          f"({args.command}, {detail}, {sr} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
