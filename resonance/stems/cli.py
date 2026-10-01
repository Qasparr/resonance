# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/stems/cli.py -- `resonance-stems` command line (v0.2.0).

Hypothesis: the stems package needs a scriptable front door for the
  three honest operations: ask what backends exist (status), split a
  track into stems (separate), and recombine stems into a mix (mix).
Method:     argparse with three subcommands.  status prints the
  DemucsAdapter probe report and whether the rubberband CLI is on PATH.
  separate runs DemucsAdapter.separate() and writes one WAV per stem;
  when no backend exists it exits 3 with the StemSeparationUnavailable
  message on stderr -- the honest fallback, visible in the terminal.
  mix builds a StemMixer from --stem name=path arguments, applies
  --volume/--mute/--solo/--tempo flags, renders, and writes the mix
  WAV.  Retimed stems drift out of sync BY DESIGN (see mixer docs).
Observation: every failure mode here is loud: missing backends exit
  non-zero with install instructions; unknown stem names and bad
  values raise instead of being skipped.
Result:     a CLI that never pretends.

Run without installing:
  python3 -m resonance.stems.cli --help
  python3 -m resonance.stems.cli status
  python3 -m resonance.stems.cli separate song.wav --out stems/
  python3 -m resonance.stems.cli mix --stem vocals=v.wav --stem drums=d.wav \\
      --volume vocals=0.8 --mute drums --tempo vocals=1.25 --out mix.wav
"""
import argparse
import sys
from pathlib import Path

from resonance.core import io
from resonance.stems.mixer import (
    ResampleStretch,
    RubberBandAdapter,
    StemMixer,
    TimeStretchUnavailable,
)
from resonance.stems.separate import DemucsAdapter, STEMS, StemSeparationUnavailable

PROG = "resonance-stems"


def _split_kv(text, flag):
    """Parse name=value for the repeated mix flags."""
    if "=" not in text:
        raise SystemExit(f"{PROG} mix: {flag} expects name=value, got {text!r}")
    name, _, value = text.partition("=")
    name, value = name.strip(), value.strip()
    if not name:
        raise SystemExit(f"{PROG} mix: {flag} got an empty name in {text!r}")
    return name, value


def cmd_status(args):
    """Print backend availability.  Always exits 0 -- absence is a
    report, not a failure."""
    report = DemucsAdapter.probe()
    print("stem separation (Demucs):")
    print(f"  available : {report['available']}")
    print(f"  torch     : {report['torch'] or 'not importable'}")
    if report["demucs"]:
        print(f"  demucs    : {report['demucs']} (python API)")
    elif report["demucs_cli"]:
        print(f"  demucs    : CLI at {report['demucs_cli']}")
    else:
        print("  demucs    : not installed")
    print(f"  note      : {report['reason']}")
    print("per-stem tempo (time-stretch):")
    try:
        rb = RubberBandAdapter()
        print(f"  rubberband: {rb.binary} (pitch-preserving)")
    except TimeStretchUnavailable:
        print(
            "  rubberband: NOT FOUND -- set_tempo() falls back to "
            "ResampleStretch (rehearsal-grade, pitch NOT preserved)"
        )
    print(f"  fallback  : ResampleStretch (numpy, pitch not preserved)")
    print("honest limit: 4 stems (vocals/drums/bass/other); "
          "'isolate the guitar from the piano' is beyond open tooling.")
    return 0


def cmd_separate(args):
    """Run Demucs and write one WAV per stem."""
    adapter = DemucsAdapter(model=args.model)
    try:
        stems = adapter.separate(args.input, model=args.model)
    except StemSeparationUnavailable as exc:
        print(f"{PROG}: cannot separate -- {exc}", file=sys.stderr)
        return 3  # 3 = loud refusal: the Demucs backend is missing
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = None
    for name in STEMS:
        buf = stems[name]
        if sr is None:
            sr = args.sr
        dest = out_dir / f"{name}.wav"
        io.write_wav(dest, buf, sr)
        print(f"wrote {dest}  ({buf.shape[1]} frames)")
    print("separation complete: these are real Demucs stems, not EQ splits.")
    return 0


def cmd_mix(args):
    """Build a StemMixer from --stem flags and render the mix."""
    if not args.stem:
        raise SystemExit(f"{PROG} mix: give at least one --stem name=path")
    mixer = StemMixer(sample_rate=args.sr)
    for text in args.stem:
        name, path = _split_kv(text, "--stem")
        audio, sr = io.read_wav(path)
        if sr != mixer.sample_rate:
            raise SystemExit(
                f"{PROG} mix: stem {name!r} is {sr} Hz but the mixer is "
                f"{mixer.sample_rate} Hz -- convert it first (no silent "
                "resampling here)."
            )
        mixer.add_stem(name, audio)
    for text in args.volume:
        name, value = _split_kv(text, "--volume")
        mixer.set_volume(name, float(value))
    for name in args.mute:
        mixer.mute(name.strip())
    for name in args.solo:
        mixer.solo(name.strip())
    for text in args.tempo:
        name, value = _split_kv(text, "--tempo")
        mixer.set_tempo(name, float(value))
    rendered = mixer.mix()
    io.write_wav(args.out, rendered, mixer.sample_rate)
    print(f"wrote {args.out}  ({rendered.shape[1]} frames, {mixer.sample_rate} Hz)")
    retimed = [n for n in mixer.names
               if mixer._stems[n]["tempo"] != 1.0]
    if retimed:
        print("NOTE: retimed stems drift out of sync with the others BY "
              f"DESIGN (remixer's choice): {', '.join(retimed)}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog=PROG,
        description="Stem separation (Demucs) and stem mixing for RESONANCE. "
        "Backends are probed at runtime; missing ones fail loudly with "
        "install instructions -- nothing is faked.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("status", help="report backend availability")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("separate", help="split a WAV into stems via Demucs")
    s.add_argument("input", help="input WAV file (mono/stereo)")
    s.add_argument("--out", default="stems",
                   help="output directory for <stem>.wav (default: stems)")
    s.add_argument("--model", default="htdemucs",
                   help="demucs model name (default: htdemucs)")
    s.add_argument("--sr", type=int, default=44100,
                   help="sample rate of the written stems (default: 44100)")
    s.set_defaults(func=cmd_separate)

    s = sub.add_parser("mix", help="mix stems with volume/mute/solo/tempo")
    s.add_argument("--stem", action="append", default=[],
                   help="name=path to a stem WAV (repeatable)")
    s.add_argument("--volume", action="append", default=[],
                   help="name=gain, e.g. vocals=0.8 (repeatable)")
    s.add_argument("--mute", action="append", default=[],
                   help="stem name to mute (repeatable)")
    s.add_argument("--solo", action="append", default=[],
                   help="stem name to solo (repeatable)")
    s.add_argument("--tempo", action="append", default=[],
                   help="name=ratio, e.g. vocals=1.25 (repeatable; "
                   "retimed stems drift BY DESIGN)")
    s.add_argument("--out", required=True, help="output mix WAV path")
    s.add_argument("--sr", type=int, default=44100,
                   help="mixer sample rate; all stems must match "
                   "(default: 44100)")
    s.set_defaults(func=cmd_mix)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
