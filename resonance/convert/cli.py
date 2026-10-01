# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/convert/cli.py -- `resonance-convert` command-line interface.

Hypothesis: a converter CLI must make the backend story visible: which
  ffmpeg was probed (path + version), what argv it ran, and -- on
  failure -- the real stderr tail instead of a vague "conversion
  failed".
Method:     argparse with subcommands -- probe (print FfmpegInfo or the
  loud missing-backend message), convert (one src -> dst with codec /
  bitrate / rate / channels options), batch (a text file of src/dst
  pairs, one per line, "src -> dst" or whitespace-separated). All
  commands return int exit codes; failures print to stderr.
Observation: without ffmpeg, every command fails loudly naming the
  missing tool and how to install it -- never an empty output file
  presented as success.
Result:    main(argv=None) -> int, wired as the `resonance-convert`
  console entry (see pyproject [project.scripts]).

No medical or therapeutic claims are made about converted audio.
"""

import argparse
import logging
import sys
from pathlib import Path

from resonance.convert.convert import (
    ConvertError,
    LoudMissingBackend,
    convert,
    probe_ffmpeg,
)

log = logging.getLogger("resonance.convert.cli")


def build_parser():
    """Construct the argparse parser (separate for testability)."""
    p = argparse.ArgumentParser(
        prog="resonance-convert",
        description=(
            "RESONANCE format converter (ffmpeg-backed). Without ffmpeg "
            "installed, every command fails loudly -- conversions are never "
            "faked."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("probe", help="print the probed ffmpeg path + version")

    cv = sub.add_parser("convert", help="convert one file to another format")
    cv.add_argument("src", help="input file")
    cv.add_argument("dst", help="output file")
    cv.add_argument("--codec", default=None, help="audio codec, e.g. libmp3lame")
    cv.add_argument("--bitrate", default=None, help="e.g. 192k")
    cv.add_argument("--sample-rate", type=int, default=None)
    cv.add_argument("--channels", type=int, default=None, choices=(1, 2))

    bt = sub.add_parser("batch", help="convert many files from a job list")
    bt.add_argument("jobfile",
                    help="text file: one 'src -> dst' pair per line; "
                         "# lines and blanks ignored")
    bt.add_argument("--codec", default=None)
    bt.add_argument("--bitrate", default=None)
    return p


def cmd_probe():
    """Print ffmpeg path/version, or the loud missing-backend error."""
    try:
        info = probe_ffmpeg()
    except LoudMissingBackend as exc:
        print(f"resonance-convert: {exc}", file=sys.stderr)
        return 3
    print(f"ffmpeg: {info.path}")
    print(f"version: {info.version}")
    return 0


def cmd_convert(src, dst, codec=None, bitrate=None, sample_rate=None,
                channels=None):
    """Run one conversion; print the verified output path."""
    try:
        out = convert(src, dst, codec=codec, bitrate=bitrate,
                      sample_rate=sample_rate, channels=channels)
    except LoudMissingBackend as exc:
        print(f"resonance-convert: {exc}", file=sys.stderr)
        return 3
    except (ConvertError, FileNotFoundError, ValueError) as exc:
        print(f"resonance-convert: FAILED: {exc}", file=sys.stderr)
        return 1
    size = Path(out).stat().st_size
    print(f"converted: {out} ({size} bytes, verified non-empty)")
    return 0


def parse_jobfile(jobfile):
    """Parse 'src -> dst' (or whitespace-separated) pairs; skip #/blanks."""
    jobs = []
    for lineno, line in enumerate(Path(jobfile).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "->" in line:
            src, dst = line.split("->", 1)
        else:
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{jobfile}:{lineno}: expected 'src -> dst', got {line!r}"
                )
            src, dst = parts
        jobs.append((src.strip(), dst.strip()))
    return jobs


def cmd_batch(jobfile, codec=None, bitrate=None):
    """Convert every job in the file; stop loudly on the first failure."""
    from resonance.convert.convert import convert_many
    try:
        jobs = parse_jobfile(jobfile)
    except (OSError, ValueError) as exc:
        print(f"resonance-convert: bad job file: {exc}", file=sys.stderr)
        return 2
    if not jobs:
        print("resonance-convert: job file is empty.", file=sys.stderr)
        return 2
    ok = 0
    try:
        for (src, dst), result in convert_many(
                jobs, codec=codec, bitrate=bitrate):
            print(f"  ok: {src} -> {result}")
            ok += 1
    except LoudMissingBackend as exc:
        print(f"resonance-convert: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 -- the loud batch failure
        print(f"resonance-convert: BATCH STOPPED after {ok} ok: {exc}",
              file=sys.stderr)
        return 1
    print(f"resonance-convert: {ok}/{len(jobs)} converted.")
    return 0


def main(argv=None):
    """CLI entry point for `resonance-convert`. Returns int exit code."""
    logging.basicConfig(level=logging.WARNING,
                        format="%(name)s: %(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "probe":
        return cmd_probe()
    if args.command == "convert":
        return cmd_convert(args.src, args.dst, codec=args.codec,
                           bitrate=args.bitrate,
                           sample_rate=args.sample_rate,
                           channels=args.channels)
    if args.command == "batch":
        return cmd_batch(args.jobfile, codec=args.codec,
                         bitrate=args.bitrate)
    return 2  # unreachable: required=True


if __name__ == "__main__":
    sys.exit(main())
