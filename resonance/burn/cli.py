# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/burn/cli.py -- `resonance-burn` command line.

HYPOTHESIS
    A burn CLI that pretends to burn without the backend tools is
    the worst lie in this codebase: the user walks away believing a
    disc exists. So every subcommand probes its backends first and
    fails LOUDLY -- naming the missing tool and how to install it --
    before touching anything. Exit code 3 is the v0.2.0 convention
    for that loud refusal. --dry-run prints the exact commands that
    WOULD run, for auditing, without executing anything.

METHOD
    argparse with subcommands:
      backends  -- print the backend probe table (always works).
      audio-cd  -- decode inputs to Red Book WAV via ffmpeg, write
                   a cue sheet with CD-Text, burn via wodim/cdrdao.
      data-disc -- build an ISO9660 image (genisoimage/xorriso),
                   verify it looks like an ISO, burn via
                   growisofs/wodim.
      dvd-video -- ffmpeg MPEG-2 encode, dvdauthor XML authoring
                   (titleset + chapters + still menu), VIDEO_TS
                   verification, ISO pack, growisofs burn.
    The device path is ALWAYS user-supplied (--device); this CLI
    never discovers drives and never picks a device for you.

OBSERVATION
    The honest CLI is mostly refusal paths on a machine without the
    tools -- and each refusal names the tool, its purpose, and the
    install command. --dry-run is the fully testable path: every
    command the burn WOULD run, printed, nothing executed.

RESULT
    main() -> int exit code; wired as the `resonance-burn` console
    script (pyproject registration is the packaging group's file --
    this module only provides the entry point).

Non-goals: Blu-ray (later roadmap); CSS/DRM circumvention (never).
No medical or therapeutic claims are made by any subcommand here.
"""

import argparse
import os
import sys
import tempfile

from .audio import burn_audio_cd, decode_to_cd_wav, write_cue
from .backends import BurnBackendError, probe_backends
from .data import burn_data, make_iso, verify_iso
from .dvd import (
    author_dvd,
    burn_dvd,
    encode_mpeg2,
    make_dvd_iso,
    verify_video_ts,
    write_dvdauthor_xml,
)


class DryRunner:
    """Fake runner for --dry-run: records argv, executes nothing."""

    def __init__(self):
        self.commands = []

    def __call__(self, argv, **kwargs):
        self.commands.append(list(argv))

        class P:
            returncode = 0
            stdout = ""
            stderr = ""
        return P()


def _fail_loud(exc):
    """BurnBackendError -> stderr + exit 3 (the missing-backend code)."""
    print(f"resonance-burn: {exc}", file=sys.stderr)
    return 3


def cmd_backends(_args):
    """Print the backend probe table (always works, no tools needed)."""
    report = probe_backends()
    for name, info in sorted(report.items()):
        status = f"FOUND at {info['path']}" if info["available"] else "MISSING"
        print(f"{name:12s} {status}")
        print(f"              purpose: {info['purpose']}")
        if not info["available"]:
            print(f"              {info['install_hint']}")
    print()
    print("Blu-ray: NOT IMPLEMENTED (later roadmap). "
          "CSS/DRM circumvention: NEVER.")
    return 0


def cmd_audio_cd(args):
    """Burn an audio CD from audio files."""
    runner = DryRunner() if args.dry_run else None
    try:
        with tempfile.TemporaryDirectory(prefix="resonance-burn-") as td:
            wavs = decode_to_cd_wav(args.inputs, td, runner=runner)
            tracks = [{"file": w, "title": os.path.basename(w),
                       "artist": args.artist or ""}
                      for w in wavs]
            cue = write_cue(tracks, os.path.join(td, "disc.cue"),
                            album_title=args.title,
                            album_artist=args.artist)
            if args.dry_run:
                argv = burn_audio_cd(wavs, args.device, cue_path=cue,
                                     backend=args.backend, runner=runner)
                print("# dry-run: the burn that WOULD run "
                      "(nothing executed):")
                for cmd in runner.commands:
                    print("  " + " ".join(cmd))
                print(f"# cue sheet that WOULD be used: {cue}")
                return 0
            argv = burn_audio_cd(wavs, args.device, cue_path=cue,
                                 backend=args.backend, runner=runner)
            print("burned audio CD via: " + " ".join(argv))
            return 0
    except BurnBackendError as exc:
        return _fail_loud(exc)


def cmd_data_disc(args):
    """Build an ISO9660 image and burn it as a data disc."""
    runner = DryRunner() if args.dry_run else None
    try:
        with tempfile.TemporaryDirectory(prefix="resonance-burn-") as td:
            iso = os.path.join(td, "data.iso")
            iso, tool = make_iso(args.inputs, iso,
                                 volume_id=args.volume_id, runner=runner)
            if args.dry_run:
                argv = burn_data(iso, args.device, media=args.media,
                                 runner=runner)
                print("# dry-run: commands that WOULD run "
                      "(nothing executed):")
                for cmd in runner.commands:
                    print("  " + " ".join(cmd))
                print(f"# ISO built with: {tool} (not actually built "
                      f"in dry-run)")
                return 0
            # Real path: the ISO was actually built above; verify it.
            if not verify_iso(iso):
                print("resonance-burn: the built image failed the "
                      "ISO9660 sanity check -- refusing to burn it.",
                      file=sys.stderr)
                return 1
            argv = burn_data(iso, args.device, media=args.media,
                             runner=runner)
            print("burned data disc via: " + " ".join(argv))
            return 0
    except BurnBackendError as exc:
        return _fail_loud(exc)


def cmd_dvd_video(args):
    """Author and burn a DVD-Video disc."""
    runner = DryRunner() if args.dry_run else None
    try:
        with tempfile.TemporaryDirectory(prefix="resonance-burn-") as td:
            mpgs = encode_mpeg2(args.inputs, td, standard=args.standard,
                                runner=runner)
            xml = write_dvdauthor_xml(mpgs, os.path.join(td, "dvd.xml"),
                                      chapters=args.chapters,
                                      menu_image=args.menu,
                                      titles=args.titles)
            vts_dir = os.path.join(td, "dvd")
            author_dvd(xml, vts_dir, runner=runner)
            if args.dry_run:
                iso = os.path.join(td, "dvd.iso")
                make_dvd_iso(vts_dir, iso, runner=runner)
                burn_dvd(iso, args.device, runner=runner)
                print("# dry-run: commands that WOULD run "
                      "(nothing executed):")
                for cmd in runner.commands:
                    print("  " + " ".join(cmd))
                print(f"# dvdauthor XML that WOULD be used: {xml}")
                return 0
            if not verify_video_ts(vts_dir):
                print("resonance-burn: VIDEO_TS failed the structure "
                      "check (missing VIDEO_TS.IFO or VTS_01 VOB) -- "
                      "refusing to burn it.", file=sys.stderr)
                return 1
            iso = os.path.join(td, "dvd.iso")
            make_dvd_iso(vts_dir, iso, runner=runner)
            argv = burn_dvd(iso, args.device, runner=runner)
            print("burned DVD-Video via: " + " ".join(argv))
            return 0
    except BurnBackendError as exc:
        return _fail_loud(exc)


def build_parser():
    """Assemble the resonance-burn argument parser."""
    parser = argparse.ArgumentParser(
        prog="resonance-burn",
        description=(
            "Burn audio CDs, data discs, and authored DVD-Video. "
            "Orchestrates system tools (ffmpeg, dvdauthor, "
            "genisoimage/xorriso, growisofs/wodim) -- never "
            "reimplements them, never fakes a burn. Missing tools "
            "fail loudly (exit 3) naming the tool and its install "
            "hint. --dry-run prints commands without executing."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_back = sub.add_parser("backends", help="probe backend tools")
    p_back.set_defaults(func=cmd_backends)

    p_acd = sub.add_parser("audio-cd", help="burn an audio CD")
    p_acd.add_argument("inputs", nargs="+", help="audio files (any "
                       "format ffmpeg reads)")
    p_acd.add_argument("--device", required=True,
                       help="drive device path, e.g. /dev/sr0 "
                            "(user-supplied, never auto-detected)")
    p_acd.add_argument("--backend", default="auto",
                       choices=["auto", "wodim", "cdrdao"])
    p_acd.add_argument("--title", default=None, help="album CD-Text title")
    p_acd.add_argument("--artist", default=None, help="album CD-Text artist")
    p_acd.add_argument("--dry-run", action="store_true",
                       help="print commands without executing")
    p_acd.set_defaults(func=cmd_audio_cd)

    p_data = sub.add_parser("data-disc", help="burn a data disc (ISO9660)")
    p_data.add_argument("inputs", nargs="+", help="files/dirs, as-is")
    p_data.add_argument("--device", required=True,
                        help="drive device path, e.g. /dev/sr0")
    p_data.add_argument("--media", default="dvd", choices=["dvd", "cd"])
    p_data.add_argument("--volume-id", default="RESONANCE")
    p_data.add_argument("--dry-run", action="store_true")
    p_data.set_defaults(func=cmd_data_disc)

    p_dvd = sub.add_parser("dvd-video", help="author and burn DVD-Video")
    p_dvd.add_argument("inputs", nargs="+", help="video files (any "
                       "format ffmpeg reads)")
    p_dvd.add_argument("--device", required=True,
                       help="drive device path, e.g. /dev/sr0")
    p_dvd.add_argument("--standard", default="pal", choices=["pal", "ntsc"])
    p_dvd.add_argument("--menu", default=None,
                       help="still-menu background image (JPEG/PNG)")
    p_dvd.add_argument("--titles", nargs="*", default=None,
                       help="menu button titles (default: Title 1, ...)")
    p_dvd.add_argument("--chapters", nargs="*", action="append",
                       default=None,
                       help="per-title chapter times, e.g. --chapters "
                            "0 5:00 10:00 (repeat per title)")
    p_dvd.add_argument("--dry-run", action="store_true")
    p_dvd.set_defaults(func=cmd_dvd_video)
    return parser


def main(argv=None):
    """resonance-burn entry point -> int exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BurnBackendError as exc:
        return _fail_loud(exc)


if __name__ == "__main__":
    sys.exit(main())
