# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/rip/cli.py -- `resonance-rip` command line.

HYPOTHESIS
    A CLI that pretends to rip without the backend tools installed is
    worse than no CLI: the user walks away believing a rip happened.
    So every subcommand probes its backend first and fails LOUDLY --
    naming the missing tool and how to install it -- before touching
    any audio. `toc --demo` and `transcribe` on a WAV file are the
    two paths that work with zero external tools, and they say so.

METHOD
    argparse with subcommands:
      toc         -- list candidate drives (probe_drives); with
                     --demo, hash a canned 3-track TOC and print both
                     disc ids, proving the MusicBrainz algorithm runs
                     without hardware.
      rip         -- rip a track via a backend; requires cdparanoia or
                     cdda2wav on PATH, else RipBackendError. Encoding
                     to mp3/ogg requires lame/oggenc likewise. Without
                     the tools this command refuses -- loudly.
      transcribe  -- read a 16-bit PCM WAV (resonance.core.io), run
                     the heuristic pitch tracker, and emit a MIDI note
                     list or ABC text. Every output carries the
                     TRANSCRIPTION, NOT EXTRACTION caveat.
      backends    -- print the backend probe table.

OBSERVATION
    The honest CLI is mostly refusal paths: no drive, no tool, no rip
    -- and each refusal is specific enough to act on. The paths that
    do run (demo TOC, WAV transcription) are fully numpy/stdlib.

RESULT
    main() -> int exit code; wired as the `resonance-rip` console
    script (pyproject registration is the packaging group's file --
    this module only provides the entry point).

No medical or therapeutic claims are made by any subcommand here.
"""

import argparse
import json
import sys

from .secure import (
    ACCURATERIP_STATUS,
    DRIVE_OFFSET_DB_STATUS,
    SECURE_RIP_SCOPE,
    RipBackendError,
    probe_backends,
    require_backend,
)
from .toc import DiscTOC, Track, probe_drives
from .transcribe import (
    TRANSCRIPTION_CAVEAT,
    to_abc,
    to_midi,
    track_pitch,
    transcribe_polyphonic,
)


def _demo_toc():
    """A canned 3-track TOC so `toc --demo` works with no hardware."""
    return DiscTOC(
        tracks=[
            Track(number=1, start_lba=0, length_lba=15000),
            Track(number=2, start_lba=15000, length_lba=18000),
            Track(number=3, start_lba=33000, length_lba=12000),
        ],
        leadout_lba=45000,
    )


def cmd_toc(args):
    """List candidate drives; --demo hashes a canned TOC."""
    if args.demo:
        toc = _demo_toc()
        print("demo TOC: 3 tracks, lead-out LBA 45000 (no hardware needed)")
        for t in toc.tracks:
            print(f"  track {t.number}: start LBA {t.start_lba}, "
                  f"length {t.length_lba} sectors "
                  f"({t.length_lba / 75:.1f} s)")
        print(f"MusicBrainz disc id: {toc.musicbrainz_disc_id()}")
        print(f"CDDB disc id (legacy): {toc.cddb_disc_id()}")
        return 0
    drives = probe_drives()
    if not drives:
        print("probe_drives: no candidate optical-drive paths found on "
              "this machine. (Listing candidates only -- this command "
              "never opens a device.)")
        return 0
    print("candidate optical-drive paths (existence only -- not opened):")
    for d in drives:
        print(f"  {d['device']}: {'exists' if d['exists'] else 'absent'}")
    print("Reading a real TOC needs cdparanoia or cdda2wav; see "
          "`resonance-rip backends`.")
    return 0


def cmd_backends(_args):
    """Print the optional-backend probe table."""
    report = probe_backends()
    for name, info in sorted(report.items()):
        status = f"FOUND at {info['path']}" if info["available"] else "MISSING"
        print(f"{name:12s} {status}")
        print(f"              purpose: {info['purpose']}")
        if not info["available"]:
            print(f"              {info['install_hint']}")
    print()
    print("Secure-rip scope: " + SECURE_RIP_SCOPE)
    print(ACCURATERIP_STATUS)
    print(DRIVE_OFFSET_DB_STATUS)
    return 0


def cmd_rip(args):
    """Rip a track -- refuses loudly without the backend tools."""
    try:
        if args.backend == "cdparanoia":
            require_backend("cdparanoia")
        else:
            require_backend("cdda2wav")
        if args.format == "mp3":
            require_backend("lame")
        elif args.format == "ogg":
            require_backend("oggenc")
    except RipBackendError as exc:
        print(f"resonance-rip: {exc}", file=sys.stderr)
        return 3  # 3 = loud refusal: a required backend is missing
    # With the tools present, sector acquisition would be delegated to
    # the backend here (cdparanoia -Z-style batch read into the
    # SecureRip provider seam). The orchestration layer is specified;
    # the per-tool command assembly is backend-version-sensitive and
    # deliberately NOT faked from here.
    print("resonance-rip: backends present, but per-tool command "
          "assembly for this backend version is not yet implemented --\n"
          "refusing to guess at your drive. (This is the honest "
          "no-silent-guess rule applied to the CLI itself.)")
    return 3


def cmd_transcribe(args):
    """WAV -> heuristic pitch transcription (MIDI note list or ABC)."""
    from resonance.core.io import read_wav
    try:
        audio, sr = read_wav(args.input)
    except Exception as exc:
        print(f"resonance-rip: cannot read {args.input}: {exc}",
              file=sys.stderr)
        print("transcribe reads 16-bit PCM WAV only (resonance.core.io).",
              file=sys.stderr)
        return 2
    print(f"# {TRANSCRIPTION_CAVEAT}", file=sys.stderr)
    if args.polyphonic:
        events = transcribe_polyphonic(audio, sr)
        print(f"# polyphonic best-effort path; {len(events)} events",
              file=sys.stderr)
    else:
        events = track_pitch(audio, sr)
        print(f"# monophonic autocorrelation tracker; {len(events)} notes",
              file=sys.stderr)
    if args.to == "midi":
        notes = to_midi(events)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                json.dump(notes, fh, indent=2)
            print(f"wrote {len(notes)} transcribed notes to {args.output}")
        else:
            print(json.dumps(notes, indent=2))
    else:
        abc_text = to_abc(events, title=args.title, bpm=args.bpm)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(abc_text)
            print(f"wrote ABC transcription ({len(events)} notes) "
                  f"to {args.output}")
        else:
            print(abc_text, end="")
    return 0


def build_parser():
    """Assemble the resonance-rip argument parser."""
    parser = argparse.ArgumentParser(
        prog="resonance-rip",
        description=(
            "Rip audio CDs to MP3/Ogg/WAV, or transcribe audio to "
            "MIDI note lists / ABC notation. Backends (cdparanoia, "
            "lame, ...) are optional and probed at runtime -- missing "
            "tools produce loud, specific failures, never silent "
            "no-ops. MIDI/ABC output is TRANSCRIPTION (heuristic "
            "pitch detection), NOT extraction."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_toc = sub.add_parser("toc", help="list candidate drives / demo TOC hash")
    p_toc.add_argument("--demo", action="store_true",
                       help="hash a canned 3-track TOC (no hardware needed)")
    p_toc.set_defaults(func=cmd_toc)

    p_back = sub.add_parser("backends", help="probe optional backend tools")
    p_back.set_defaults(func=cmd_backends)

    p_rip = sub.add_parser("rip", help="rip a CD track (needs backend tools)")
    p_rip.add_argument("--device", default="/dev/cdrom",
                       help="drive device path (default: /dev/cdrom)")
    p_rip.add_argument("--track", type=int, default=1,
                       help="track number to rip (default: 1)")
    p_rip.add_argument("--format", choices=["wav", "mp3", "ogg"],
                       default="wav", help="target format (default: wav)")
    p_rip.add_argument("--backend", choices=["cdparanoia", "cdda2wav"],
                       default="cdparanoia", help="sector reader backend")
    p_rip.add_argument("--rereads", type=int, default=3,
                       help="re-reads per suspect sector (default: 3)")
    p_rip.add_argument("--offset-correction", type=int, default=0,
                       help="drive read-offset correction in samples "
                            "(default: 0; NOT a database lookup)")
    p_rip.set_defaults(func=cmd_rip)

    p_tr = sub.add_parser("transcribe",
                          help="WAV -> heuristic pitch transcription")
    p_tr.add_argument("--input", required=True,
                      help="16-bit PCM WAV file to transcribe")
    p_tr.add_argument("--to", choices=["midi", "abc"], default="abc",
                      help="output kind (default: abc)")
    p_tr.add_argument("--polyphonic", action="store_true",
                      help="best-effort polyphonic path (labeled)")
    p_tr.add_argument("--bpm", type=float, default=120.0,
                      help="tempo for ABC quantization (default: 120)")
    p_tr.add_argument("--title", default="Transcription",
                      help="title for the ABC header")
    p_tr.add_argument("--output", default=None,
                      help="write to file instead of stdout")
    p_tr.set_defaults(func=cmd_transcribe)
    return parser


def main(argv=None):
    """resonance-rip entry point -> int exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RipBackendError as exc:
        print(f"resonance-rip: {exc}", file=sys.stderr)
        return 3  # 3 = loud refusal: a required backend is missing


if __name__ == "__main__":
    sys.exit(main())
