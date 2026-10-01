# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/tags/cli.py -- the `resonance-tag` executable.

Hypothesis: tagging deserves a dumb, honest command line: read
  prints tags, write sets them, ensure forces v2.4. Nothing clever,
  because clever CLI tools are where tags go to be corrupted.
Method:     argparse with two subcommands. `read` dumps the dict
  from tags.id3.read_tags as JSON-ish lines (cover printed as byte
  count, not raw bytes -- stdout is not a binary sewer). `write`
  accepts --title/--artist/--album/--track/--genre/--year and
  --cover pointing at an image file; `ensure-v24` calls ensure_v24.
Observation: `resonance-tag read file.mp3` exits 0 and names every
  text frame; `resonance-tag write --title T file.mp3` leaves a
  verified ID3v2.4 file behind.
Result:     main() -- the entry point. Exit codes: 0 success,
  2 bad arguments, 1 on failure (missing file, missing mutagen,
  version-verify failure).
"""
import argparse
import sys

from .id3 import ID3_VERSION, PUBLIC_KEYS, ensure_v24, read_tags, write_tags


def build_parser():
    p = argparse.ArgumentParser(
        prog="resonance-tag",
        description=(
            "Read/write ID3v2.4 tags (mutagen-backed). The only ID3 "
            "version this tool writes is v2.4 -- the real current "
            "version. There is no ID3v4 and this tool will not print one."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("read", help="print a file's tags")
    r.add_argument("path", help="audio file with an ID3 tag")

    w = sub.add_parser("write", help="write tags as ID3v2.4")
    w.add_argument("path", help="audio file to tag")
    w.add_argument("--title", help="TIT2")
    w.add_argument("--artist", help="TPE1")
    w.add_argument("--album", help="TALB")
    w.add_argument("--track", dest="tracknumber", help="TRCK (e.g. 3 or 3/12)")
    w.add_argument("--genre", help="TCON")
    w.add_argument("--year", help="TDRC (YYYY)")
    w.add_argument("--cover", help="image file to embed as APIC cover art")

    e = sub.add_parser("ensure-v24", help="rewrite tag as ID3v2.4 in place")
    e.add_argument("path", help="audio file to convert")
    return p


def _cmd_read(args):
    tags = read_tags(args.path)
    version = tags.pop("id3_version", None)
    cover = tags.pop("cover", None)
    if version is None:
        print("id3_version: none (no ID3 header)")
    else:
        print("id3_version: {}.{}.{}".format(*version))
    for key in PUBLIC_KEYS:
        if key == "cover":
            continue
        if key in tags:
            print(f"{key}: {tags[key]}")
    if cover:
        print(f"cover: {len(cover['data'])} bytes, {cover['mime']}")


def _cmd_write(args):
    tags = {}
    for key in ("title", "artist", "album", "tracknumber", "genre", "year"):
        value = getattr(args, key)
        if value:
            tags[key] = value
    if args.cover:
        with open(args.cover, "rb") as fh:
            data = fh.read()
        mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        tags["cover"] = {"data": data, "mime": mime}
    if not tags:
        print("resonance-tag write: nothing to write -- pass at least one --field",
              file=sys.stderr)
        return 2
    write_tags(args.path, tags)
    print(f"wrote {len(tags)} tag(s) as ID3v2.4 to {args.path}")


def _cmd_ensure(args):
    version = ensure_v24(args.path)
    print("ID3 version now: {}.{}.{}".format(*version))


def main(argv=None):
    """Entry point for the `resonance-tag` console script."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "read":
            _cmd_read(args)
        elif args.command == "write":
            return _cmd_write(args) or 0
        elif args.command == "ensure-v24":
            _cmd_ensure(args)
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"resonance-tag: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
