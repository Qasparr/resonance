# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/metadata/cli.py -- the `resonance-lyrics` executable.

Hypothesis: fetching lyrics deserves the same dumb-honest CLI as
  tagging: `fetch` prints plain + synced, `karaoke` prints the
  current-line window for a given timestamp, and the tool never
  pretends a guess is a fact.
Method:     argparse. `fetch ARTIST TITLE` hits LRCLIB and prints
  plain lyrics, then synced lines with timestamps. `karaoke
  ARTIST TITLE --at SECONDS` renders the karaoke window at that
  playback position (for TUI/pipe consumers; the real-time TUI is
  a later milestone). `identify ARTIST TITLE` queries MusicBrainz
  and prints candidate MBIDs with scores, resolving the top one.
Observation: `resonance-lyrics fetch "Pink Floyd" "Time"` exits 0
  with lyrics or prints "not found"; network failure exits 1
  with the loud error.
Result:     main() -- the entry point. Exit codes: 0 success,
  2 bad arguments, 1 on failure. No cached files are written by
  this tool; caching is a later milestone's decision.
"""
import argparse
import sys

from .karaoke import KaraokeTrack
from .lrclib import LRCLibError, get_lyrics
from .musicbrainz import MusicBrainzError, get_recording, search_recording


def build_parser():
    p = argparse.ArgumentParser(
        prog="resonance-lyrics",
        description="Fetch lyrics (LRCLIB) and recording identity (MusicBrainz).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="fetch plain + synced lyrics")
    f.add_argument("artist", help="artist name")
    f.add_argument("title", help="track title")
    f.add_argument("--album", default="", help="album name (disambiguation)")
    f.add_argument("--duration", type=float, default=None,
                   help="track length in seconds (disambiguation)")

    k = sub.add_parser("karaoke", help="render karaoke window at a timestamp")
    k.add_argument("artist", help="artist name")
    k.add_argument("title", help="track title")
    k.add_argument("--at", dest="at", type=float, default=0.0,
                   help="playback position in seconds (default 0)")
    k.add_argument("--before", type=int, default=3, help="lines before current")
    k.add_argument("--after", type=int, default=2, help="lines after current")

    i = sub.add_parser("identify", help="search MusicBrainz for recording candidates")
    i.add_argument("artist", help="artist name")
    i.add_argument("title", help="track title")
    i.add_argument("--details", action="store_true",
                   help="fetch full details for the top candidate")
    return p


def _cmd_fetch(args):
    result = get_lyrics(args.artist, args.title,
                        album=args.album, duration_s=args.duration)
    if result is None:
        print(f"not found: {args.artist} -- {args.title}")
        return 0
    print(f"{result['artist']} -- {result['title']}")
    if result["instrumental"]:
        print("[instrumental: no lyrics]")
        return 0
    if result["plain"]:
        print("--- plain ---")
        print(result["plain"])
    if result["synced"]:
        print("--- synced ---")
        for t, line in result["synced"]:
            print(f"[{int(t // 60):02d}:{t % 60:05.2f}] {line}")


def _cmd_karaoke(args):
    result = get_lyrics(args.artist, args.title)
    if result is None or not result["synced"]:
        print(f"no synced lyrics for {args.artist} -- {args.title}")
        return 0
    track = KaraokeTrack(result["synced"])
    print(track.render_terminal(args.at, before=args.before, after=args.after))


def _cmd_identify(args):
    candidates = search_recording(args.artist, args.title)
    if not candidates:
        print(f"no MusicBrainz recordings for {args.artist} -- {args.title}")
        return 0
    for c in candidates:
        print(f"score={c['score']:>3}  mbid={c['mbid']}  {c['artist']} -- {c['title']}")
    if args.details:
        top = candidates[0]
        d = get_recording(top["mbid"])
        print(f"top: {d['artist']} -- {d['title']} [{d['album'] or 'no album'} "
              f"{d['year'] or 'no year'}] "
              f"{d['length_s']:.0f}s" if d["length_s"] else "")


def main(argv=None):
    """Entry point for the `resonance-lyrics` console script."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "fetch":
            return _cmd_fetch(args) or 0
        elif args.command == "karaoke":
            return _cmd_karaoke(args) or 0
        elif args.command == "identify":
            return _cmd_identify(args) or 0
    except (LRCLibError, MusicBrainzError) as exc:
        print(f"resonance-lyrics: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
