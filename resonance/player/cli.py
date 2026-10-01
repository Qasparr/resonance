# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/player/cli.py -- `resonance-play` command-line interface.

Hypothesis: the player needs a daily-driver CLI that is honest at
  every step: it shows which audio backend was probed, says SILENT
  REHEARSAL out loud when no backend exists, and labels the tempo
  backend (pitch preserved or NOT) before any sound is claimed.
Method:     argparse with subcommands -- list (playable files in a
  directory), play (load and play; --tempo sets the master ratio,
  --rehearsal forces silent rehearsal even when a backend exists),
  status (probe backends + tempo backend story without playing).
  play() blocks on a simple tick loop until the playlist ends or
  Ctrl-C; Ctrl-C stops the player cleanly (temp WAVs removed).
Observation: every command prints the backend story first; `play`
  refuses an empty playlist loudly instead of idling.
Result:    main(argv=None) -> int, wired as the `resonance-play`
  console entry (see pyproject [project.scripts]).

No medical or therapeutic claims are made about playback.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from resonance.player.engine import Player, probe_audio_backends
from resonance.player.tempo import MasterTempo

log = logging.getLogger("resonance.player.cli")

#: Suffixes the player can load: .wav natively, the rest via ffmpeg.
PLAYABLE_SUFFIXES = {
    ".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".oga", ".opus",
    ".m4a", ".mp4", ".aac", ".wma", ".wv", ".ape", ".webm", ".mkv",
}


def build_parser():
    """Construct the argparse parser (separate for testability)."""
    p = argparse.ArgumentParser(
        prog="resonance-play",
        description=(
            "RESONANCE player: playlist transport with an honest audio "
            "backend. Without a system player (aplay/paplay/afplay/ffplay) "
            "it runs in SILENT REHEARSAL mode -- real transport, no sound, "
            "said out loud."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("list", help="list playable files in a directory")
    ls.add_argument("directory", help="directory to scan for audio files")

    pl = sub.add_parser("play", help="play files (one or more paths)")
    pl.add_argument("paths", nargs="+", help="audio file(s) to play")
    pl.add_argument("--tempo", type=float, default=1.0,
                    help="master tempo ratio 0.25..4.0 (default 1.0)")
    pl.add_argument("--rehearsal", action="store_true",
                    help="force silent rehearsal even if a backend exists")
    pl.add_argument("--no-autoplay", action="store_true",
                    help="stop after each track instead of advancing")

    st = sub.add_parser("status", help="probe backends; print the honest story")
    st.add_argument("--tempo", type=float, default=1.0,
                    help="tempo ratio to report the backend story for")
    return p


def cmd_list(directory):
    """Print playable files found in directory. Returns int exit code."""
    d = Path(directory)
    if not d.is_dir():
        print(f"resonance-play: not a directory: {d}", file=sys.stderr)
        return 2
    files = sorted(f for f in d.iterdir()
                   if f.is_file() and f.suffix.lower() in PLAYABLE_SUFFIXES)
    if not files:
        print(f"resonance-play: no playable audio files in {d}")
        return 0
    for f in files:
        print(f)
    return 0


def cmd_status(tempo_ratio=1.0):
    """Print backend probe results + tempo backend story. Returns int."""
    backends = probe_audio_backends()
    print("Audio backends (probe order aplay/paplay/afplay/ffplay):")
    if backends:
        for b in backends:
            print(f"  found: {b.name} (seek support: "
                  f"{'yes' if b.supports_seek else 'no'})")
        print(f"Using: {backends[0].name}")
    else:
        print("  NONE FOUND -- the player will run in SILENT REHEARSAL mode:")
        print("  real transport, real clock, NO audible sound. Install a")
        print("  system player (e.g. ffmpeg, which provides ffplay) for audio.")
    print()
    tempo = MasterTempo(ratio=tempo_ratio)
    print(tempo.describe())
    return 0


def cmd_play(paths, tempo_ratio=1.0, rehearsal=False, autoplay=True):
    """Load paths and play through them. Returns int exit code."""
    tempo = MasterTempo(ratio=tempo_ratio)
    if rehearsal:
        from resonance.player.engine import SILENT_REHEARSAL, AudioBackend
        backend = AudioBackend(name=SILENT_REHEARSAL, argv=(),
                               kind="rehearsal")
    else:
        backend = None  # Player probes honestly itself
    player = Player(tempo=tempo, backend=backend, autoplay=autoplay)
    print(player.describe().splitlines()[0] if not player.audible
          else f"backend: {player.backend.name} (audio mode)")
    print(tempo.describe())
    loaded = 0
    for path in paths:
        try:
            track = player.load(path)
        except Exception as exc:  # noqa: BLE001 -- loud per-file failure
            print(f"resonance-play: could not load {path}: {exc}",
                  file=sys.stderr)
            continue
        print(f"  queued: {track.title} ({track.duration:.1f}s)")
        loaded += 1
    if loaded == 0:
        print("resonance-play: nothing playable was loaded.", file=sys.stderr)
        return 2
    try:
        player.play(0)
        while player.state == "playing":
            time.sleep(0.2)
            player.tick()
    except KeyboardInterrupt:
        print("\nresonance-play: interrupted.")
    finally:
        player.stop(reason="interrupted" if player.state != "stopped"
                    else "finished")
    print("resonance-play: done.")
    return 0


def main(argv=None):
    """CLI entry point for `resonance-play`. Returns int exit code."""
    logging.basicConfig(level=logging.WARNING,
                        format="%(name)s: %(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "list":
        return cmd_list(args.directory)
    if args.command == "status":
        return cmd_status(args.tempo)
    if args.command == "play":
        return cmd_play(args.paths, tempo_ratio=args.tempo,
                        rehearsal=args.rehearsal,
                        autoplay=not args.no_autoplay)
    return 2  # unreachable: required=True


if __name__ == "__main__":
    sys.exit(main())
