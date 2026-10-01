# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/ui/cli.py -- `resonance-ui` command line.

HYPOTHESIS
    A skin author needs one fast answer -- "is my skin valid, and if
    not, what exactly is wrong" -- without launching a player. A
    separate validate-skin command gives that answer with ALL problems
    reported at once (the contract's total-validation rule), while
    `run` exercises the TUI headlessly so the hooks and regions can be
    watched without a terminal UI framework.

METHOD
    argparse with subcommands:
      validate-skin <path>  -- load the JSON file, print valid/invalid,
                               every problem, every fallback log line,
                               and the resulting effective skin's region
                               order. Exit 0 when valid, 1 when the
                               skin has error-severity problems, 2 when
                               rejected whole (unparseable).
      run                   -- headless demo: apply the default skin,
                               play a fake track, print N rendered
                               frames (fires all five contract hooks
                               through the plugin manager).
    `run` prints ANSI to stdout; pipe it or view it, no curses.

OBSERVATION
    validate-skin is the contract made executable: malformed colors
    fall back loudly, a hidden transport is re-added loudly, unknown
    bindings fall back to mandala loudly -- and the exit code tells a
    script which of those happened.

RESULT
    main() -> int exit code; wired as the `resonance-ui` console
    script (pyproject registration is the packaging group's file --
    this module only provides the entry point).

No medical or therapeutic claims are made by any subcommand here.
"""

import argparse
import json
import sys

from .skin import apply_skin, load_skin_file, plugin_manager
from .tui import Player, run_demo


def cmd_validate(args):
    """Validate a skin file; report everything; exit by severity."""
    result = load_skin_file(args.path)
    print(f"skin: {args.path}")
    print(f"name: {result.name}  version: {result.version}")
    if result.rejected:
        print("REJECTED WHOLE: the file did not parse as a JSON object.")
        for line in result.log:
            print(f"  {line}")
        print("-> player.skin.rejected fires; the previous skin stays.")
        return 2
    print(f"valid: {result.valid} "
          f"({len(result.problems)} problem(s) reported)")
    for line in result.log:
        print(f"  {line}")
    regions = result.effective.get("layout", {}).get("regions", [])
    print(f"effective regions: {regions} "
          f"(transport always present: {'transport' in regions})")
    print(f"effective binding: "
          f"{result.effective.get('visualizer', {}).get('binding')}")
    return 0 if result.valid else 1


def cmd_run(args):
    """Headless demo run: default skin, fake track, printed frames."""
    # A tiny echo plugin on the shared manager so the hooks are
    # visible in the demo output -- the contract's hook points, fired.
    seen = []

    @plugin_manager.on("player.track.start")
    def _echo_start(payload):
        seen.append(f"hook player.track.start: {payload['title']}")

    @plugin_manager.on("player.track.end")
    def _echo_end(payload):
        seen.append(f"hook player.track.end: {payload['title']}")

    @plugin_manager.on("player.skin.apply")
    def _echo_apply(payload):
        seen.append(f"hook player.skin.apply: {payload['skin_name']}")

    print(run_demo(frames=args.frames))
    print("hooks observed during demo:")
    for line in seen:
        print(f"  {line}")
    return 0


def cmd_skin_demo(args):
    """Apply a skin file to a headless player and render one frame."""
    player = Player()
    result, applied = player.apply_skin(args.path)
    print(f"applied: {applied}  valid: {result.valid}")
    for line in result.log:
        print(f"  {line}")
    if applied:
        print()
        print(player.render_to_string(phase=0.25))
    return 0 if applied else 2


def build_parser():
    """Assemble the resonance-ui argument parser."""
    parser = argparse.ArgumentParser(
        prog="resonance-ui",
        description=(
            "Skinnable terminal player UI for RESONANCE. Skins are JSON "
            "files honoring docs/skin-contract.md; validation is total "
            "(all problems reported at once) with per-field fallbacks. "
            "The transport region can never be hidden. Headless-friendly: "
            "rendering is plain ANSI text, no curses."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate-skin",
                           help="validate a skin JSON file, report all "
                                "problems (exit 0 valid / 1 problems / "
                                "2 rejected)")
    p_val.add_argument("path", help="path to the skin JSON file")
    p_val.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run",
                           help="headless demo: play a fake track, print "
                                "rendered frames, show hooks firing")
    p_run.add_argument("--frames", type=int, default=8,
                       help="visualizer frames to print (default: 8)")
    p_run.set_defaults(func=cmd_run)

    p_demo = sub.add_parser("skin-demo",
                            help="apply a skin file and render one frame")
    p_demo.add_argument("path", help="path to the skin JSON file")
    p_demo.set_defaults(func=cmd_skin_demo)
    return parser


def main(argv=None):
    """resonance-ui entry point -> int exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
