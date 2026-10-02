# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance/ai/cli.py -- `resonance-ai` command line.

HYPOTHESIS
    The wing's CLI must make the honesty boundaries VISIBLE: what
    runs offline, what needs a key, what needs weights. So every
    cloud subcommand fails with exit 3 and a message naming the
    missing credential (never a silent degradation), every local
    subcommand works with no key and no weights, and `probe`
    prints the whole backend picture on demand.

METHOD
    argparse with subcommands:
      analyze <wav>            -- measured track report (offline)
      advise <wav> --goal ".." -- Gemini mastering advice (key)
      do "<words>" --in <wav> --out <wav> [--ai]
                               -- parse + apply; local parser by
                               default, --ai uses Gemini (key)
      compose chords|melody|drums [--out wav]  (offline)
      compose idea --prompt ".."               (key)
      probe                    -- backend availability report
    The key arrives via --api-key-env (default GEMINI_API_KEY);
    it is never a CLI argument value (shell history is a log).

RESULT
    main() -> int. Exit 0 ok, 1 usage/engine errors, 2 command
    refused (unknown intent), 3 missing backend (no key, no
    torch/weights) -- the repo-wide missing-backend standard.
"""
import argparse
import os
import sys

import numpy as np

from resonance.core.io import read_wav, write_wav
from resonance.ai.analyze import analyze_track, format_report
from resonance.ai.command import (
    KNOWN_ACTIONS,
    ai_parse,
    execute_intent,
    parse_command,
)
from resonance.ai.compose import (
    ai_idea,
    chord_progression,
    drum_pattern,
    melody_from_chords,
    progression_to_text,
)
from resonance.ai.gemini import GeminiKeyMissing
from resonance.ai.master import advise, auto_chain
from resonance.ai.adapters import ModelNotAvailable
from resonance.ai.generate import MusicGenAdapter
from resonance.ai.transcribe import WhisperAdapter

EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_NO_BACKEND = 0, 1, 2, 3


def _api_key(args):
    return os.environ.get(args.api_key_env)


def _require_key(args):
    key = _api_key(args)
    if not key:
        print(f"resonance-ai: no API key. Set {args.api_key_env} "
              f"(get one at https://aistudio.google.com). This is a "
              f"cloud feature and will not run without it.",
              file=sys.stderr)
        raise SystemExit(EXIT_NO_BACKEND)
    return key


def cmd_analyze(args):
    audio, sr = read_wav(args.wav)
    result = analyze_track(audio, sr)
    print(format_report(result))
    return EXIT_OK


def cmd_advise(args):
    key = _require_key(args)
    audio, sr = read_wav(args.wav)
    result = analyze_track(audio, sr)
    print(format_report(result))
    print()
    try:
        advice = advise(result, args.goal, api_key=key)
    except GeminiKeyMissing as exc:
        print(f"resonance-ai: {exc}", file=sys.stderr)
        return EXIT_NO_BACKEND
    print("--- Gemini mastering advice (advice, not applied) ---")
    print(advice.text)
    if advice.code_executed:
        print("(the model ran code while reasoning)")
    print()
    print("--- deterministic auto-chain (offline, measured) ---")
    effects, reasons = auto_chain(result)
    for eff, why in zip(effects, reasons):
        print(f"  {eff}  # {why}")
    if not effects:
        print(f"  (none)  # {reasons[0]}")
    return EXIT_OK


def cmd_do(args):
    audio, sr = read_wav(args.input)
    if args.ai:
        key = _require_key(args)
        try:
            intent = ai_parse(args.words, api_key=key)
        except GeminiKeyMissing as exc:
            print(f"resonance-ai: {exc}", file=sys.stderr)
            return EXIT_NO_BACKEND
        except ValueError as exc:
            print(f"resonance-ai: {exc}", file=sys.stderr)
            return EXIT_REFUSED
    else:
        intent = parse_command(args.words)
    print(f"intent: {intent.action} (confidence {intent.confidence:.2f}, "
          f"source={intent.source})")
    print(f"  {intent.explanation}")
    if not intent.known:
        return EXIT_REFUSED
    try:
        out, report = execute_intent(intent, audio, sr)
    except ValueError as exc:
        print(f"resonance-ai: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    write_wav(args.output, out, sr)
    print(report)
    print(f"wrote {args.output}")
    return EXIT_OK


def cmd_compose(args):
    if args.what == "chords":
        prog = chord_progression(args.key, args.mood)
        print(progression_to_text(prog))
        return EXIT_OK
    if args.what == "melody":
        prog = chord_progression(args.key, args.mood)
        mel = melody_from_chords(prog, seed=args.seed,
                                 notes_per_chord=args.notes_per_chord)
        from resonance.core.notes import midi_to_note
        print(" ".join(midi_to_note(m) for m in mel))
        return EXIT_OK
    if args.what == "drums":
        audio, grid = drum_pattern(args.style, bars=args.bars,
                                   bpm=args.bpm, sr=44100)
        print(grid)
        if args.out:
            write_wav(args.out, audio, 44100)
            print(f"wrote {args.out}")
        return EXIT_OK
    if args.what == "idea":
        key = _require_key(args)
        try:
            result = ai_idea(args.prompt, api_key=key)
        except GeminiKeyMissing as exc:
            print(f"resonance-ai: {exc}", file=sys.stderr)
            return EXIT_NO_BACKEND
        print(result.text)
        return EXIT_OK
    print(f"resonance-ai: unknown compose target {args.what!r}",
          file=sys.stderr)
    return EXIT_ERROR


def cmd_probe(args):
    print("resonance-ai backend probe (guests, not residents):")
    print()
    key = _api_key(args)
    print(f"  gemini cloud: {'key present' if key else 'NO KEY'} "
          f"({args.api_key_env})")
    print()
    print(MusicGenAdapter().status_text())
    print()
    print(WhisperAdapter().status_text())
    return EXIT_OK


def build_parser():
    p = argparse.ArgumentParser(
        prog="resonance-ai",
        description="The AI Wing: analysis, mastering advice, "
                    "natural-language control, composition helpers.")
    p.add_argument("--api-key-env", default="GEMINI_API_KEY",
                   help="env var holding the Gemini key (never pass "
                        "the key as an argument)")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="measured track report (offline)")
    a.add_argument("wav")

    v = sub.add_parser("advise",
                       help="Gemini mastering advice + offline auto-chain")
    v.add_argument("wav")
    v.add_argument("--goal", required=True)

    d = sub.add_parser("do", help='run a natural-language command ("make '
                                  'it darker")')
    d.add_argument("words")
    d.add_argument("--in", dest="input", required=True)
    d.add_argument("--out", dest="output", required=True)
    d.add_argument("--ai", action="store_true",
                   help="parse with Gemini instead of the local parser")

    c = sub.add_parser("compose", help="composition helpers")
    c.add_argument("what", choices=["chords", "melody", "drums", "idea"])
    c.add_argument("--key", default="C")
    c.add_argument("--mood", default="hopeful")
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--notes-per-chord", type=int, default=4)
    c.add_argument("--style", default="boom_bap")
    c.add_argument("--bars", type=int, default=2)
    c.add_argument("--bpm", type=float, default=90.0)
    c.add_argument("--out", default=None)
    c.add_argument("--prompt", default="")

    sub.add_parser("probe", help="backend availability report")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "analyze":
            return cmd_analyze(args)
        if args.command == "advise":
            return cmd_advise(args)
        if args.command == "do":
            return cmd_do(args)
        if args.command == "compose":
            return cmd_compose(args)
        if args.command == "probe":
            return cmd_probe(args)
    except ModelNotAvailable as exc:
        print(f"resonance-ai: {exc}", file=sys.stderr)
        return EXIT_NO_BACKEND
    except (OSError, ValueError) as exc:
        print(f"resonance-ai: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
