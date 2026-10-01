# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/api/serve.py -- the `resonance-serve` CLI.

Hypothesis: serving the app is a one-liner; the machinery belongs in
  resonance.api.service, not in the entry point.
Method:     parse --host/--port/--jobs-dir, refuse loudly when uvicorn
  or fastapi is missing (the lazy-import contract), then run.
Observation: `resonance-serve --help` works without fastapi installed;
  only an actual serve demands it.
Result:     the [project.scripts] entry point.
"""
import argparse
import sys


def main(argv=None):
    """CLI entry point for `resonance-serve`.

    Returns 0 on clean shutdown, 2 on usage/backend errors (the
    uvicorn-never-started case). Ctrl-C stops the server; uvicorn
    raises KeyboardInterrupt, which we swallow into a clean exit.
    """
    parser = argparse.ArgumentParser(
        prog="resonance-serve",
        description="Serve the RESONANCE render-job API (FastAPI + uvicorn).",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765,
                        help="bind port (default 8765)")
    parser.add_argument("--jobs-dir", default=None,
                        help="job WAV output dir (default: temp dir)")
    args = parser.parse_args(argv)

    # The refuse-loudly contract: argparse works with zero deps, so
    # --help and usage errors never need fastapi; only a real serve
    # does, and only then do we demand it.
    try:
        import uvicorn  # noqa: F401
    except Exception:
        print("error: resonance-serve needs uvicorn: pip install 'resonance[api]'",
              file=sys.stderr)
        return 2

    from resonance.api.service import create_app, fastapi_available

    if not fastapi_available():
        print("error: resonance-serve needs fastapi: pip install 'resonance[api]'",
              file=sys.stderr)
        return 2

    app = create_app(jobs_dir=args.jobs_dir)
    print(f"resonance-serve: listening on http://{args.host}:{args.port} "
          f"(jobs -> {args.jobs_dir or '<temp dir>'})")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except KeyboardInterrupt:
        print("\nresonance-serve: stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
