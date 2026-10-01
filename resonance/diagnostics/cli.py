# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/diagnostics/cli.py -- the `resonance-diag` executable.

Hypothesis: the honest benchmark is a real one -- time.perf_counter
  around actual renders, and print the samples-per-second numbers.
  This CLI is the whole diagnostics suite at the command line: no
  required args, real numbers, nonzero exit if a benchmark fails its
  own assertions (measure_throughput asserts finiteness and positivity).
Method:     measure_throughput over three real renders -- a binaural
  tone, an 808 pattern, the built-in ABC tune -- then a table to
  stdout. --runs controls repeats (default 3).
Observation: `resonance-diag --help` exits 0 and names the tool;
  `resonance-diag` prints three real throughput lines and exits 0.
Result:     the [project.scripts] `resonance-diag` entry point.

Exit codes: 0 on success, 2 on bad arguments, 1 if any benchmark
asserts (a failed benchmark is data, and the CLI reports it as red).
"""
import argparse
import sys


def _benchmarks():
    """The suite: (name, render_fn, args, kwargs) triples, all real renders."""
    from resonance.abc import parse_abc, render_tune
    from resonance.abc.tunes import BUILTIN_TUNE
    from resonance.binaural.generator import binaural_beat
    from resonance.synth.cli import default_pattern
    from resonance.synth import render_pattern

    tune = parse_abc(BUILTIN_TUNE)
    return [
        ("binaural 10 Hz / 528 Hz, 4 s stereo",
         binaural_beat, (10.0,), {"carrier": 528, "duration": 4.0,
                                  "sample_rate": 44100}),
        ("808 pattern, 128 BPM, 1 bar",
         render_pattern, (default_pattern(), 128.0),
         {"bars": 1, "swing": 0.0, "sr": 44100}),
        ("ABC 'The North Gate', stereo",
         render_tune, (tune,), {"sr": 44100, "stereo": True}),
    ]


def build_parser():
    """The argparse contract: the suite runs with zero arguments."""
    p = argparse.ArgumentParser(
        prog="resonance-diag",
        description="Run the RESONANCE real-throughput benchmark suite.",
    )
    p.add_argument("--runs", type=int, default=3,
                   help="repeats per benchmark (default: 3)")
    return p


def main(argv=None):
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    from resonance.diagnostics.measure import measure_throughput

    try:
        for name, fn, fargs, fkwargs in _benchmarks():
            r = measure_throughput(fn, *fargs, n_runs=args.runs, **fkwargs)
            sps = r["samples_per_second"]
            realtime = sps / (r["channels"] * 44100)
            print(f"{name}: {sps:,.0f} samples/s "
                  f"({realtime:.1f}x realtime, "
                  f"{r['seconds_per_run']*1000:.1f} ms/run over "
                  f"{r['runs']} run(s))")
    except Exception as exc:
        print(f"resonance-diag: error: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    print("resonance-diag: suite complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
