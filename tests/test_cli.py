# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_cli.py -- script-style tests for the resonance CLI executables.

Run:  python3 tests/test_cli.py        (from the repo root)
   or python3 -m pytest tests/test_cli.py

Style (matching tests/test_telemetry.py in bleatirc): each test prints
"  ok: <name>"; the end prints "<N> cli tests passed." Any failure
raises immediately -- the first red line is the diagnosis.

Two test kinds per CLI:
  1. --help: invoked as `sys.executable -m resonance.<mod>.cli --help`
     (module form, so no pip install is needed), assert exit 0 and
     that the output names the tool.
  2. smoke: run the CLI with defaults in a tmp dir, assert it exits 0
     and produces its artifact (WAV with a RIFF header / SVG frames).
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


ROOT = str(Path(__file__).resolve().parent.parent)


def run_cli(module, *args, cwd=None):
    """Invoke a CLI as `python -m resonance.<module> <args>`; return CompletedProcess.

    PYTHONPATH pins the repo root so the subprocess tests the working
    tree, never an installed copy (the test venv has resonance pip-
    installed; without the pin, `python -m` would import the stale
    install instead of the code under test).
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", f"resonance.{module}", *args],
        capture_output=True, text=True, cwd=cwd or str(Path.cwd()),
        timeout=300, env=env,
    )


def t_help(module, prog):
    r = run_cli(module, "--help")
    assert r.returncode == 0, f"{module}: --help exited {r.returncode}: {r.stderr}"
    assert prog in r.stdout, f"{module}: --help did not name the tool {prog!r}"


check("t_help_binaural", lambda: t_help("binaural.cli", "resonance-binaural"))
check("t_help_synth", lambda: t_help("synth.cli", "resonance-808"))
check("t_help_abc", lambda: t_help("abc.cli", "resonance-abc"))
check("t_help_viz", lambda: t_help("viz.cli", "resonance-viz"))
check("t_help_diag", lambda: t_help("diagnostics.cli", "resonance-diag"))
check("t_help_edit", lambda: t_help("core.cli", "resonance-edit"))
check("t_help_serve", lambda: t_help("api.serve", "resonance-serve"))


def is_wav(path):
    with open(path, "rb") as fh:
        return fh.read(4) == b"RIFF"


def t_smoke_binaural():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "alpha.wav")
        r = run_cli("binaural.cli", "--beat", "10", "--carrier", "528",
                    "--duration", "1", "-o", out, cwd=tmp)
        assert r.returncode == 0, f"binaural failed: {r.stderr}"
        assert is_wav(out), "binaural did not produce a valid WAV"
check("t_smoke_binaural", t_smoke_binaural)


def t_smoke_synth():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "loop.wav")
        r = run_cli("synth.cli", "--bpm", "128", "--bars", "1", "-o", out,
                    cwd=tmp)
        assert r.returncode == 0, f"808 failed: {r.stderr}"
        assert is_wav(out), "808 did not produce a valid WAV"
check("t_smoke_synth", t_smoke_synth)


def t_smoke_abc():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "gate.wav")
        r = run_cli("abc.cli", "-o", out, cwd=tmp)
        assert r.returncode == 0, f"abc failed: {r.stderr}"
        assert is_wav(out), "abc did not produce a valid WAV"
check("t_smoke_abc", t_smoke_abc)


def t_smoke_viz():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "frames")
        r = run_cli("viz.cli", "--beat", "10", "--duration", "1", "--fps", "4",
                    "-o", out, cwd=tmp)
        assert r.returncode == 0, f"viz failed: {r.stderr}"
        svgs = sorted(Path(out).glob("frame-*.svg"))
        assert len(svgs) == 4, f"expected 4 frames, got {len(svgs)}"
        first = svgs[0].read_text(encoding="utf-8")
        assert "<svg" in first and "data-rotation-rad" in first, \
            "frame missing the viz contract stamps"
check("t_smoke_viz", t_smoke_viz)


def t_smoke_diag():
    r = run_cli("diagnostics.cli", "--runs", "1")
    assert r.returncode == 0, f"diag failed: {r.stderr}"
    assert "samples/s" in r.stdout, "diag printed no throughput numbers"
check("t_smoke_diag", t_smoke_diag)


def t_smoke_edit():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "edited.wav")
        r = run_cli("core.cli", "-o", out, cwd=tmp)
        assert r.returncode == 0, f"edit failed: {r.stderr}"
        assert is_wav(out), "edit did not produce a valid WAV"
check("t_smoke_edit", t_smoke_edit)


def t_bad_arg_exits_nonzero():
    # argparse's own contract: a bad flag exits 2, loudly, not 0.
    r = run_cli("binaural.cli", "--beat", "not-a-number")
    assert r.returncode == 2, f"bad arg exited {r.returncode}, expected 2"
check("t_bad_arg_exits_nonzero", t_bad_arg_exits_nonzero)


print(f"\n{PASSED} cli tests passed.")
