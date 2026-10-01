# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_stems.py -- script-style tests for resonance.stems (v0.2.0 group C).

Run:  python3 tests/test_stems.py        (from the repo root)
   or python3 -m pytest tests/test_stems.py

Style: each test prints "  ok: <name>"; the end prints
"<N> stems tests passed."  Any failure raises immediately -- the first
red line is the diagnosis.

The level math (gain/mute/solo) is asserted EXACT -- np.array_equal
against a hand computation of the same expression, not approximate
equality -- because the mixer contract is "sample-true".  No real
models are needed: stems are synthetic sines, and the missing-backend
paths are exercised with a genuinely absent torch/demucs/rubberband
plus import-monkeypatching for the branches we cannot reach for real.
"""
import builtins
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.core import io
from resonance.stems import (
    STEMS,
    DemucsAdapter,
    ResampleStretch,
    RubberBandAdapter,
    StemMixer,
    StemSeparationUnavailable,
    TimeStretchUnavailable,
)

PASSED = 0


def check(name, cond):
    global PASSED
    assert cond, f"FAILED: {name}"
    PASSED += 1
    print(f"  ok: {name}")


# -- synthetic stems ---------------------------------------------------------------
SR = 8000
N = 8000
_t = np.arange(N, dtype=np.float32) / SR


def sine(freq, phase=0.0):
    return np.sin(2 * np.pi * freq * _t + phase).astype(np.float32)


def stereo(mono, right_scale=0.5):
    return np.stack([mono, mono * np.float32(right_scale)]).astype(np.float32)


STEM_WAVES = {
    "vocals": stereo(sine(440.0)),
    "drums": stereo(sine(110.0, 0.7)),
    "bass": stereo(sine(55.0, 1.3)),
    "other": stereo(sine(330.0, 2.1)),
}
MONO_WAVE = sine(220.0)  # mono (N,) -- mixer must store it dual-mono


def make_mixer(**kw):
    m = StemMixer(sample_rate=SR, stretch=ResampleStretch(), **kw)
    for name, wave in STEM_WAVES.items():
        m.add_stem(name, wave)
    return m


def manual_mix(mixer):
    """Hand computation of mix(), mirroring the documented semantics:
    contribution = audio * float32(gain) for audible stems (skip muted;
    skip non-soloed when any solo is active); stretch when tempo != 1;
    the mix length is the longest (possibly retimed) stem, audible or
    not -- mute/solo/gain never change the timeline; zero-pad; sum in
    insertion order."""
    any_solo = any(s["soloed"] for s in mixer._stems.values())
    contribs = []
    lengths = []
    for name in mixer.names:
        s = mixer._stems[name]
        n = s["audio"].shape[1]
        sounds = not s["muted"] and (not any_solo or s["soloed"])
        if sounds:
            c = s["audio"] * np.float32(s["gain"])
            if s["tempo"] != 1.0:
                c = ResampleStretch().stretch(c, s["tempo"])
            c = np.asarray(c, dtype=np.float32)
            contribs.append(c)
            lengths.append(c.shape[1])
        else:
            lengths.append(
                ResampleStretch().output_length(n, s["tempo"])
                if s["tempo"] != 1.0 else n
            )
    m = max(lengths) if lengths else 0
    out = np.zeros((2, m), dtype=np.float32)
    for c in contribs:
        out[:, : c.shape[1]] += c
    return out


# -- the level math, exact ------------------------------------------------------------
check("stem vocabulary is the honest four",
      STEMS == ("vocals", "drums", "bass", "other"))

m = make_mixer()
check("default gain-1 mix is bit-exact vs manual sum",
      np.array_equal(m.mix(), manual_mix(m)))

m = make_mixer()
m.set_volume("vocals", 0.5)
m.set_volume("bass", 2.0)
check("per-stem gain is bit-exact vs manual",
      np.array_equal(m.mix(), manual_mix(m)))

m = make_mixer()
m.set_volume("drums", 0.0)
check("gain 0.0 silences exactly",
      np.array_equal(m.mix(), manual_mix(m)))

m = make_mixer()
m.mute("drums")
check("mute excludes the stem, bit-exact",
      np.array_equal(m.mix(), manual_mix(m)))
check("muted stem renders None",
      m.render_stem("drums") is None)

m = make_mixer()
m.solo("vocals")
check("solo isolates one stem, bit-exact",
      np.array_equal(m.mix(), manual_mix(m)))
check("soloed mix equals the soloed stem's gain-1 signal",
      np.array_equal(m.mix(), STEM_WAVES["vocals"]))

m = make_mixer()
m.solo("vocals")
m.solo("bass")
check("two solos sum, bit-exact",
      np.array_equal(m.mix(), manual_mix(m)))

m = make_mixer()
m.solo("vocals")
m.mute("vocals")
check("mute wins over solo: muted+soloed stem is silent but owns timeline",
      np.array_equal(m.mix(), np.zeros((2, N), dtype=np.float32)))

m = make_mixer()
m.solo("vocals")
m.mute("drums")  # non-soloed mute changes nothing audible
check("solo still sounds when an un-soloed stem is muted",
      np.array_equal(m.mix(), STEM_WAVES["vocals"]))

m = make_mixer()
m.mute("vocals")
m.mute("drums")
m.mute("bass")
m.mute("other")
check("all muted renders honest digital silence",
      np.array_equal(m.mix(), np.zeros((2, N), dtype=np.float32)))

m = StemMixer(sample_rate=SR, stretch=ResampleStretch())
m.add_stem("lead", MONO_WAVE)
expected = np.stack([MONO_WAVE, MONO_WAVE]).astype(np.float32)
check("mono stem is stored and mixed dual-mono, bit-exact",
      np.array_equal(m.mix(), expected))

# -- per-stem tempo: drift by design ------------------------------------------------------
m = make_mixer()
m.set_tempo("vocals", 2.0)
rendered = m.render_stem("vocals")
check("tempo 2.0 halves the stem length (retime is real)",
      rendered.shape == (2, N // 2))
check("retimed mix stays bit-exact vs manual",
      np.array_equal(m.mix(), manual_mix(m)))
check("mix pads to the longest stem (others keep full length)",
      m.mix().shape == (2, N))

m = make_mixer()
m.set_tempo("other", 0.5)
check("tempo 0.5 doubles the stem length: drift BY DESIGN",
      m.render_stem("other").shape == (2, 2 * N))
check("mix length follows the longest retimed stem",
      m.mix().shape == (2, 2 * N))
check("drifted mix is still bit-exact vs manual",
      np.array_equal(m.mix(), manual_mix(m)))

m = make_mixer()
m.set_tempo("vocals", 1.0)
check("tempo 1.0 is the identity (exact copy)",
      np.array_equal(m.render_stem("vocals"), STEM_WAVES["vocals"]))

for bad in (0, -1.0, float("nan"), float("inf"), "fast"):
    m = make_mixer()
    try:
        m.set_tempo("vocals", bad)
        raise SystemExit(f"set_tempo({bad!r}) should have raised")
    except ValueError:
        pass
check("bad tempo ratios raise ValueError, loudly", True)

m = make_mixer()
try:
    m.set_volume("vocals", -0.5)
    raise SystemExit("negative gain should have raised")
except ValueError:
    pass
check("negative gain raises ValueError", True)

m = make_mixer()
for op in (lambda: m.mute("nope"), lambda: m.solo("nope"),
           lambda: m.set_volume("nope", 1.0), lambda: m.set_tempo("nope", 2.0),
           lambda: m.remove_stem("nope")):
    try:
        op()
        raise SystemExit("unknown stem name should have raised")
    except KeyError:
        pass
check("unknown stem names raise KeyError, never silent no-ops", True)

check("ResampleStretch is labeled rehearsal-grade (pitch NOT preserved)",
      ResampleStretch.preserves_pitch is False)
check("RubberBandAdapter claims pitch preservation (class attr, no binary needed)",
      RubberBandAdapter.preserves_pitch is True)

# -- loud failures: genuinely absent backends ----------------------------------------------
report = DemucsAdapter.probe()
check("probe() reports unavailable with torch genuinely absent",
      report["available"] is False and "torch" in report["reason"].lower())
check("probe() never raises for missing deps", isinstance(report, dict))

real_import = builtins.__import__


def _block(names):
    def fake_import(name, *a, **k):
        if name in names or any(name.startswith(n + ".") for n in names):
            raise ImportError(f"No module named {name!r} (test-injected)")
        return real_import(name, *a, **k)
    return fake_import


builtins.__import__ = _block({"torch", "demucs"})
try:
    try:
        DemucsAdapter().separate("whatever.wav")
        raise SystemExit("missing torch should raise StemSeparationUnavailable")
    except StemSeparationUnavailable as exc:
        msg = str(exc)
        assert "torch" in msg.lower(), "message must name torch"
        assert "pip install torch" in msg, "message must say how to install torch"
        assert "Refusing rather than faking" in msg
finally:
    builtins.__import__ = real_import
check("missing torch -> loud StemSeparationUnavailable naming torch + install", True)

# torch present (faked), demucs missing -> the demucs branch of the loud error
fake_torch = types.ModuleType("torch")
fake_torch.__version__ = "9.9.9-test"
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = fake_torch
builtins.__import__ = _block({"demucs"})
try:
    try:
        DemucsAdapter().separate("whatever.wav")
        raise SystemExit("missing demucs should raise StemSeparationUnavailable")
    except StemSeparationUnavailable as exc:
        msg = str(exc)
        assert "demucs" in msg.lower(), "message must name demucs"
        assert "pip install demucs" in msg, "message must say how to install demucs"
finally:
    builtins.__import__ = real_import
    del sys.modules["torch"]
check("missing demucs (torch faked present) -> loud error naming demucs", True)

try:
    RubberBandAdapter()
    raise SystemExit("missing rubberband binary should raise")
except TimeStretchUnavailable as exc:
    assert "rubberband" in str(exc).lower(), "message must name rubberband"
    assert "apt install rubberband-cli" in str(exc)
check("missing rubberband CLI -> loud TimeStretchUnavailable with install", True)

import io as _sio
import contextlib
buf = _sio.StringIO()
with contextlib.redirect_stdout(buf):
    auto = StemMixer(sample_rate=SR)  # rubberband genuinely absent here
check("StemMixer auto-fallback uses ResampleStretch when rubberband absent",
      isinstance(auto.stretch, ResampleStretch))
check("auto-fallback prints its rehearsal-grade caveat loudly",
      "rehearsal-grade" in buf.getvalue().lower()
      and "pitch" in buf.getvalue().lower())

# -- CLI ---------------------------------------------------------------------------------------
def run_cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", "resonance.stems.cli", *argv],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
    )


r = run_cli("--help")
check("cli --help exits 0 and names resonance-stems",
      r.returncode == 0 and "resonance-stems" in r.stdout)

r = run_cli("status")
check("cli status exits 0 and reports honestly",
      r.returncode == 0 and "available" in r.stdout.lower()
      and "honest limit" in r.stdout.lower())

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for name, wave in STEM_WAVES.items():
        io.write_wav(tmp / f"{name}.wav", wave, SR)
    out = tmp / "mix.wav"
    r = run_cli("mix",
                "--stem", f"vocals={tmp / 'vocals.wav'}",
                "--stem", f"drums={tmp / 'drums.wav'}",
                "--volume", "vocals=0.5",
                "--mute", "drums",
                "--tempo", "vocals=2.0",
                "--out", str(out), "--sr", str(SR))
    assert r.returncode == 0, f"cli mix failed:\n{r.stdout}\n{r.stderr}"
    assert "BY DESIGN" in r.stdout, "cli must say the drift is by design"
    rendered, rsr = io.read_wav(out)
    expected_len = max(N // 2, N)  # vocals retimed to N/2, drums muted
    check("cli mix renders the mix WAV (vocals 0.5, drums muted, vocals 2x)",
          rsr == SR and rendered.shape == (2, expected_len))
    # bit-exact against the in-process mixer with the same settings
    m = make_mixer()
    m.set_volume("vocals", 0.5)
    m.mute("drums")
    m.mute("bass")
    m.mute("other")
    m.set_tempo("vocals", 2.0)
    wav16, _ = io.read_wav(out)  # 16-bit round-trip, as the CLI documents
    check("cli mix output matches the in-process mixer up to 16-bit WAV",
          np.allclose(wav16, m.mix(), atol=1.0 / 32767 + 1e-6))

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    song = tmp / "song.wav"
    io.write_wav(song, STEM_WAVES["vocals"], SR)
    r = run_cli("separate", str(song), "--out", str(tmp / "stems"))
    check("cli separate with no backend exits 3 naming torch on stderr",
          r.returncode == 3 and "torch" in r.stderr.lower())

print(f"{PASSED} stems tests passed.")
