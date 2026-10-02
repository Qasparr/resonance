# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_ai.py -- script-style tests for resonance.ai (The AI Wing).

Run:  python3 tests/test_ai.py        (from the repo root)
   or python3 -m pytest tests/test_ai.py

Style: each test prints "  ok: <name>"; the end prints
"<N> ai tests passed." Any failure raises immediately.

Ground truth is synthetic and exact: sines with known RMS/peak/
crest, canned Gemini response payloads (NO network in this suite
-- the key-resolution and parsing paths are tested, the wire is
not), real DSP checks on the NL executor (darken must measurably
attenuate treble), real music-theory checks on the composer.
"""
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.ai.adapters import (
    BackendProbe,
    LocalModelAdapter,
    ModelNotAvailable,
)
from resonance.ai.analyze import analyze_track, format_report
from resonance.ai.command import (
    Intent,
    ai_parse,
    execute_intent,
    parse_command,
)
from resonance.ai.compose import (
    chord_progression,
    drum_pattern,
    melody_from_chords,
    progression_to_text,
)
from resonance.ai.gemini import (
    GeminiClient,
    GeminiError,
    GeminiKeyMissing,
    build_request_body,
    parse_response,
    resolve_key,
)
from resonance.ai.generate import MusicGenAdapter, arrange_sections
from resonance.ai.master import auto_chain, build_advice_prompt, advise
from resonance.ai.transcribe import WhisperAdapter
from resonance.core.io import write_wav

PASSED = 0
SR = 44100


def check(name, fn):
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


def sine(freq, amp, dur_s=1.0, sr=SR):
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# -- analyze -----------------------------------------------------------

def test_analyze_sine_ground_truth():
    a = sine(440.0, 0.5)
    r = analyze_track(a, SR)
    assert abs(r.rms - 0.5 / np.sqrt(2)) < 1e-3, r.rms
    assert abs(r.peak - 0.5) < 1e-3, r.peak
    assert abs(r.crest_factor - np.sqrt(2)) < 0.05, r.crest_factor
    assert abs(r.spectral_centroid_hz - 440.0) < 5.0, r.spectral_centroid_hz
    assert r.clipped_samples == 0
    assert r.channels == 1
    assert abs(r.duration_s - 1.0) < 1e-6


def test_analyze_clipping_detected():
    a = sine(440.0, 1.5)  # hot: pins past full scale
    r = analyze_track(a, SR)
    assert r.clipped_samples > 0, "hot sine must show clipping"
    assert any("clipped" in n for n in r.notes)


def test_analyze_stereo_width():
    s = sine(440.0, 0.5)
    dual_mono = np.stack([s, s])
    assert analyze_track(dual_mono, SR).stereo_width < 1e-6
    wide = np.stack([s, 0.5 * s])
    w = analyze_track(wide, SR).stereo_width
    assert abs(w - 1.0 / 3.0) < 0.02, w  # side/mid = 0.25/0.75


def test_analyze_rejects_empty_and_bad_sr():
    try:
        analyze_track(np.zeros(0, dtype=np.float32), SR)
    except ValueError:
        pass
    else:
        raise AssertionError("empty audio must raise")
    try:
        analyze_track(sine(440.0, 0.5), 0)
    except ValueError:
        pass
    else:
        raise AssertionError("sr=0 must raise")


def test_format_report_labels_rms_not_lufs():
    r = analyze_track(sine(440.0, 0.5), SR)
    text = format_report(r)
    assert "NOT LUFS" in text
    assert "crest" in text


# -- gemini client (no network) ----------------------------------------

def test_resolve_key_missing_names_env():
    env = "RESONANCE_TEST_KEY_ABSENT_XYZ"
    os.environ.pop(env, None)
    try:
        resolve_key(None, env_var=env)
    except GeminiKeyMissing as exc:
        assert env in str(exc), str(exc)
    else:
        raise AssertionError("missing key must raise")


def test_resolve_key_explicit_and_env():
    assert resolve_key("k-explicit") == "k-explicit"
    os.environ["RESONANCE_TEST_KEY_XYZ"] = "k-env"
    try:
        assert resolve_key(None, env_var="RESONANCE_TEST_KEY_XYZ") == "k-env"
        # explicit wins over env
        assert resolve_key("k-win", env_var="RESONANCE_TEST_KEY_XYZ") == "k-win"
    finally:
        del os.environ["RESONANCE_TEST_KEY_XYZ"]


def test_build_request_body_thinking_and_tools():
    body = build_request_body("hi", think_budget=4096, code_execution=True)
    assert body["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 4096
    assert body["tools"] == [{"codeExecution": {}}]
    plain = build_request_body("hi")
    assert "thinkingConfig" not in plain["generationConfig"]
    assert "tools" not in plain


def _canned_payload():
    return {
        "candidates": [{
            "content": {"parts": [
                {"thought": True, "text": "internal reasoning"},
                {"text": "Hello "},
                {"executableCode": {"language": "PYTHON", "code": "1+1"}},
                {"codeExecutionResult": {"outcome": "OUTCOME_OK",
                                         "output": "2"}},
                {"text": "world"},
            ]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 5,
                          "candidatesTokenCount": 3},
    }


def test_parse_response_skips_thoughts_reports_code():
    res = parse_response(_canned_payload())
    assert res.text == "Hello world", res.text
    assert res.code_executed is True
    assert res.usage["promptTokenCount"] == 5


def test_parse_response_empty_raises():
    try:
        parse_response({"candidates": []})
    except GeminiError:
        pass
    else:
        raise AssertionError("empty candidates must raise")


def test_client_construction_without_key_raises_before_network():
    env = "RESONANCE_TEST_KEY_ABSENT_XYZ"
    os.environ.pop(env, None)
    try:
        GeminiClient(api_key=None, key_env=env)
    except GeminiKeyMissing:
        pass
    else:
        raise AssertionError("client without key must raise (no I/O)")


# -- master ------------------------------------------------------------

def test_auto_chain_clipped_gets_limiter():
    r = analyze_track(sine(440.0, 1.5), SR)
    effects, reasons = auto_chain(r)
    assert any(e["type"] == "limiter" for e in effects), effects
    assert any("clipped" in w for w in reasons)


def test_auto_chain_faint_gets_normalize():
    r = analyze_track(sine(440.0, 0.01), SR)  # ~ -43 dB RMS
    effects, reasons = auto_chain(r)
    assert any(e["type"] == "normalize" for e in effects), effects


def test_auto_chain_clean_is_empty_and_says_so():
    r = analyze_track(sine(440.0, 0.5), SR)
    effects, reasons = auto_chain(r)
    assert effects == []
    assert any("clean" in w for w in reasons), reasons


def test_advice_prompt_carries_goal_and_measurements():
    r = analyze_track(sine(440.0, 0.5), SR)
    p = build_advice_prompt(r, "make it slam")
    assert "make it slam" in p
    assert "MEASURED" in p
    assert "crest" in p


def test_advise_without_key_raises_loudly():
    r = analyze_track(sine(440.0, 0.5), SR)
    env = "RESONANCE_TEST_KEY_ABSENT_XYZ"
    os.environ.pop(env, None)
    old = os.environ.get("GEMINI_API_KEY")
    os.environ.pop("GEMINI_API_KEY", None)
    try:
        advise(r, "goal", api_key=None)
    except GeminiKeyMissing:
        pass
    else:
        raise AssertionError("advise without key must raise")
    finally:
        if old is not None:
            os.environ["GEMINI_API_KEY"] = old


# -- command: local parser ---------------------------------------------

def test_parse_darken():
    i = parse_command("make it darker")
    assert i.action == "darken" and i.known
    assert i.params["freq"] == 4000.0
    assert i.confidence > 0.9


def test_parse_tighten_timing_with_bpm():
    i = parse_command("tighten the timing at 120 bpm")
    assert i.action == "tighten_timing"
    assert i.params["bpm"] == 120.0
    assert i.confidence > 0.8


def test_parse_tighten_timing_without_bpm_is_hedged():
    i = parse_command("tighten the timing")
    assert i.action == "tighten_timing"
    assert i.params["bpm"] is None
    assert i.confidence < 0.9  # hedged: BPM missing


def test_parse_unknown_is_not_a_guess():
    i = parse_command("do a barrel roll")
    assert i.action == "unknown" and not i.known
    assert i.confidence == 0.0


def test_execute_darken_really_attenuates_treble():
    lo = sine(200.0, 0.4, dur_s=2.0)
    hi = sine(8000.0, 0.4, dur_s=2.0)
    mix = (lo + hi).astype(np.float32)
    out, report = execute_intent(parse_command("make it darker"), mix, SR)
    assert "lowpass" in report
    # Measure 8 kHz bin before/after (skip filter start-up transient).
    spec_in = np.abs(np.fft.rfft(mix[SR // 2:]))
    spec_out = np.abs(np.fft.rfft(out[SR // 2:]))
    freqs = np.fft.rfftfreq(len(mix) - SR // 2, d=1.0 / SR)
    k = int(np.argmin(np.abs(freqs - 8000.0)))
    assert spec_out[k] < 0.5 * spec_in[k], "treble must drop"


def test_execute_louder_doubles_rms():
    a = sine(440.0, 0.25)
    out, report = execute_intent(parse_command("louder"), a, SR)
    assert "+6 dB" in report
    r_in = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    r_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert abs(r_out / r_in - 2.0) < 0.05, (r_in, r_out)


def test_execute_unknown_refuses():
    try:
        execute_intent(Intent(action="unknown"), sine(440.0, 0.5), SR)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown intent must be refused")


def test_execute_tighten_without_bpm_refuses():
    i = parse_command("tighten the timing")
    try:
        execute_intent(i, sine(440.0, 0.5), SR)
    except ValueError as exc:
        assert "BPM" in str(exc)
    else:
        raise AssertionError("tighten without BPM must be refused")


# -- compose -----------------------------------------------------------

def test_chord_progression_hopeful_in_c():
    prog = chord_progression("C", "hopeful")
    assert [c["name"] for c in prog] == ["C", "G", "Am", "F"], prog
    assert prog[0]["midi"] == [48, 52, 55]
    assert prog[2]["midi"] == [57, 60, 64]  # Am


def test_chord_progression_rejects_bad_inputs():
    for fn in (lambda: chord_progression("C", "mystical"),
               lambda: chord_progression("H", "hopeful")):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("bad compose input must raise")


def test_melody_is_deterministic_and_sane():
    prog = chord_progression("C", "hopeful")
    m1 = melody_from_chords(prog, seed=7)
    m2 = melody_from_chords(prog, seed=7)
    assert m1 == m2 and len(m1) == 16
    assert all(24 <= m <= 96 for m in m1)
    assert "Am" in progression_to_text(prog)


def test_drum_pattern_renders_and_labels():
    audio, grid = drum_pattern("boom_bap", bars=1, bpm=90.0)
    assert audio.dtype == np.float32
    assert float(np.max(np.abs(audio))) <= 0.891
    assert "boom_bap" in grid and "bar 1" in grid
    try:
        drum_pattern("polka")
    except ValueError:
        pass
    else:
        raise AssertionError("bad style must raise")


# -- adapters: honest loud skip ----------------------------------------

def test_probe_reports_missing_torch_honestly():
    probe = BackendProbe(extra_modules=("transformers",),
                         weights_env="RESONANCE_TEST_W_XYZ")
    rep = probe.report()
    assert set(rep) >= {"modules", "weights_present", "cuda", "ready"}
    assert rep["ready"] is False  # this env has no torch/weights
    missing = probe.missing_pieces()
    assert any("torch" in line for line in missing), missing


def test_adapters_refuse_without_backend():
    for adapter, call in (
            (MusicGenAdapter(), lambda a: a.generate("x", 1.0)),
            (WhisperAdapter(), lambda a: a.transcribe(np.zeros(8), 8000))):
        assert adapter.available() is False
        try:
            call(adapter)
        except ModelNotAvailable as exc:
            assert "not available" in str(exc)
        else:
            raise AssertionError("adapter must refuse without backend")


def test_arrange_sections_joins_with_crossfades():
    s1 = sine(440.0, 0.5, dur_s=2.0)
    s2 = sine(660.0, 0.5, dur_s=2.0)
    joined, prov = arrange_sections(
        {"intro": s1, "outro": s2}, ["intro", "outro"],
        sr=SR, crossfade_s=0.5)
    n_fade = int(0.5 * SR)
    assert len(joined) == len(s1) + len(s2) - n_fade
    assert prov["order"] == ["intro", "outro"]
    assert prov["labeled"] == "generated"
    try:
        arrange_sections({"a": s1}, [])
    except ValueError:
        pass
    else:
        raise AssertionError("empty order must raise")
    try:
        arrange_sections({"a": s1}, ["zzz"])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown section must raise")


# -- CLI ---------------------------------------------------------------

ROOT = str(Path(__file__).resolve().parent.parent)


def run_cli(*args, cwd=None, env=None):
    e = dict(os.environ)
    e["PYTHONPATH"] = ROOT
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, "-m", "resonance.ai.cli", *args],
        capture_output=True, text=True, cwd=cwd or ROOT, env=e)


def test_cli_help():
    r = run_cli("--help")
    assert r.returncode == 0 and "resonance-ai" in r.stdout


def test_cli_probe_reports_backends():
    r = run_cli("probe")
    assert r.returncode == 0
    assert "musicgen" in r.stdout and "whisper" in r.stdout


def test_cli_analyze_and_do_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        dst = os.path.join(tmp, "out.wav")
        write_wav(src, sine(440.0, 0.5, dur_s=1.0), SR)
        r = run_cli("analyze", src)
        assert r.returncode == 0 and "crest" in r.stdout, r.stderr
        r = run_cli("do", "make it darker", "--in", src, "--out", dst)
        assert r.returncode == 0, r.stderr
        assert "lowpass" in r.stdout
        with open(dst, "rb") as fh:
            assert fh.read(4) == b"RIFF"


def test_cli_do_unknown_is_exit_2():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        dst = os.path.join(tmp, "out.wav")
        write_wav(src, sine(440.0, 0.5, dur_s=0.5), SR)
        r = run_cli("do", "do a barrel roll", "--in", src, "--out", dst)
        assert r.returncode == 2, (r.returncode, r.stdout)


def test_cli_advise_without_key_is_exit_3():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        write_wav(src, sine(440.0, 0.5, dur_s=0.5), SR)
        # ensure no key leaks in from the real environment
        r = run_cli("advise", src, "--goal", "test",
                    env={"GEMINI_API_KEY": "", "PATH": os.environ["PATH"],
                         "PYTHONPATH": ROOT, "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
        assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)


def test_cli_compose_offline():
    r = run_cli("compose", "chords", "--key", "C", "--mood", "hopeful")
    assert r.returncode == 0 and "Am" in r.stdout, r.stderr
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "drums.wav")
        r = run_cli("compose", "drums", "--style", "boom_bap",
                    "--bars", "1", "--out", dst)
        assert r.returncode == 0, r.stderr
        assert os.path.exists(dst)


if __name__ == "__main__":
    check("analyze sine ground truth", test_analyze_sine_ground_truth)
    check("analyze clipping detected", test_analyze_clipping_detected)
    check("analyze stereo width", test_analyze_stereo_width)
    check("analyze rejects empty/bad sr", test_analyze_rejects_empty_and_bad_sr)
    check("report labels RMS not LUFS", test_format_report_labels_rms_not_lufs)
    check("resolve_key missing names env", test_resolve_key_missing_names_env)
    check("resolve_key explicit+env", test_resolve_key_explicit_and_env)
    check("request body thinking+tools", test_build_request_body_thinking_and_tools)
    check("parse_response canned payload", test_parse_response_skips_thoughts_reports_code)
    check("parse_response empty raises", test_parse_response_empty_raises)
    check("client w/o key raises pre-network",
          test_client_construction_without_key_raises_before_network)
    check("auto_chain clipped->limiter", test_auto_chain_clipped_gets_limiter)
    check("auto_chain faint->normalize", test_auto_chain_faint_gets_normalize)
    check("auto_chain clean empty+says so", test_auto_chain_clean_is_empty_and_says_so)
    check("advice prompt carries goal", test_advice_prompt_carries_goal_and_measurements)
    check("advise w/o key raises loudly", test_advise_without_key_raises_loudly)
    check("parse 'make it darker'", test_parse_darken)
    check("parse tighten w/ bpm", test_parse_tighten_timing_with_bpm)
    check("parse tighten w/o bpm hedged", test_parse_tighten_timing_without_bpm_is_hedged)
    check("parse unknown not a guess", test_parse_unknown_is_not_a_guess)
    check("execute darken attenuates treble",
          test_execute_darken_really_attenuates_treble)
    check("execute louder doubles rms", test_execute_louder_doubles_rms)
    check("execute unknown refused", test_execute_unknown_refuses)
    check("execute tighten w/o bpm refused",
          test_execute_tighten_without_bpm_refuses)
    check("chords hopeful in C", test_chord_progression_hopeful_in_c)
    check("chords reject bad inputs", test_chord_progression_rejects_bad_inputs)
    check("melody deterministic+sane", test_melody_is_deterministic_and_sane)
    check("drum pattern renders+labeled", test_drum_pattern_renders_and_labels)
    check("probe honest about torch", test_probe_reports_missing_torch_honestly)
    check("adapters refuse w/o backend", test_adapters_refuse_without_backend)
    check("arrange_sections joins", test_arrange_sections_joins_with_crossfades)
    check("cli --help", test_cli_help)
    check("cli probe", test_cli_probe_reports_backends)
    check("cli analyze+do roundtrip", test_cli_analyze_and_do_roundtrip)
    check("cli do unknown exit 2", test_cli_do_unknown_is_exit_2)
    check("cli advise w/o key exit 3", test_cli_advise_without_key_is_exit_3)
    check("cli compose offline", test_cli_compose_offline)
    print(f"\n{PASSED} ai tests passed.")
