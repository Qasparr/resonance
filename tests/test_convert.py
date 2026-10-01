# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_convert.py -- script-style tests for resonance.convert.

Run:  python3 tests/test_convert.py      (from the repo root)
   or python3 -m pytest tests/test_convert.py

Style (matching the other suites): each test prints "  ok: <name>"; the
end prints "<N> convert tests passed." Any failure raises immediately --
the first red line is the diagnosis.

Covers: probe_ffmpeg's loud failure when ffmpeg is absent (PATH
monkeypatched, shutil.which faked), successful probe parsing, the pure
build_convert_args argv builder (no binary needed), convert()'s
never-fake guarantees (missing/empty output -> ConvertError; failed
runs delete partial files; convert() itself never writes a file),
transcode() returning a real verified path, convert_many batch
semantics (stop on first failure), decode_to_pcm parsing real f32le
bytes (mocked subprocess) including corrupt-stream rejection, and the
CLI probe/list surfaces.

NO network, NO real ffmpeg execution in the mocked tests. One live
smoke test is included but SKIPS itself when no real ffmpeg is on PATH
-- it never fails for lack of the tool.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.convert import (
    ConvertError,
    FfmpegInfo,
    LoudMissingBackend,
    build_convert_args,
    convert,
    convert_many,
    decode_to_pcm,
    probe_ffmpeg,
    transcode,
)

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


def no_ffmpeg():
    """Context: shutil.which finds nothing (ffmpeg absent)."""
    return mock.patch.object(shutil, "which", return_value=None)


def fake_ffmpeg_at(path="/usr/bin/ffmpeg"):
    """Context: shutil.which finds ffmpeg at `path`."""
    return mock.patch.object(shutil, "which",
                             side_effect=lambda name: path
                             if name == "ffmpeg" else None)


class FakeCompleted:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def touch(path, size=128):
    Path(path).write_bytes(b"\x00" * size)
    return str(path)


# ---------------------------------------------------------------- probe

def t_probe_missing_is_loud():
    with no_ffmpeg():
        try:
            probe_ffmpeg()
        except LoudMissingBackend as exc:
            msg = str(exc)
            assert "ffmpeg" in msg
            assert "apt install ffmpeg" in msg, "must name the install"
            assert "never fakes a conversion" in msg
        else:
            raise AssertionError("absent ffmpeg must raise loudly")


def t_probe_success_parses_version():
    out = FakeCompleted(
        returncode=0,
        stdout="ffmpeg version 6.1.1 Copyright (c) 2000-2023\nbuilt with gcc",
    )
    with fake_ffmpeg_at(), mock.patch("subprocess.run", return_value=out):
        info = probe_ffmpeg()
    assert isinstance(info, FfmpegInfo)
    assert info.path == "/usr/bin/ffmpeg"
    assert info.version.startswith("ffmpeg version 6.1.1"), info.version


def t_probe_broken_install_is_loud():
    out = FakeCompleted(returncode=1, stdout="", stderr="boom")
    with fake_ffmpeg_at(), mock.patch("subprocess.run", return_value=out):
        try:
            probe_ffmpeg()
        except LoudMissingBackend as exc:
            assert "unusable" in str(exc)
        else:
            raise AssertionError("broken ffmpeg must raise loudly")


# ------------------------------------------------------- argv building

def t_build_args_full():
    argv = build_convert_args(
        "in.flac", "out.mp3", codec="libmp3lame", bitrate="192k",
        sample_rate=44100, channels=2, extra_args=("-q:a", "2"),
    )
    assert argv[0] == "-y", argv
    assert "-v" in argv and "error" in argv
    assert argv[argv.index("-i") + 1] == "in.flac"
    assert argv[argv.index("-c:a") + 1] == "libmp3lame"
    assert argv[argv.index("-b:a") + 1] == "192k"
    assert argv[argv.index("-ar") + 1] == "44100"
    assert argv[argv.index("-ac") + 1] == "2"
    assert argv[-3:] == ["-q:a", "2", "out.mp3"], argv[-3:]


def t_build_args_minimal():
    argv = build_convert_args("a.wav", "b.wav")
    assert argv == ["-y", "-v", "error", "-i", "a.wav", "b.wav"], argv


def t_build_args_rejects_bad_values():
    for kw in ({"channels": 3}, {"channels": 0},
               {"sample_rate": 0}, {"sample_rate": -1}):
        try:
            build_convert_args("a.wav", "b.wav", **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} must be rejected")


# ------------------------------------------------------- convert()

def _src_file(tmp):
    src = str(Path(tmp) / "in.wav")
    Path(src).write_bytes(b"RIFF" + b"\x00" * 100)  # existence, not validity
    return src


def t_convert_success_verifies_output():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        src = _src_file(tmp)
        dst = str(Path(tmp) / "out.mp3")
        seen = {}

        def fake_run(info, argv, timeout=600):
            seen["argv"] = argv
            seen["path"] = info.path
            touch(dst, size=256)  # the "ffmpeg" really wrote a file

        with fake_ffmpeg_at(), \
                mock.patch("resonance.convert.convert._run_ffmpeg",
                           side_effect=fake_run):
            out = convert(src, dst, codec="libmp3lame", bitrate="192k")
        assert out == dst
        assert Path(dst).stat().st_size == 256
        assert "-c:a" in seen["argv"] and "-b:a" in seen["argv"]
        assert seen["path"] == "/usr/bin/ffmpeg"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_convert_never_fakes_missing_output():
    """convert() must not declare success when ffmpeg wrote nothing."""
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        src = _src_file(tmp)
        dst = str(Path(tmp) / "out.mp3")
        with fake_ffmpeg_at(), \
                mock.patch("resonance.convert.convert._run_ffmpeg",
                           return_value=None):  # writes nothing
            try:
                convert(src, dst)
            except ConvertError as exc:
                assert "produced no file" in str(exc)
            else:
                raise AssertionError("missing output must raise")
        assert not Path(dst).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_convert_never_fakes_empty_output():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        src = _src_file(tmp)
        dst = str(Path(tmp) / "out.mp3")

        def fake_run(info, argv, timeout=600):
            Path(dst).write_bytes(b"")  # empty corpse

        with fake_ffmpeg_at(), \
                mock.patch("resonance.convert.convert._run_ffmpeg",
                           side_effect=fake_run):
            try:
                convert(src, dst)
            except ConvertError as exc:
                assert "EMPTY" in str(exc)
            else:
                raise AssertionError("empty output must raise")
        assert not Path(dst).exists(), "empty file must be deleted"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_convert_failure_deletes_partial():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        src = _src_file(tmp)
        dst = str(Path(tmp) / "out.mp3")

        def fake_run(info, argv, timeout=600):
            touch(dst, size=64)  # partial, then boom
            raise ConvertError("ffmpeg failed (exit 1)")

        with fake_ffmpeg_at(), \
                mock.patch("resonance.convert.convert._run_ffmpeg",
                           side_effect=fake_run):
            try:
                convert(src, dst)
            except ConvertError:
                pass
            else:
                raise AssertionError("ffmpeg failure must propagate")
        assert not Path(dst).exists(), "partial output must be deleted"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_convert_missing_src_is_loud():
    with fake_ffmpeg_at():
        with mock.patch("resonance.convert.convert._run_ffmpeg",
                        side_effect=AssertionError("must not run")):
            try:
                convert("/tmp/definitely-not-here-resonance.wav",
                        "/tmp/out.wav")
            except FileNotFoundError as exc:
                assert "not found" in str(exc)
            else:
                raise AssertionError("missing source must raise")


def t_convert_without_ffmpeg_is_loud():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        src = _src_file(tmp)
        with no_ffmpeg():
            try:
                convert(src, str(Path(tmp) / "out.mp3"))
            except LoudMissingBackend as exc:
                assert "ffmpeg" in str(exc)
            else:
                raise AssertionError("no-ffmpeg convert must raise loudly")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_transcode_returns_real_path():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        src = _src_file(tmp)
        dst = str(Path(tmp) / "out.ogg")

        def fake_run(info, argv, timeout=600):
            touch(dst, size=512)

        with fake_ffmpeg_at(), \
                mock.patch("resonance.convert.convert._run_ffmpeg",
                           side_effect=fake_run):
            out = transcode(src, dst, codec="libvorbis")
        assert out == dst and Path(out).stat().st_size == 512
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------- convert_many

def t_convert_many_batch():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        jobs = []
        for i in range(3):
            src = _src_file(tmp) + str(i)
            Path(src).write_bytes(b"x")
            dst = str(Path(tmp) / f"out{i}.mp3")
            jobs.append((src, dst))

        def fake_run(info, argv, timeout=600):
            touch(argv[-1], size=100)

        with fake_ffmpeg_at(), \
                mock.patch("resonance.convert.convert._run_ffmpeg",
                           side_effect=fake_run):
            results = convert_many(jobs, codec="libmp3lame")
        assert len(results) == 3
        for (src, dst), result in results:
            assert result == dst, (src, result)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_convert_many_dict_jobs_with_overrides():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        src = _src_file(tmp)
        dst = str(Path(tmp) / "out.mp3")
        seen = {}

        def fake_run(info, argv, timeout=600):
            seen["argv"] = argv
            touch(argv[-1], size=100)

        with fake_ffmpeg_at(), \
                mock.patch("resonance.convert.convert._run_ffmpeg",
                           side_effect=fake_run):
            results = convert_many(
                [{"src": src, "dst": dst, "bitrate": "320k"}],
                codec="libmp3lame",
            )
        assert results[0][1] == dst
        assert "320k" in seen["argv"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_convert_many_stops_on_first_failure():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        jobs = []
        for i in range(3):
            src = str(Path(tmp) / f"in{i}.wav")
            Path(src).write_bytes(b"x")
            jobs.append((src, str(Path(tmp) / f"out{i}.mp3")))
        calls = []

        def fake_run(info, argv, timeout=600):
            calls.append(argv[-1])
            if len(calls) == 2:
                raise ConvertError("boom on job 2")
            touch(argv[-1], size=100)

        with fake_ffmpeg_at(), \
                mock.patch("resonance.convert.convert._run_ffmpeg",
                           side_effect=fake_run):
            try:
                convert_many(jobs)
            except ConvertError:
                pass
            else:
                raise AssertionError("batch failure must propagate")
        assert len(calls) == 2, "batch must STOP at the first failure"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------- decode_to_pcm

def _f32le_bytes(samples):
    return np.asarray(samples, dtype=np.float32).astype("<f4").tobytes()


def t_decode_to_pcm_stereo():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        src = str(Path(tmp) / "in.mp3")
        Path(src).write_bytes(b"fake")
        # interleaved stereo: L=[0.5, -0.5], R=[0.25, -0.25]
        raw = _f32le_bytes([0.5, 0.25, -0.5, -0.25])
        stderr = "Stream #0:0: Audio: mp3, 48000 Hz, stereo"
        completed = FakeCompleted(returncode=0, stdout=raw, stderr=stderr)

        def fake_run(cmd, **kw):
            if "-version" in cmd:  # probe_ffmpeg's own version check
                return FakeCompleted(returncode=0,
                                     stdout="ffmpeg version 6.1.1\n")
            if "-hide_banner" in cmd:  # _sniff_rate's stream-line query
                return FakeCompleted(returncode=0, stdout="",
                                     stderr=stderr)
            assert "-acodec" in cmd and "pcm_f32le" in cmd
            assert cmd[-1] == "-"  # stdout streaming
            return completed

        with fake_ffmpeg_at(), mock.patch("subprocess.run", fake_run):
            buf, sr = decode_to_pcm(src)
        assert buf.shape == (2, 2), buf.shape
        assert buf.dtype == np.float32
        assert np.allclose(buf[0], [0.5, -0.5])
        assert np.allclose(buf[1], [0.25, -0.25])
        assert sr == 48000, sr
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_decode_to_pcm_mono_and_resample():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        src = str(Path(tmp) / "in.ogg")
        Path(src).write_bytes(b"fake")
        raw = _f32le_bytes([0.1, 0.2, 0.3])

        def fake_run(cmd, **kw):
            if "-version" in cmd:  # probe_ffmpeg's own version check
                return FakeCompleted(returncode=0,
                                     stdout="ffmpeg version 6.1.1\n")
            assert "-ar" in cmd and "22050" in cmd
            assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
            return FakeCompleted(returncode=0, stdout=raw, stderr="")

        with fake_ffmpeg_at(), mock.patch("subprocess.run", fake_run):
            buf, sr = decode_to_pcm(src, sample_rate=22050, channels=1)
        assert buf.shape == (3,) and sr == 22050
        assert np.allclose(buf, [0.1, 0.2, 0.3])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _decode_run(cmd, *, stdout=b"", stderr="", returncode=0,
                sniff_stderr="", **kw):
    """subprocess.run fake for decode tests: answers -version and the
    -hide_banner rate sniff, else the decode step."""
    if "-version" in cmd:  # probe_ffmpeg's own version check
        return FakeCompleted(returncode=0, stdout="ffmpeg version 6.1.1\n")
    if "-hide_banner" in cmd:  # _sniff_rate's stream-line query
        s = sniff_stderr
        return FakeCompleted(returncode=0, stdout="",
                             stderr=s.encode() if isinstance(s, str) else s)
    if isinstance(stderr, str):
        stderr = stderr.encode()
    return FakeCompleted(returncode=returncode, stdout=stdout, stderr=stderr)


def t_decode_to_pcm_rejects_corrupt():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        src = str(Path(tmp) / "in.mp3")
        Path(src).write_bytes(b"fake")
        with fake_ffmpeg_at():
            # not a multiple of 4 bytes
            bad = mock.patch("subprocess.run",
                             side_effect=lambda cmd, **kw: _decode_run(
                                 cmd, stdout=b"\x01\x02\x03", **kw))
            with bad:
                try:
                    decode_to_pcm(src)
                except ConvertError as exc:
                    assert "multiple of 4" in str(exc)
                else:
                    raise AssertionError("corrupt stream must raise")
            # zero bytes: nothing decoded
            empty = mock.patch("subprocess.run",
                               side_effect=lambda cmd, **kw: _decode_run(
                                   cmd, stdout=b"", **kw))
            with empty:
                try:
                    decode_to_pcm(src)
                except ConvertError as exc:
                    assert "zero audio bytes" in str(exc)
                else:
                    raise AssertionError("empty stream must raise")
            # ffmpeg itself fails on the decode step
            fail = mock.patch("subprocess.run",
                              side_effect=lambda cmd, **kw: _decode_run(
                                  cmd, stderr="Invalid data",
                                  returncode=0 if "-version" in cmd else 1,
                                  **kw))
            with fail:
                try:
                    decode_to_pcm(src)
                except ConvertError as exc:
                    assert "could not decode" in str(exc)
                else:
                    raise AssertionError("ffmpeg failure must raise")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_decode_to_pcm_missing_file():
    with fake_ffmpeg_at():
        try:
            decode_to_pcm("/tmp/definitely-not-here-resonance.mp3")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing file must raise")


def t_decode_to_pcm_no_ffmpeg_is_loud():
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        src = str(Path(tmp) / "in.mp3")
        Path(src).write_bytes(b"fake")
        with no_ffmpeg():
            try:
                decode_to_pcm(src)
            except LoudMissingBackend as exc:
                assert "ffmpeg" in str(exc)
            else:
                raise AssertionError("no-ffmpeg decode must raise loudly")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------- cli

def t_cli_probe_missing_ffmpeg():
    from resonance.convert.cli import main
    with no_ffmpeg():
        rc = main(["probe"])
    assert rc == 3, rc


def t_cli_probe_success():
    from resonance.convert.cli import main
    out = FakeCompleted(returncode=0, stdout="ffmpeg version 7.0\n")
    with fake_ffmpeg_at(), mock.patch("subprocess.run", return_value=out):
        rc = main(["probe"])
    assert rc == 0, rc


def t_cli_parse_jobfile():
    from resonance.convert.cli import parse_jobfile
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_")
    try:
        jf = Path(tmp) / "jobs.txt"
        jf.write_text("# comment\n\na.wav -> b.mp3\nc.flac d.ogg\n")
        jobs = parse_jobfile(str(jf))
        assert jobs == [("a.wav", "b.mp3"), ("c.flac", "d.ogg")], jobs
        jf.write_text("garbage line with three tokens here\n")
        try:
            parse_jobfile(str(jf))
        except ValueError as exc:
            assert "expected 'src -> dst'" in str(exc)
        else:
            raise AssertionError("bad job line must raise")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def t_live_ffmpeg_smoke():
    """Real end-to-end conversion -- SKIPS itself without a real ffmpeg."""
    if shutil.which("ffmpeg") is None:
        print("    (skip: no real ffmpeg on PATH)")
        return
    tmp = tempfile.mkdtemp(prefix="resonance_test_convert_live_")
    try:
        # generate a real 1s 440Hz tone with the REAL ffmpeg
        src = str(Path(tmp) / "tone.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=1",
             "-ar", "44100", "-ac", "1", src],
            check=True, timeout=60,
        )
        dst = str(Path(tmp) / "tone.mp3")
        out = convert(src, dst, codec="libmp3lame", bitrate="128k")
        assert Path(out).stat().st_size > 0
        buf, sr = decode_to_pcm(dst, channels=1)
        assert buf.shape[0] > 40000, buf.shape  # ~1s of audio, real
        assert sr == 44100, sr
        # the decoded tone is really ~440 Hz (FFT check, honest)
        spectrum = np.abs(np.fft.rfft(buf))
        peak = np.argmax(spectrum[1:]) + 1
        freq = peak * sr / buf.shape[0]
        assert abs(freq - 440) < 15, f"decoded tone peaks at {freq}Hz"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS = [
    ("probe missing ffmpeg is loud", t_probe_missing_is_loud),
    ("probe success parses version", t_probe_success_parses_version),
    ("probe broken install is loud", t_probe_broken_install_is_loud),
    ("argv builder: full options", t_build_args_full),
    ("argv builder: minimal", t_build_args_minimal),
    ("argv builder rejects bad values", t_build_args_rejects_bad_values),
    ("convert success verifies output", t_convert_success_verifies_output),
    ("convert never fakes missing output", t_convert_never_fakes_missing_output),
    ("convert never fakes empty output", t_convert_never_fakes_empty_output),
    ("convert failure deletes partial file", t_convert_failure_deletes_partial),
    ("convert missing source is loud", t_convert_missing_src_is_loud),
    ("convert without ffmpeg is loud", t_convert_without_ffmpeg_is_loud),
    ("transcode returns real path", t_transcode_returns_real_path),
    ("convert_many batch", t_convert_many_batch),
    ("convert_many dict jobs + overrides", t_convert_many_dict_jobs_with_overrides),
    ("convert_many stops on first failure", t_convert_many_stops_on_first_failure),
    ("decode_to_pcm stereo parse", t_decode_to_pcm_stereo),
    ("decode_to_pcm mono + resample args", t_decode_to_pcm_mono_and_resample),
    ("decode_to_pcm rejects corrupt streams", t_decode_to_pcm_rejects_corrupt),
    ("decode_to_pcm missing file", t_decode_to_pcm_missing_file),
    ("decode_to_pcm without ffmpeg is loud", t_decode_to_pcm_no_ffmpeg_is_loud),
    ("cli probe missing ffmpeg -> 3", t_cli_probe_missing_ffmpeg),
    ("cli probe success -> 0", t_cli_probe_success),
    ("cli jobfile parsing", t_cli_parse_jobfile),
    ("live ffmpeg smoke (skips if absent)", t_live_ffmpeg_smoke),
]


def main():
    for name, fn in TESTS:
        check(name, fn)
    print(f"{PASSED} convert tests passed.")


if __name__ == "__main__":
    main()
