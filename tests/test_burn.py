# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_burn.py -- script-style tests for resonance.burn.

Run:  python3 tests/test_burn.py         (from the repo root)
   or python3 -m pytest tests/test_burn.py

Style: each test prints "  ok: <name>"; the end prints
"<N> burn tests passed." Any failure raises immediately.

The honest-testing rule for this module: NO real drive is ever
touched. External tools are faked two ways -- (1) a fake `runner`
that records argv instead of executing, and (2) fake tool binaries
on PATH (empty executable files in a tmp bin dir) so the probing
layer sees "installed" tools. Command assembly is what's tested;
a fake burn is never presented as a real one.
"""
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.burn import (
    BACKEND_TOOLS,
    BurnBackendError,
    BurnError,
    author_dvd,
    burn_audio_cd,
    burn_data,
    burn_dvd,
    decode_to_cd_wav,
    encode_mpeg2,
    make_dvd_iso,
    make_iso,
    probe_backends,
    require_any_backend,
    require_backend,
    run_tool,
    verify_iso,
    verify_video_ts,
    write_cue,
    write_dvdauthor_xml,
)

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


class FakeRunner:
    """Records argv, executes nothing. Returns success."""

    def __init__(self, returncode=0, stderr=""):
        self.commands = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, argv, **kwargs):
        self.commands.append(list(argv))

        class P:
            pass
        p = P()
        p.returncode = self.returncode
        p.stdout = ""
        p.stderr = self.stderr
        return p


def fake_bin_dir(*tools):
    """A tmp dir on PATH containing fake executable `tools`.

    Returns (tmpdir, old_path); the caller restores PATH. shutil.which
    then "finds" the tools -- probing sees them installed, but they
    are empty files that never run (the fake runner intercepts).
    """
    td = tempfile.mkdtemp(prefix="resonance-fakebin-")
    for tool in tools:
        p = os.path.join(td, tool)
        with open(p, "w") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    return td


def t_probe_table_shape():
    report = probe_backends()
    for name in ("ffmpeg", "dvdauthor", "genisoimage", "xorriso",
                 "growisofs", "wodim", "cdrdao"):
        assert name in report, f"{name} missing from probe table"
        info = report[name]
        for key in ("available", "path", "purpose", "install_hint"):
            assert key in info, f"{name} missing {key}"
        assert isinstance(info["available"], bool)


def t_require_unknown_backend():
    try:
        require_backend("definitely-not-a-real-tool")
    except BurnBackendError as exc:
        assert "unknown backend" in str(exc)
        return
    raise AssertionError("unknown backend must raise BurnBackendError")


def t_require_missing_names_tool_and_hint():
    td = fake_bin_dir()  # empty PATH dir: nothing installed
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td
    try:
        require_backend("ffmpeg")
    except BurnBackendError as exc:
        msg = str(exc)
        assert "ffmpeg" in msg and "install" in msg.lower(), msg
        return
    finally:
        os.environ["PATH"] = old
    raise AssertionError("missing ffmpeg must raise BurnBackendError")


def t_require_any_backend_picks_first_available():
    td = fake_bin_dir("xorriso")
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td + os.pathsep + old
    try:
        name, path = require_any_backend("genisoimage", "xorriso")
        assert name == "xorriso", f"want xorriso, got {name}"
        assert path.endswith("xorriso")
    finally:
        os.environ["PATH"] = old


def t_require_any_backend_all_missing_loud():
    td = fake_bin_dir()
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td
    try:
        require_any_backend("genisoimage", "xorriso")
    except BurnBackendError as exc:
        assert "none of the candidate backends" in str(exc)
        return
    finally:
        os.environ["PATH"] = old
    raise AssertionError("all-missing must raise BurnBackendError")


def t_run_tool_failure_loud():
    r = FakeRunner(returncode=2, stderr="boom\nline2")
    try:
        run_tool(["wodim", "--badflag"], runner=r)
    except BurnError as exc:
        assert "wodim" in str(exc) and "boom" in str(exc)
        return
    raise AssertionError("nonzero exit must raise BurnError")


def t_decode_assembles_ffmpeg_redbook():
    td = fake_bin_dir("ffmpeg")
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td + os.pathsep + old
    try:
        r = FakeRunner()
        with tempfile.TemporaryDirectory() as work:
            src = os.path.join(work, "song.mp3")
            open(src, "wb").write(b"fake")
            wavs = decode_to_cd_wav([src], os.path.join(work, "out"),
                                    runner=r)
        assert len(wavs) == 1 and wavs[0].endswith("track01.wav")
        cmd = r.commands[0]
        assert cmd[0] == "ffmpeg"
        for flag in ("-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le"):
            assert flag in cmd, f"{flag} missing from {cmd}"
    finally:
        os.environ["PATH"] = old


def t_write_cue_structure_and_cdtext():
    with tempfile.TemporaryDirectory() as td:
        cue = os.path.join(td, "disc.cue")
        tracks = [
            {"file": "/tmp/track01.wav", "title": "First Song",
             "artist": "The Artist"},
            {"file": "/tmp/track02.wav", "title": "Second Song"},
        ]
        write_cue(tracks, cue, album_title="The Album",
                  album_artist="The Artist")
        text = open(cue).read()
        assert 'TITLE "The Album"' in text
        assert 'PERFORMER "The Artist"' in text
        assert 'FILE "/tmp/track01.wav" WAVE' in text
        assert "TRACK 01 AUDIO" in text and "TRACK 02 AUDIO" in text
        assert 'TITLE "First Song"' in text
        assert "INDEX 01 00:00:00" in text
        # FILE must precede its TRACK (standard cue layout).
        assert text.index('FILE "/tmp/track01.wav"') < \
            text.index("TRACK 01 AUDIO")


def t_burn_audio_cd_wodim_argv():
    td = fake_bin_dir("wodim")
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td + os.pathsep + old
    try:
        r = FakeRunner()
        argv = burn_audio_cd(["/tmp/a.wav", "/tmp/b.wav"], "/dev/sr0",
                             cue_path="/tmp/disc.cue", backend="wodim",
                             runner=r)
        assert argv[0] == "wodim" and "dev=/dev/sr0" in argv
        assert "-audio" in argv and "-text" in argv
        assert argv[-2:] == ["/tmp/a.wav", "/tmp/b.wav"]
    finally:
        os.environ["PATH"] = old


def t_burn_audio_cd_cdrdao_needs_cue():
    td = fake_bin_dir("cdrdao")
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td + os.pathsep + old
    try:
        try:
            burn_audio_cd(["/tmp/a.wav"], "/dev/sr0", backend="cdrdao",
                          runner=FakeRunner())
        except ValueError as exc:
            assert "cue_path" in str(exc)
            return
        raise AssertionError("cdrdao without cue must raise ValueError")
    finally:
        os.environ["PATH"] = old


def t_burn_audio_cd_missing_backend_exit3_shape():
    td = fake_bin_dir()  # nothing installed
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td
    try:
        burn_audio_cd(["/tmp/a.wav"], "/dev/sr0", backend="wodim",
                      runner=FakeRunner())
    except BurnBackendError as exc:
        assert "wodim" in str(exc)
        return
    finally:
        os.environ["PATH"] = old
    raise AssertionError("missing wodim must raise BurnBackendError")


def t_make_iso_genisoimage_argv():
    td = fake_bin_dir("genisoimage")
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td + os.pathsep + old
    try:
        r = FakeRunner()
        with tempfile.TemporaryDirectory() as work:
            iso = os.path.join(work, "out.iso")
            got, tool = make_iso([work], iso, volume_id="TEST",
                                  runner=r)
        assert got == iso and tool == "genisoimage"
        cmd = r.commands[0]
        assert cmd[:3] == ["genisoimage", "-o", iso]
        for flag in ("-J", "-r", "-V", "TEST"):
            assert flag in cmd
    finally:
        os.environ["PATH"] = old


def t_make_iso_xorriso_fallback():
    td = fake_bin_dir("xorriso")  # genisoimage absent, xorriso present
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td + os.pathsep + old
    try:
        r = FakeRunner()
        with tempfile.TemporaryDirectory() as work:
            iso = os.path.join(work, "out.iso")
            got, tool = make_iso([work], iso, runner=r)
        assert tool == "xorriso"
        assert r.commands[0][:3] == ["xorriso", "-as", "mkisofs"]
    finally:
        os.environ["PATH"] = old


def t_burn_data_dvd_and_cd_argv():
    td = fake_bin_dir("growisofs", "wodim")
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td + os.pathsep + old
    try:
        r = FakeRunner()
        argv = burn_data("/tmp/x.iso", "/dev/sr0", media="dvd",
                         runner=r)
        assert argv[0] == "growisofs" and "-dvd-compat" in argv
        assert argv[-1] == "/dev/sr0=/tmp/x.iso"
        argv = burn_data("/tmp/x.iso", "/dev/sr0", media="cd",
                         runner=r)
        assert argv[0] == "wodim" and "dev=/dev/sr0" in argv
        try:
            burn_data("/tmp/x.iso", "/dev/sr0", media="bluray",
                      runner=r)
        except ValueError:
            pass
        else:
            raise AssertionError("bad media must raise ValueError")
    finally:
        os.environ["PATH"] = old


def t_verify_iso_magic():
    with tempfile.TemporaryDirectory() as td:
        good = os.path.join(td, "good.iso")
        with open(good, "wb") as fh:
            fh.write(b"\x00" * 16 * 2048)
            fh.write(b"\x01CD001\x01" + b"\x00" * 2041)
        assert verify_iso(good) is True
        bad = os.path.join(td, "bad.iso")
        with open(bad, "wb") as fh:
            fh.write(b"\x00" * 40000)
        assert verify_iso(bad) is False
        assert verify_iso(os.path.join(td, "missing.iso")) is False


def t_encode_mpeg2_assembles_ffmpeg():
    td = fake_bin_dir("ffmpeg")
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td + os.pathsep + old
    try:
        r = FakeRunner()
        with tempfile.TemporaryDirectory() as work:
            src = os.path.join(work, "clip.mp4")
            open(src, "wb").write(b"fake")
            mpgs = encode_mpeg2([src], work, standard="pal", runner=r)
        assert mpgs[0].endswith("title01.mpg")
        cmd = r.commands[0]
        assert "-target" in cmd and "pal-dvd" in cmd
        try:
            encode_mpeg2([src], work, standard="vhs", runner=r)
        except ValueError:
            pass
        else:
            raise AssertionError("bad standard must raise ValueError")
    finally:
        os.environ["PATH"] = old


def t_dvdauthor_xml_valid_and_honest():
    import xml.etree.ElementTree as ET
    with tempfile.TemporaryDirectory() as td:
        xml = os.path.join(td, "dvd.xml")
        write_dvdauthor_xml(
            ["/tmp/title01.mpg", "/tmp/title02.mpg"], xml,
            chapters=[["0", "5:00"], ["0"]],
            menu_image="/tmp/menu.png",
            titles=["Intro", "Main"])
        text = open(xml).read()
        # Well-formed XML (parse check) ...
        ET.fromstring(text.replace('dest="__VIDEO_TS__"', 'dest="x"'))
        # ... with chapters, menu buttons, and no motion-menu claims.
        assert "<chapters>0 5:00</chapters>" in text
        assert 'name="Intro"' in text and "jump titleset 1 title 2" in text
        assert "menu.png" in text


def t_verify_video_ts():
    with tempfile.TemporaryDirectory() as td:
        vts = os.path.join(td, "VIDEO_TS")
        os.makedirs(vts)
        assert verify_video_ts(td) is False  # nothing in it yet
        open(os.path.join(vts, "VIDEO_TS.IFO"), "wb").write(b"x")
        assert verify_video_ts(td) is False  # IFO but no VOB
        open(os.path.join(vts, "VTS_01_1.VOB"), "wb").write(b"x")
        assert verify_video_ts(td) is True


def t_burn_dvd_argv():
    td = fake_bin_dir("growisofs")
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td + os.pathsep + old
    try:
        r = FakeRunner()
        argv = burn_dvd("/tmp/dvd.iso", "/dev/sr0", runner=r)
        assert argv == ["growisofs", "-dvd-compat", "-Z",
                        "/dev/sr0=/tmp/dvd.iso"]
    finally:
        os.environ["PATH"] = old


def t_make_dvd_iso_argv():
    td = fake_bin_dir("genisoimage")
    old = os.environ.get("PATH", "")
    os.environ["PATH"] = td + os.pathsep + old
    try:
        from resonance.burn.dvd import make_dvd_iso
        r = FakeRunner()
        iso, tool = make_dvd_iso("/tmp/vts", "/tmp/dvd.iso", runner=r)
        assert tool == "genisoimage"
        assert "-dvd-video" in r.commands[0]
    finally:
        os.environ["PATH"] = old


def t_cli_backends_always_works():
    r = subprocess.run(
        [sys.executable, "-m", "resonance.burn.cli", "backends"],
        capture_output=True, text=True, timeout=60, env=_env(),
        cwd=str(Path.cwd()))
    assert r.returncode == 0, r.stderr
    assert "ffmpeg" in r.stdout and "growisofs" in r.stdout


def t_cli_help():
    r = subprocess.run(
        [sys.executable, "-m", "resonance.burn.cli", "--help"],
        capture_output=True, text=True, timeout=60, env=_env(),
        cwd=str(Path.cwd()))
    assert r.returncode == 0 and "resonance-burn" in r.stdout


def t_cli_missing_backend_exit_3():
    # Empty PATH bin dir: ffmpeg missing -> loud refusal, exit 3.
    td = fake_bin_dir()
    with tempfile.TemporaryDirectory() as work:
        src = os.path.join(work, "song.wav")
        open(src, "wb").write(b"fake")
        env = _env()
        env["PATH"] = td  # nothing installed
        r = subprocess.run(
            [sys.executable, "-m", "resonance.burn.cli", "audio-cd",
             src, "--device", "/dev/sr0"],
            capture_output=True, text=True, timeout=120, env=env,
            cwd=str(Path.cwd()))
        assert r.returncode == 3, f"want exit 3, got {r.returncode}"
        assert "ffmpeg" in r.stderr and "install" in r.stderr.lower()


def t_cli_dry_run_prints_commands():
    # Fake tools on PATH: dry-run must print the WOULD-BE commands
    # and execute nothing (exit 0).
    td = fake_bin_dir("ffmpeg", "wodim")
    with tempfile.TemporaryDirectory() as work:
        src = os.path.join(work, "song.wav")
        open(src, "wb").write(b"RIFFfake")
        env = _env()
        env["PATH"] = td + os.pathsep + env.get("PATH", "")
        r = subprocess.run(
            [sys.executable, "-m", "resonance.burn.cli", "audio-cd",
             src, "--device", "/dev/sr0", "--dry-run"],
            capture_output=True, text=True, timeout=120, env=env,
            cwd=str(Path.cwd()))
        assert r.returncode == 0, r.stderr
        assert "dry-run" in r.stdout
        assert "ffmpeg" in r.stdout and "wodim" in r.stdout


def _env():
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


check("probe_table_shape", t_probe_table_shape)
check("require_unknown_backend", t_require_unknown_backend)
check("require_missing_names_tool_and_hint", t_require_missing_names_tool_and_hint)
check("require_any_backend_picks_first_available", t_require_any_backend_picks_first_available)
check("require_any_backend_all_missing_loud", t_require_any_backend_all_missing_loud)
check("run_tool_failure_loud", t_run_tool_failure_loud)
check("decode_assembles_ffmpeg_redbook", t_decode_assembles_ffmpeg_redbook)
check("write_cue_structure_and_cdtext", t_write_cue_structure_and_cdtext)
check("burn_audio_cd_wodim_argv", t_burn_audio_cd_wodim_argv)
check("burn_audio_cd_cdrdao_needs_cue", t_burn_audio_cd_cdrdao_needs_cue)
check("burn_audio_cd_missing_backend_exit3_shape", t_burn_audio_cd_missing_backend_exit3_shape)
check("make_iso_genisoimage_argv", t_make_iso_genisoimage_argv)
check("make_iso_xorriso_fallback", t_make_iso_xorriso_fallback)
check("burn_data_dvd_and_cd_argv", t_burn_data_dvd_and_cd_argv)
check("verify_iso_magic", t_verify_iso_magic)
check("encode_mpeg2_assembles_ffmpeg", t_encode_mpeg2_assembles_ffmpeg)
check("dvdauthor_xml_valid_and_honest", t_dvdauthor_xml_valid_and_honest)
check("verify_video_ts", t_verify_video_ts)
check("burn_dvd_argv", t_burn_dvd_argv)
check("make_dvd_iso_argv", t_make_dvd_iso_argv)
check("cli_backends_always_works", t_cli_backends_always_works)
check("cli_help", t_cli_help)
check("cli_missing_backend_exit_3", t_cli_missing_backend_exit_3)
check("cli_dry_run_prints_commands", t_cli_dry_run_prints_commands)

print(f"\n{PASSED} burn tests passed.")
