# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
tests/test_player.py -- script-style tests for resonance.player.

Run:  python3 tests/test_player.py      (from the repo root)
   or python3 -m pytest tests/test_player.py

Style (matching the other suites): each test prints "  ok: <name>"; the
end prints "<N> player tests passed." Any failure raises immediately --
the first red line is the diagnosis.

Covers: the transport state machine (play/pause/resume/stop/seek/next),
silent-rehearsal honesty (no subprocess ever spawned, .audible False,
the SILENT banner in describe()/status()), backend probing and ffplay
-ss command building, the MasterTempo ratio validation + BPM slider,
the Rubber Band adapter's loud missing-backend failure, the honest
numpy fallback labeling, extreme-ratio artifact warnings, plugin hook
firing (player.track.start / player.track.end), and the tick clock.

NO network, NO real audio output, NO audio hardware needed: a fake
clock drives transport, and subprocess/shutil.which are monkeypatched.
"""
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resonance.core import io as core_io
from resonance.player import (
    HOOK_TRACK_END,
    HOOK_TRACK_START,
    SILENT_REHEARSAL,
    AudioBackend,
    LoudMissingBackend,
    MasterTempo,
    NumpyResampleStretcher,
    Player,
    RubberBandAdapter,
    TimeStretcher,
    find_audio_backend,
    probe_audio_backends,
)
from resonance.plugins.manager import PluginManager

PASSED = 0


def check(name, fn):
    """Run one test; print ok or raise."""
    global PASSED
    fn()
    PASSED += 1
    print(f"  ok: {name}")


class FakeClock:
    """Injectable monotonic clock for transport tests."""

    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_wav(seconds=2.0, sr=44100, stereo=False):
    """Write a real fixture WAV to a temp dir; return (path, tmpdir)."""
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float64) / sr
    buf = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    if stereo:
        buf = np.stack([buf, buf * 0.5])
    tmp = tempfile.mkdtemp(prefix="resonance_test_player_")
    path = str(Path(tmp) / "tone.wav")
    core_io.write_wav(path, buf, sample_rate=sr)
    return path, tmp


def rehearsal_player(clock=None, **kw):
    """Player forced into silent rehearsal -- no subprocess, ever."""
    backend = AudioBackend(name=SILENT_REHEARSAL, argv=(), kind="rehearsal")
    kw.setdefault("tempo", MasterTempo(adapter=NumpyResampleStretcher()))
    return Player(backend=backend, clock=clock or FakeClock(), **kw)


# ---------------------------------------------------------------- states

def t_state_machine():
    clock = FakeClock()
    p = rehearsal_player(clock=clock)
    path, tmp = make_wav(seconds=2.0)
    try:
        track = p.load(path)
        assert abs(track.duration - 2.0) < 0.01, track.duration
        assert p.state == "stopped"
        p.play(0)
        assert p.state == "playing", p.state
        clock.advance(0.5)
        p.tick()
        assert abs(p.position - 0.5) < 1e-9, p.position
        p.pause()
        assert p.state == "paused"
        frozen = p.position
        clock.advance(5.0)
        p.tick()
        assert p.position == frozen, "paused clock must not advance"
        p.resume()
        assert p.state == "playing"
        clock.advance(0.25)
        p.tick()
        assert abs(p.position - (frozen + 0.25)) < 1e-9, p.position
        p.stop()
        assert p.state == "stopped" and p.position == 0.0
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


def t_play_empty_playlist_raises():
    p = rehearsal_player()
    try:
        p.play()
    except RuntimeError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("play() on empty playlist must raise")


def t_play_bad_index_raises():
    p = rehearsal_player()
    path, tmp = make_wav()
    try:
        p.load(path)
        try:
            p.play(7)
        except IndexError as exc:
            assert "out of range" in str(exc)
        else:
            raise AssertionError("bad index must raise")
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


def t_seek_clamps():
    p = rehearsal_player()
    path, tmp = make_wav(seconds=4.0)
    try:
        p.load(path)
        assert p.seek(999.0) == 4.0
        assert p.position == 4.0
        assert p.seek(-3.0) == 0.0
        assert p.seek(1.5) == 1.5
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


def t_seek_without_track_raises():
    p = rehearsal_player()
    try:
        p.seek(1.0)
    except RuntimeError as exc:
        assert "nothing" in str(exc)
    else:
        raise AssertionError("seek with no track must raise")


def t_next_and_previous():
    clock = FakeClock()
    p = rehearsal_player(clock=clock)
    paths = []
    try:
        for _ in range(2):
            path, tmp = make_wav(seconds=1.0)
            paths.append((path, tmp))
            p.load(path)
        p.play(0)
        assert p.index == 0
        p.next()
        assert p.index == 1 and p.state == "playing"
        assert p.position == 0.0
        p.previous()
        assert p.index == 0
        # next() past the end stops honestly
        p.play(1)
        assert p.next() is None
        assert p.state == "stopped"
    finally:
        import shutil as _sh
        for _p, _t in paths:
            _sh.rmtree(_t, ignore_errors=True)


def t_autoplay_advances_on_track_end():
    clock = FakeClock()
    p = rehearsal_player(clock=clock, autoplay=True)
    paths = []
    try:
        for _ in range(2):
            path, tmp = make_wav(seconds=1.0)
            paths.append((path, tmp))
            p.load(path)
        p.play(0)
        clock.advance(1.5)  # past the 1.0s track end
        p.tick()
        assert p.index == 1, p.index
        assert p.state == "playing"
        clock.advance(1.5)
        p.tick()
        assert p.state == "stopped", "playlist end must stop"
    finally:
        import shutil as _sh
        for _p, _t in paths:
            _sh.rmtree(_t, ignore_errors=True)


# ------------------------------------------------------- rehearsal honesty

def t_silent_rehearsal_never_spawns():
    """The anti-fake core: rehearsal mode must never touch subprocess."""
    p = rehearsal_player()
    path, tmp = make_wav(seconds=1.0)
    try:
        p.load(path)
        with mock.patch("subprocess.Popen",
                        side_effect=AssertionError("Popen must not run")):
            p.play(0)
            p.pause()
            p.resume()
            p.seek(0.5)
            p.stop()
        assert p.audible is False
        assert p.mode == SILENT_REHEARSAL
        st = p.status()
        assert st["audible"] is False
        assert st["mode"] == SILENT_REHEARSAL
        assert "SILENT REHEARSAL" in p.describe()
        assert "NO SOUND IS EMITTED" in p.describe()
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


def t_find_backend_sentinel_when_none():
    with mock.patch.object(shutil, "which", return_value=None):
        b = find_audio_backend()
        assert b.name == SILENT_REHEARSAL
        assert b.audible is False
        assert probe_audio_backends() == []


def t_probe_finds_aplay_first():
    def fake_which(name):
        return {"aplay": "/usr/bin/aplay"}.get(name)
    with mock.patch.object(shutil, "which", side_effect=fake_which):
        backends = probe_audio_backends()
        assert len(backends) == 1
        assert backends[0].name == "aplay"
        cmd = backends[0].command("/tmp/x.wav")
        assert cmd[0] == "/usr/bin/aplay" and cmd[-1] == "/tmp/x.wav"


def t_ffplay_command_carries_seek_offset():
    def fake_which(name):
        return {name: f"/usr/bin/{name}" for name in ("ffplay",)}.get(name)
    with mock.patch.object(shutil, "which", side_effect=fake_which):
        b = find_audio_backend()
        assert b.name == "ffplay" and b.supports_seek
        cmd = b.command("/tmp/x.wav", offset=12.5)
        assert "-ss" in cmd
        assert cmd[cmd.index("-ss") + 1] == "12.500"


def t_prefer_missing_backend_is_still_sentinel():
    def fake_which(name):
        return "/usr/bin/aplay" if name == "aplay" else None
    with mock.patch.object(shutil, "which", side_effect=fake_which):
        b = find_audio_backend(prefer="ffplay")
        assert b.name == SILENT_REHEARSAL, (
            "a missing preferred backend must not masquerade as another")


# ------------------------------------------------------------- tempo

def t_tempo_ratio_validation():
    m = MasterTempo(adapter=NumpyResampleStretcher())
    assert m.set_ratio(1.5) == 1.5
    for bad in (0.0, -1.0, 0.24, 4.01, 10.0):
        try:
            m.set_ratio(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ratio {bad} must be rejected")


def t_tempo_set_bpm():
    m = MasterTempo(adapter=NumpyResampleStretcher())
    assert abs(m.set_bpm(120, 126) - 1.05) < 1e-9
    assert abs(m.set_bpm(100, 50) - 0.5) < 1e-9
    for bad in (0, -120):
        try:
            m.set_bpm(bad, 120)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive BPM must be rejected")


def t_rubberband_missing_is_loud():
    with mock.patch.object(shutil, "which", return_value=None):
        try:
            RubberBandAdapter()
        except LoudMissingBackend as exc:
            msg = str(exc)
            assert "rubberband" in msg
            assert "apt install rubberband-cli" in msg
            assert "brew install rubberband" in msg
        else:
            raise AssertionError("absent rubberband must raise loudly")
        # adapter="rubberband" must raise, never silently fall back
        try:
            MasterTempo(adapter="rubberband")
        except LoudMissingBackend:
            pass
        else:
            raise AssertionError("'rubberband' adapter must not fall back")


def t_tempo_auto_fallback_is_loud_and_labeled():
    with mock.patch.object(shutil, "which", return_value=None):
        m = MasterTempo()  # auto
    assert isinstance(m.stretcher, NumpyResampleStretcher)
    assert m.pitch_preserved is False
    assert m.grade == "rehearsal"
    assert m.fallback_reason is not None
    assert "rubberband" in m.fallback_reason
    desc = m.describe()
    assert "pitch preserved: NO" in desc
    assert "rehearsal" in desc


def t_numpy_fallback_honest_labeling():
    s = NumpyResampleStretcher()
    assert s.pitch_preserved is False
    assert s.grade == "rehearsal"
    assert "pitch NOT preserved" in s.label
    assert s.probe() is True


def t_numpy_resample_lengths_and_shapes():
    s = NumpyResampleStretcher()
    mono = np.ones(1000, dtype=np.float32)
    out = s.stretch(mono, 2.0)
    assert out.shape == (500,), out.shape
    assert out.dtype == np.float32
    stereo = np.ones((2, 1000), dtype=np.float32)
    out2 = s.stretch(stereo, 0.5)
    assert out2.shape == (2, 2000), out2.shape
    # ratio 1.0 via MasterTempo.process returns a copy, unchanged
    m = MasterTempo(adapter=NumpyResampleStretcher())
    same = m.process(mono)
    assert same.shape == mono.shape and np.array_equal(same, mono)
    assert same is not mono


def t_extreme_ratio_warns():
    m = MasterTempo(adapter=NumpyResampleStretcher())
    m.set_ratio(3.0)
    buf = np.ones(100, dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.process(buf)
    assert any("outside the comfort zone" in str(w.message)
               for w in caught), "extreme ratio must warn about artifacts"


def t_stretcher_interface():
    # TimeStretcher is an interface: probe/stretch raise NotImplementedError
    base = TimeStretcher()
    for fn in (base.probe, lambda: base.stretch(np.ones(4), 1.0)):
        try:
            fn()
        except NotImplementedError:
            pass
        else:
            raise AssertionError("base TimeStretcher must not implement")


# ------------------------------------------------------------- hooks

def t_plugin_hooks_fire():
    clock = FakeClock()
    mgr = PluginManager()
    events = []
    mgr.on(HOOK_TRACK_START)(lambda **kw: events.append(
        ("start", kw["index"], kw["audible"])))
    mgr.on(HOOK_TRACK_END)(lambda **kw: events.append(
        ("end", kw["index"], kw["reason"], kw["audible"])))
    p = rehearsal_player(clock=clock, plugin_manager=mgr)
    path, tmp = make_wav(seconds=1.0)
    try:
        p.load(path)
        p.play(0)
        clock.advance(1.5)
        p.tick()  # finishes the only track
        starts = [e for e in events if e[0] == "start"]
        ends = [e for e in events if e[0] == "end"]
        assert starts == [("start", 0, False)], starts
        assert ends == [("end", 0, "finished", False)], ends
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- audio-mode spawn

class FakePopen:
    """Stand-in for a backend process; records argv, never makes sound."""

    instances = []

    def __init__(self, cmd, **kw):
        self.cmd = cmd
        self._poll = None
        FakePopen.instances.append(self)

    def poll(self):
        return self._poll

    def terminate(self):
        self._poll = -15

    def wait(self, timeout=None):
        return self._poll

    def kill(self):
        self._poll = -9


def t_audio_mode_spawns_backend_and_cleans_up():
    def fake_which(name):
        return "/usr/bin/ffplay" if name == "ffplay" else None
    FakePopen.instances = []
    clock = FakeClock()
    tempo = MasterTempo(adapter=NumpyResampleStretcher())
    with mock.patch.object(shutil, "which", side_effect=fake_which):
        p = Player(clock=clock, tempo=tempo)
    assert p.audible is True and p.backend.name == "ffplay"
    path, tmp = make_wav(seconds=2.0)
    try:
        p.load(path)
        with mock.patch("subprocess.Popen", FakePopen):
            p.play(0)
            assert len(FakePopen.instances) == 1
            cmd = FakePopen.instances[0].cmd
            assert cmd[0] == "/usr/bin/ffplay"
            assert cmd[-1].endswith(".wav")
            # temp WAV exists during playback...
            wav = cmd[-1]
            assert Path(wav).exists()
            # ...and the backend dying mid-track ends the track honestly
            FakePopen.instances[0]._poll = 1
            p.tick()
            assert p.state == "stopped", p.state
            assert not Path(wav).exists(), "temp WAV must be cleaned up"
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


def t_load_non_wav_uses_converter_loudly():
    """Loading e.g. .mp3 without ffmpeg must fail LOUDLY, never as silence."""
    p = rehearsal_player()
    with mock.patch.object(shutil, "which", return_value=None):
        try:
            p.load("/tmp/does-not-exist.mp3")
        except FileNotFoundError:
            pass  # missing file: loud, correct
        else:
            raise AssertionError("missing file must raise")
    # existing non-wav file, no ffmpeg -> LoudMissingBackend, not silence
    tmp = tempfile.mkdtemp(prefix="resonance_test_player_")
    fake_mp3 = str(Path(tmp) / "x.mp3")
    Path(fake_mp3).write_bytes(b"ID3" + b"\x00" * 100)
    try:
        from resonance.convert import LoudMissingBackend as ConvMissing
        with mock.patch.object(shutil, "which", return_value=None):
            try:
                p.load(fake_mp3)
            except ConvMissing as exc:
                assert "ffmpeg" in str(exc)
            else:
                raise AssertionError("no-ffmpeg .mp3 load must raise loudly")
        assert p.playlist == [], "failed load must not enqueue silence"
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------- status/cli surface

def t_status_shape():
    p = rehearsal_player()
    path, tmp = make_wav(seconds=3.0)
    try:
        p.load(path)
        p.play(0)
        st = p.status()
        for key in ("state", "mode", "audible", "backend", "track_index",
                    "track_title", "position_s", "duration_s",
                    "playlist_len", "tempo"):
            assert key in st, f"status missing {key}"
        assert st["track_title"] == "tone"
        assert st["duration_s"] == 3.0
        assert "pitch preserved: NO" in st["tempo"]  # numpy fallback
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


def t_cli_status_runs():
    from resonance.player.cli import main
    with mock.patch.object(shutil, "which", return_value=None):
        rc = main(["status"])
    assert rc == 0, rc


def t_cli_list_runs():
    from resonance.player.cli import main
    tmp = tempfile.mkdtemp(prefix="resonance_test_cli_")
    try:
        wav_path, _ = make_wav()
        # move the fixture wav into the listed dir
        dest = str(Path(tmp) / "a.wav")
        Path(wav_path).rename(dest)
        rc = main(["list", tmp])
        assert rc == 0, rc
        rc = main(["list", str(Path(tmp) / "nope")])
        assert rc == 2, rc
    finally:
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)


TESTS = [
    ("transport state machine", t_state_machine),
    ("play on empty playlist raises", t_play_empty_playlist_raises),
    ("play bad index raises", t_play_bad_index_raises),
    ("seek clamps to duration", t_seek_clamps),
    ("seek with no track raises", t_seek_without_track_raises),
    ("next/previous", t_next_and_previous),
    ("autoplay advances on track end", t_autoplay_advances_on_track_end),
    ("silent rehearsal never spawns a subprocess", t_silent_rehearsal_never_spawns),
    ("sentinel backend when no system player", t_find_backend_sentinel_when_none),
    ("probe finds aplay first", t_probe_finds_aplay_first),
    ("ffplay command carries -ss offset", t_ffplay_command_carries_seek_offset),
    ("missing preferred backend stays sentinel", t_prefer_missing_backend_is_still_sentinel),
    ("tempo ratio validation", t_tempo_ratio_validation),
    ("tempo set_bpm slider math", t_tempo_set_bpm),
    ("rubberband absence is loud", t_rubberband_missing_is_loud),
    ("auto fallback is loud and labeled", t_tempo_auto_fallback_is_loud_and_labeled),
    ("numpy fallback honest labeling", t_numpy_fallback_honest_labeling),
    ("numpy resample lengths and shapes", t_numpy_resample_lengths_and_shapes),
    ("extreme ratio warns of artifacts", t_extreme_ratio_warns),
    ("TimeStretcher is an interface", t_stretcher_interface),
    ("plugin hooks fire", t_plugin_hooks_fire),
    ("audio mode spawns backend, cleans temp wav", t_audio_mode_spawns_backend_and_cleans_up),
    ("non-wav load without ffmpeg fails loudly", t_load_non_wav_uses_converter_loudly),
    ("status dict shape", t_status_shape),
    ("cli status runs", t_cli_status_runs),
    ("cli list runs", t_cli_list_runs),
]


def main():
    for name, fn in TESTS:
        check(name, fn)
    print(f"{PASSED} player tests passed.")


if __name__ == "__main__":
    main()
