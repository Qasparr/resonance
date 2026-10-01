# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/player/engine.py -- playlist transport with an honest backend
story.

Hypothesis: a "player" that silently does nothing audible is a lie --
  and so is a "player" that claims cross-platform audio from pure
  Python. The honest architecture: this module owns the TRANSPORT
  (playlist, play/pause/stop/seek, position clock -- all real, all
  testable), while SOUND comes from real system players spawned as
  subprocesses, probed at runtime in a documented order.
Method:     probe_audio_backends() checks PATH for aplay (Linux/ALSA),
  paplay (PulseAudio), afplay (macOS), ffplay (ffmpeg's player) and
  returns the first found as the AudioBackend. Player.play() writes the
  (tempo-processed) track to a temp WAV and spawns the backend on it
  (ffplay gets -ss for pause/resume/seek offsets; the others restart
  from the top -- documented, not hidden). If NO backend is found, the
  engine enters SILENT REHEARSAL mode: transport and the position clock
  are fully real, but no sound emits -- and .audible is False, status()
  says so in capital letters, and the CLI prints a banner. The state is
  impossible to mistake for audible playback.
Observation: plugin hooks player.track.start / player.track.end fire
  from the transport in both modes, so plugins observe real state
  transitions. Pausing kills the subprocess and freezes the clock;
  resuming re-spawns (ffplay resumes at the offset via -ss). The master
  tempo (resonance.player.tempo.MasterTempo) processes the buffer at
  play time, so the slider is real DSP, not a label.
Result:    tests cover the full state machine, the silent-rehearsal
  honesty (no subprocess ever spawned, .audible False, status loud),
  hook firing, and seek/pause/resume semantics -- with a fake clock, no
  audio hardware, no network.

SILENT REHEARSAL, STATED PLAINLY: if you see mode "silent rehearsal",
  you are hearing NOTHING. The transport is real -- position advances,
  tracks start and end, hooks fire -- which makes it useful for testing
  playlists, timing, and plugin behavior. It is NEVER presented as
  audible playback anywhere in this codebase.

No medical or therapeutic claims are made about playback; this is a
transport and process-spawning layer.
"""

import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from resonance.core import buffers, io
from resonance.player.tempo import MasterTempo

log = logging.getLogger("resonance.player.engine")

__all__ = [
    "AudioBackend",
    "SILENT_REHEARSAL",
    "Track",
    "Player",
    "probe_audio_backends",
    "find_audio_backend",
    "HOOK_TRACK_START",
    "HOOK_TRACK_END",
]

HOOK_TRACK_START = "player.track.start"
HOOK_TRACK_END = "player.track.end"

#: Sentinel backend used when no system player exists. kind == "rehearsal",
#: audible == False, and every surface that prints it says SILENT.
SILENT_REHEARSAL = "silent rehearsal"


@dataclass
class AudioBackend:
    """One concrete way to make sound: a system player on PATH.

    name: human label ("aplay", "ffplay", ...). argv: command template;
    "{file}" is replaced with the WAV path, "{offset}" with seconds
    (only backends that support seeking declare supports_seek).
    """

    name: str
    argv: tuple
    supports_seek: bool = False
    kind: str = "system"

    @property
    def audible(self):
        """True -- every AudioBackend except the rehearsal sentinel."""
        return self.kind == "system"

    def command(self, wav_path, offset=0.0):
        """Build the real argv for one playback of wav_path."""
        out = []
        for token in self.argv:
            token = token.replace("{file}", str(wav_path))
            token = token.replace("{offset}", f"{float(offset):.3f}")
            out.append(token)
        return out


#: Probe order: platform-native first, ffplay (ships with ffmpeg) last
#: as the cross-platform fallback. Each entry: (name, argv, seeks?).
_BACKEND_CANDIDATES = (
    ("aplay", ("aplay", "-q", "{file}"), False),        # Linux / ALSA
    ("paplay", ("paplay", "{file}"), False),            # PulseAudio
    ("afplay", ("afplay", "{file}"), False),            # macOS
    ("ffplay", ("ffplay", "-nodisp", "-autoexit", "-v", "error",
                "-ss", "{offset}", "{file}"), True),    # ffmpeg's player
)


def probe_audio_backends():
    """Return every usable system audio backend, in probe order.

    Each entry is an AudioBackend with a resolved binary path. An empty
    list means: no system player found -- the Player will run in SILENT
    REHEARSAL mode (documented loudly, never presented as audible).
    """
    found = []
    for name, argv, seeks in _BACKEND_CANDIDATES:
        path = shutil.which(name)
        if path is None:
            continue
        # argv[0] is the binary name; swap in the resolved path.
        resolved = (path,) + tuple(argv[1:])
        found.append(AudioBackend(name=name, argv=resolved,
                                  supports_seek=seeks))
    return found


def find_audio_backend(prefer=None):
    """Return the first probed backend, or the SILENT_REHEARSAL sentinel.

    prefer: optionally name one backend ("ffplay"); if it is not found,
    the result is still the sentinel -- never a different backend
    masquerading as the preferred one.
    """
    backends = probe_audio_backends()
    if prefer is not None:
        for b in backends:
            if b.name == prefer:
                return b
        return AudioBackend(name=SILENT_REHEARSAL, argv=(), kind="rehearsal")
    if backends:
        return backends[0]
    return AudioBackend(name=SILENT_REHEARSAL, argv=(), kind="rehearsal")


@dataclass
class Track:
    """One loaded track: engine-native float32 audio plus metadata."""

    path: str
    buf: np.ndarray = field(repr=False)
    sample_rate: int = 44100
    title: str = ""

    def __post_init__(self):
        self.buf = buffers.validate(self.buf, name="Track.buf")
        self.sample_rate = int(self.sample_rate)
        if not self.title:
            self.title = Path(self.path).stem

    @property
    def frames(self):
        return self.buf.shape[-1]

    @property
    def duration(self):
        """Seconds, from real sample count -- not a tag, not a guess."""
        return self.frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def channels(self):
        return 2 if self.buf.ndim == 2 else 1


class Player:
    """Playlist transport: load/play/pause/stop/seek with an honest backend.

    States: "stopped" -> "playing" <-> "paused" -> "stopped". The
    position clock is real (monotonic, injectable for tests); tick()
    advances it and fires HOOK_TRACK_END when a track finishes, then
    auto-advances when autoplay is on.

    Audible output exists ONLY if a system backend was probed. Otherwise
    mode is "silent rehearsal": every transport operation works, the
    clock advances, hooks fire -- but .audible is False and status()
    says SILENT REHEARSAL in plain language.
    """

    VALID_STATES = ("stopped", "playing", "paused")

    def __init__(self, plugin_manager=None, tempo=None, clock=None,
                 backend=None, autoplay=True):
        self.playlist = []
        self.state = "stopped"
        self.index = None
        self.position = 0.0          # seconds into the current track
        self._clock = clock or time.monotonic
        self._t0 = None              # clock reading when play began/resumed
        self._pos0 = 0.0             # position at that reading
        self._proc = None            # backend subprocess, if any
        self._wav_path = None        # temp WAV feeding the backend
        self.manager = plugin_manager
        self.tempo = tempo if tempo is not None else MasterTempo()
        self.autoplay = bool(autoplay)
        self.backend = backend if backend is not None else find_audio_backend()
        log.info("Player: backend=%s audible=%s",
                 self.backend.name, self.backend.audible)

    # -- honesty surface ------------------------------------------------
    @property
    def audible(self):
        """False in silent rehearsal -- NOTHING is heard in that mode."""
        return self.backend.audible

    @property
    def mode(self):
        """'audio' or 'silent rehearsal' -- printed by status() loudly."""
        return "audio" if self.audible else SILENT_REHEARSAL

    def _fire(self, hook, **kwargs):
        if self.manager is not None:
            self.manager.fire(hook, player=self, **kwargs)

    # -- loading ----------------------------------------------------------
    def load(self, path):
        """Load one file into the playlist; returns the Track.

        .wav goes through resonance.core.io (stdlib, no deps). Anything
        else goes through resonance.convert.decode_to_pcm -- which
        raises LoudMissingBackend if ffmpeg is absent. Nothing is ever
        loaded as silence-pretending-to-be-audio.
        """
        path = str(path)
        suffix = Path(path).suffix.lower()
        if suffix == ".wav":
            buf, sr = io.read_wav(path)
        else:
            from resonance.convert import decode_to_pcm
            buf, sr = decode_to_pcm(path)
        track = Track(path=path, buf=buf, sample_rate=sr)
        self.playlist.append(track)
        log.info("Player: loaded %s (%.1fs, %dch, %dHz)",
                 track.title, track.duration, track.channels, track.sample_rate)
        return track

    def clear(self):
        """Stop and empty the playlist."""
        self.stop(reason="cleared")
        self.playlist = []

    # -- transport ----------------------------------------------------------
    def play(self, index=None):
        """Play track `index` (default: current, or 0). Fires track.start.

        In audio mode a temp WAV of the tempo-processed mix is written
        and the backend spawned. In silent rehearsal the clock starts
        and no subprocess exists -- transport is real, sound is absent,
        and status() says so.
        """
        if not self.playlist:
            raise RuntimeError("Player.play: playlist is empty -- load() first.")
        if index is not None:
            index = int(index)
            if not (0 <= index < len(self.playlist)):
                raise IndexError(
                    f"Player.play: index {index} out of range "
                    f"(0..{len(self.playlist) - 1})"
                )
            self.index = index
        elif self.index is None:
            self.index = 0
        if self.state == "paused" and index is None:
            return self.resume()
        track = self.playlist[self.index]
        self._start_track(track, offset=0.0)
        return track

    def _render_for_backend(self, track):
        """Apply master tempo -> temp WAV the backend can play."""
        processed = self.tempo.process(track.buf)
        tmp = tempfile.NamedTemporaryFile(
            prefix="resonance_play_", suffix=".wav", delete=False)
        tmp.close()
        io.write_wav(tmp.name, processed, sample_rate=track.sample_rate)
        return tmp.name

    def _start_track(self, track, offset=0.0):
        self._kill_backend()
        self._t0 = self._clock()
        self._pos0 = float(offset)
        self.position = float(offset)
        if self.audible:
            self._wav_path = self._render_for_backend(track)
            cmd = self.backend.command(self._wav_path, offset=offset)
            log.info("Player: spawning %s", " ".join(cmd))
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            log.info("Player: SILENT REHEARSAL -- transport running, "
                     "no audio emitted (no system player found).")
        self.state = "playing"
        self._fire(HOOK_TRACK_START, track=track, index=self.index,
                   audible=self.audible)

    def pause(self):
        """Freeze: kill the backend subprocess, stop the clock."""
        if self.state != "playing":
            return self.state
        self._sync_position()
        self._kill_backend()
        self.state = "paused"
        log.info("Player: paused at %.2fs", self.position)
        return self.state

    def resume(self):
        """Resume from the frozen position (ffplay seeks via -ss)."""
        if self.state != "paused" or self.index is None:
            return self.state
        track = self.playlist[self.index]
        if self.audible and not self.backend.supports_seek and self.position > 0:
            log.warning(
                "Player: backend %s cannot seek -- resuming from the top, "
                "not from %.2fs. (ffplay supports resume offsets.)",
                self.backend.name, self.position,
            )
            self._start_track(track, offset=0.0)
        else:
            self._start_track(track, offset=self.position)
        return self.state

    def stop(self, reason="stopped"):
        """Stop: kill backend, zero position, fire track.end."""
        track = self.playlist[self.index] if self.index is not None else None
        self._kill_backend()
        self.state = "stopped"
        self.position = 0.0
        self._t0 = None
        if track is not None:
            self._fire(HOOK_TRACK_END, track=track, index=self.index,
                       reason=reason, audible=self.audible)
        log.info("Player: stopped (%s)", reason)
        return self.state

    def seek(self, seconds):
        """Jump to `seconds` in the current track (clamped).

        With a loaded playlist but no current track, track 0 becomes
        current. With nothing loaded at all, raises loudly.
        """
        if self.index is None:
            if not self.playlist:
                raise RuntimeError("Player.seek: nothing loaded/playing.")
            self.index = 0
        track = self.playlist[self.index]
        seconds = max(0.0, min(float(seconds), track.duration))
        was_playing = self.state == "playing"
        if was_playing:
            self._start_track(track, offset=seconds)
        elif self.state == "paused":
            self.position = seconds
            self._pos0 = seconds
        else:
            self.position = seconds
        log.info("Player: seek to %.2fs of %s", seconds, track.title)
        return seconds

    def next(self):
        """Advance to the next track (fires end/start hooks)."""
        if not self.playlist or self.index is None:
            return None
        nxt = self.index + 1
        if nxt >= len(self.playlist):
            self.stop(reason="playlist exhausted")
            return None
        self._finish_track(reason="skipped")
        self.index = nxt
        self._start_track(self.playlist[nxt], offset=0.0)
        return self.playlist[nxt]

    def previous(self):
        """Back to the previous track, or restart this one at the top."""
        if not self.playlist or self.index is None:
            return None
        if self.index > 0:
            self._finish_track(reason="skipped")
            self.index -= 1
        self._start_track(self.playlist[self.index], offset=0.0)
        return self.playlist[self.index]

    # -- the clock ----------------------------------------------------------
    def _sync_position(self):
        """Fold elapsed clock time into self.position (playing only)."""
        if self.state == "playing" and self._t0 is not None:
            self.position = self._pos0 + (self._clock() - self._t0)

    def tick(self):
        """Advance the transport: call periodically (or in tests).

        Returns the state. When the position passes the track end,
        fires track.end(reason="finished") and auto-advances (or stops
        at the playlist end). In audio mode a dead backend process is
        treated as end-of-track -- the transport never pretends a dead
        player is still playing.
        """
        if self.state != "playing" or self.index is None:
            return self.state
        self._sync_position()
        track = self.playlist[self.index]
        backend_died = (
            self._proc is not None and self._proc.poll() is not None
            and self.position < track.duration
        )
        if backend_died:
            log.warning("Player: backend process died mid-track; "
                        "ending track honestly.")
        if self.position >= track.duration or backend_died:
            self._finish_track(reason="finished")
            nxt = self.index + 1
            if self.autoplay and nxt < len(self.playlist):
                self.index = nxt
                self._start_track(self.playlist[nxt], offset=0.0)
            else:
                # The end hook already fired in _finish_track -- halt
                # without firing it a second time.
                self._kill_backend()
                self.state = "stopped"
                self.position = 0.0
        return self.state

    def _finish_track(self, reason):
        track = self.playlist[self.index]
        self._kill_backend()
        self._fire(HOOK_TRACK_END, track=track, index=self.index,
                   reason=reason, audible=self.audible)

    def _kill_backend(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except OSError:
                pass
            self._proc = None
        if self._wav_path is not None:
            try:
                Path(self._wav_path).unlink(missing_ok=True)
            except OSError:
                pass
            self._wav_path = None
        self._t0 = None

    # -- introspection ------------------------------------------------------
    def status(self):
        """Loud, honest status dict -- mode is unmistakable."""
        self._sync_position()
        track = self.playlist[self.index] if self.index is not None else None
        return {
            "state": self.state,
            "mode": self.mode,
            "audible": self.audible,
            "backend": self.backend.name,
            "track_index": self.index,
            "track_title": track.title if track else None,
            "position_s": round(self.position, 2),
            "duration_s": round(track.duration, 2) if track else None,
            "playlist_len": len(self.playlist),
            "tempo": self.tempo.describe(),
        }

    def describe(self):
        """One screen of status, with the rehearsal banner when silent."""
        s = self.status()
        lines = []
        if not s["audible"]:
            lines.append(
                "!!! SILENT REHEARSAL MODE -- NO SOUND IS EMITTED. "
                "Transport is real; audio is absent (no system player found)."
            )
        lines.append(f"state: {s['state']} | backend: {s['backend']} "
                     f"| mode: {s['mode']}")
        if s["track_title"]:
            lines.append(f"track [{s['track_index']}]: {s['track_title']} "
                         f"({s['position_s']}s / {s['duration_s']}s)")
        lines.append(f"playlist: {s['playlist_len']} track(s)")
        lines.append(s["tempo"])
        return "\n".join(lines)
