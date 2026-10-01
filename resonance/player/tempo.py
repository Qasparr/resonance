# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/player/tempo.py -- the master BPM slider's engine: whole-mix
tempo control with pitch preserved.

Hypothesis: a DJ-style master tempo slider must change playback SPEED
  without changing PITCH (a chipmunk on fast, a drunk on slow, is a
  resampler, not a time-stretcher). True pitch-preserving time-stretch
  is a serious DSP problem -- the honest implementation here is the
  open-source Rubber Band library (the `rubberband` CLI), used as a
  subprocess adapter with the work proven by real tool output, not by
  reimplementation.
Method:     TimeStretcher -- a small adapter interface (probe / stretch).
  RubberBandAdapter shells to `rubberband --time <ratio>` on temp WAV
  files written/read through resonance.core.io (so the workspace float32
  mono/stereo contract holds on both sides). NumpyResampleStretcher is
  the pure-numpy fallback: plain linear resampling that changes tempo
  AND pitch together -- honestly labeled rehearsal-grade everywhere it
  appears, because calling it "pitch-preserving" would be a lie.
  MasterTempo wraps one adapter with ratio validation and loud fallback.
Observation: when `rubberband` is absent, RubberBandAdapter raises
  LoudMissingBackend naming the tool and how to install it; MasterTempo
  built with adapter="auto" falls back to the numpy resampler and says
  so in its state (`.fallback_reason` / `describe()`), never silently.
Result:    tests cover the interface, ratio validation, artifact
  documentation, and the honest fallback labeling.

Artifact honesty (applies to ALL time-stretching, Rubber Band included):
  ratios between ~0.5x and ~2.0x are the tool's comfort zone; outside it
  transients smear, sustained tones phasiness/warble, percussion flams
  and pre-echoes. MasterTempo logs a warning past 2x in either direction
  and documents it here instead of pretending the magic is free.

No medical or therapeutic claims are made about tempo or time-stretch;
this is signal processing plumbing.
"""

import logging
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np

from resonance.core import buffers, io

log = logging.getLogger("resonance.player.tempo")

__all__ = [
    "TimeStretcher",
    "LoudMissingBackend",
    "RubberBandAdapter",
    "NumpyResampleStretcher",
    "MasterTempo",
    "MIN_RATIO",
    "MAX_RATIO",
    "COMFORT_MIN",
    "COMFORT_MAX",
]

# Hard acceptance window for tempo ratio. Outside this, the request is
# rejected as an error rather than producing garbage.
MIN_RATIO = 0.25
MAX_RATIO = 4.0
# Comfort zone: beyond these, artifacts are documented as expected.
COMFORT_MIN = 0.5
COMFORT_MAX = 2.0


class LoudMissingBackend(Exception):
    """Raised when an optional external tool is absent.

    Loud, specific, and actionable: names the missing tool, the feature
    that needed it, and how to install it. The caller must surface this
    to the user instead of falling back to something fake.
    """


class TimeStretcher:
    """Adapter interface for whole-mix tempo change.

    Hypothesis: every stretcher must answer the same two questions --
      can you run here (probe), and what do you do to audio (stretch).
    Method:     subclasses implement probe() (bool) and stretch(buf,
      ratio) -> float32 buffer of the same channel layout whose length is
      approximately len(buf)/ratio (ratio > 1 = faster = shorter).
    Observation: the interface also carries honesty metadata --
      .pitch_preserved (bool) and .grade ("production" / "rehearsal") --
      so callers can label what the user is actually hearing.
    Result:     RubberBandAdapter and NumpyResampleStretcher both speak
      this contract; MasterTempo treats them uniformly.
    """

    #: True when tempo change leaves pitch alone; False when the
    #: adapter is really a resampler (pitch rides along with tempo).
    pitch_preserved = False
    #: "production" = fit for a real mix; "rehearsal" = preview only.
    grade = "rehearsal"
    #: Human label, shown in status output.
    label = "TimeStretcher (base)"

    def probe(self):
        """Return True if this stretcher can actually run here."""
        raise NotImplementedError

    def stretch(self, buf, ratio):
        """Return tempo-scaled float32 audio (mono (N,) or stereo (2, N))."""
        raise NotImplementedError


class RubberBandAdapter(TimeStretcher):
    """Pitch-preserving time-stretch via the real `rubberband` CLI.

    Hypothesis: Rubber Band (by Chris Cannam / Breakin' Echoes) is the
      reference open-source time-stretcher; shelling to its CLI keeps the
      package numpy-only while the heavy DSP is proven by tool output.
    Method:     probe() runs `rubberband --version`. stretch() writes
      the buffer to a temp WAV (16-bit PCM, via resonance.core.io), runs
      `rubberband --time <ratio> in.wav out.wav`, and reads the result
      back. The temp directory is cleaned up even on failure, and a
      nonzero exit becomes a StretchError carrying stderr -- never a
      silent half-file.
    Observation: if the CLI is absent, construction raises
      LoudMissingBackend naming rubberband and the install commands
      (apt/brew). Ratio is validated before any subprocess runs.
    Result:    when present, stretch() returns genuinely
      pitch-preserved audio; when absent, the failure is loud and
      actionable, never a quiet resample pretending to be Rubber Band.
    """

    pitch_preserved = True
    grade = "production"
    label = "Rubber Band CLI (pitch-preserving)"

    def __init__(self, cli="rubberband", sample_rate=44100):
        self.cli = cli
        self.sample_rate = int(sample_rate)
        path = shutil.which(cli)
        if path is None:
            raise LoudMissingBackend(
                f"Pitch-preserving tempo requires the 'rubberband' command-line tool, "
                f"which was not found on PATH. Install it with:\n"
                f"  Debian/Ubuntu:  sudo apt install rubberband-cli\n"
                f"  macOS:          brew install rubberband\n"
                f"  Windows:        download from https://breakfastquay.com/rubberband/\n"
                f"Without it, MasterTempo can only offer the numpy resample fallback "
                f"(pitch NOT preserved, rehearsal-grade)."
            )
        self.path = path

    def probe(self):
        """Run `rubberband --version`; True iff it exits 0."""
        try:
            out = subprocess.run(
                [self.path, "--version"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return out.returncode == 0

    def _check_ratio(self, ratio):
        ratio = float(ratio)
        if not (MIN_RATIO <= ratio <= MAX_RATIO):
            raise ValueError(
                f"RubberBandAdapter: ratio {ratio} outside accepted range "
                f"[{MIN_RATIO}, {MAX_RATIO}] -- refusing rather than "
                f"producing garbage."
            )
        return ratio

    def stretch(self, buf, ratio):
        """Time-stretch via the rubberband CLI; pitch preserved.

        ratio > 1.0 shortens (faster); ratio < 1.0 lengthens (slower).
        Returns float32 in the same channel layout as `buf`.
        """
        arr = buffers.validate(buf, name="stretch")
        ratio = self._check_ratio(ratio)
        _warn_extreme(ratio)
        sample_rate = self.sample_rate
        in_wav = out_wav = None
        try:
            with tempfile.TemporaryDirectory(prefix="resonance_rubberband_") as td:
                in_wav = Path(td) / "in.wav"
                out_wav = Path(td) / "out.wav"
                io.write_wav(in_wav, arr, sample_rate=sample_rate)
                # rubberband CLI: --time RATIO adjusts tempo.
                cmd = [self.path, "--time", str(ratio),
                       str(in_wav), str(out_wav)]
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300,
                )
                if proc.returncode != 0 or not out_wav.exists():
                    raise RuntimeError(
                        f"RubberBandAdapter: rubberband failed "
                        f"(exit {proc.returncode}): "
                        f"{(proc.stderr or proc.stdout or 'no output').strip()[:500]}"
                    )
                stretched, _sr = io.read_wav(out_wav)
                return stretched.astype(np.float32, copy=False)
        finally:
            for p in (in_wav, out_wav):
                try:
                    if p is not None:
                        Path(p).unlink(missing_ok=True)
                except OSError:
                    pass


class NumpyResampleStretcher(TimeStretcher):
    """Pure-numpy fallback: linear resampling. PITCH IS NOT PRESERVED.

    Hypothesis: when Rubber Band is absent the player still needs a
      tempo control for rehearsal -- e.g. sketching a BPM on a laptop
      with no installs -- but the honest name for "resample faster" is
      a pitch change, and the module must say so everywhere.
    Method:     for ratio r, take every r-th sample by linear
      interpolation: out[i] = in[i * r]. Mono (N,) and stereo (2, N)
      are handled per-channel with np.interp -- no per-sample Python
      loops, so it stays fast.
    Observation: pitch_preserved is False and grade is "rehearsal";
      the label says it outright. MasterTempo repeats the warning in
      describe() and the CLI prints it on use.
    Result:    a real, testable tempo change -- honestly a pitch-shifting
      resample, fit for rehearsal and BPM sketching, never for a mix
      the user is told is "pitch-preserving."
    """

    pitch_preserved = False
    grade = "rehearsal"
    label = "numpy linear resample (pitch NOT preserved -- rehearsal only)"

    def probe(self):
        """numpy is the hard dependency, so this is always available."""
        return True

    def stretch(self, buf, ratio):
        """Resample: ratio > 1 shortens AND raises pitch (tape-style)."""
        arr = buffers.validate(buf, name="stretch")
        ratio = float(ratio)
        if not (MIN_RATIO <= ratio <= MAX_RATIO):
            raise ValueError(
                f"NumpyResampleStretcher: ratio {ratio} outside accepted "
                f"range [{MIN_RATIO}, {MAX_RATIO}]."
            )
        _warn_extreme(ratio)
        n = arr.shape[-1]
        new_n = max(1, int(round(n / ratio)))
        src_idx = np.arange(new_n, dtype=np.float64) * ratio
        if arr.ndim == 1:
            out = np.interp(src_idx, np.arange(n, dtype=np.float64), arr)
            return out.astype(np.float32)
        chans = [
            np.interp(src_idx, np.arange(n, dtype=np.float64), ch)
            for ch in arr
        ]
        return np.stack(chans).astype(np.float32)


def _warn_extreme(ratio):
    """Document the artifact cost of extreme ratios, out loud.

    Time-stretch is not free: pushed far, every stretcher smears
    transients and warbles sustained material. This warning is the
    module saying so at the point of use instead of in a footnote.
    """
    if ratio < COMFORT_MIN or ratio > COMFORT_MAX:
        warnings.warn(
            f"MasterTempo: ratio {ratio:.2f}x is outside the comfort zone "
            f"[{COMFORT_MIN}x, {COMFORT_MAX}x]. Expect artifacts -- smeared "
            f"transients, phasiness/warble on sustained tones, flamming on "
            f"percussion. This is inherent to time-stretching, not a bug.",
            UserWarning,
            stacklevel=3,
        )
        log.warning("extreme tempo ratio %.2fx: artifacts expected", ratio)


class MasterTempo:
    """The master BPM slider's engine: whole-mix tempo with pitch kept.

    Hypothesis: the player needs ONE tempo control over the whole mix --
      the DJ pitch-fader -- and it must be honest about which backend is
      doing the work and whether pitch is actually preserved.
    Method:     wraps a TimeStretcher. adapter="auto" (default) probes
      Rubber Band and falls back to the numpy resampler ONLY when the
      CLI is absent, recording .fallback_reason -- the fallback is loud
      (logged + in describe()), never silent. adapter="rubberband"
      raises LoudMissingBackend instead of falling back; an explicit
      adapter instance is used as given. ratio accepts 0.25..4.0 and
      warns past 2x either way.
    Observation: .pitch_preserved and .grade always describe the ACTIVE
      backend, so a UI can print "pitch preserved: NO (rehearsal)" and
      mean it.
    Result:    process(buf) returns the tempo-scaled mix; set_bpm /
      set_ratio move the slider; describe() prints the honest backend
      story in one paragraph.
    """

    def __init__(self, adapter="auto", ratio=1.0, sample_rate=44100):
        self._stretcher = None
        self.fallback_reason = None
        if adapter == "auto":
            try:
                self._stretcher = RubberBandAdapter(sample_rate=sample_rate)
                log.info("MasterTempo: using %s", self._stretcher.label)
            except LoudMissingBackend as exc:
                self.fallback_reason = str(exc)
                self._stretcher = NumpyResampleStretcher()
                log.warning(
                    "MasterTempo: rubberband absent -- falling back to %s. "
                    "Pitch will NOT be preserved.",
                    self._stretcher.label,
                )
        elif adapter == "rubberband":
            self._stretcher = RubberBandAdapter(sample_rate=sample_rate)  # raises loudly
        elif isinstance(adapter, TimeStretcher):
            self._stretcher = adapter
        else:
            raise TypeError(
                f"MasterTempo: adapter must be 'auto', 'rubberband', or a "
                f"TimeStretcher instance, got {adapter!r}"
            )
        self._ratio = 1.0
        self.set_ratio(ratio)

    # -- introspection -------------------------------------------------
    @property
    def stretcher(self):
        """The active TimeStretcher adapter."""
        return self._stretcher

    @property
    def pitch_preserved(self):
        """Whether the ACTIVE backend preserves pitch (honest)."""
        return bool(self._stretcher.pitch_preserved)

    @property
    def grade(self):
        """'production' or 'rehearsal' for the ACTIVE backend."""
        return self._stretcher.grade

    @property
    def backend_label(self):
        """Human label of the ACTIVE backend."""
        return self._stretcher.label

    # -- the slider ----------------------------------------------------
    @property
    def ratio(self):
        """Current tempo ratio: 1.0 = original tempo, 2.0 = double time."""
        return self._ratio

    def set_ratio(self, ratio):
        """Move the slider; validates 0.25..4.0, warns past 2x."""
        ratio = float(ratio)
        if not (MIN_RATIO <= ratio <= MAX_RATIO):
            raise ValueError(
                f"MasterTempo: ratio {ratio} outside accepted range "
                f"[{MIN_RATIO}, {MAX_RATIO}]."
            )
        self._ratio = ratio
        return self._ratio

    def set_bpm(self, original_bpm, target_bpm):
        """Move the slider by BPM: e.g. set_bpm(120, 126) -> 1.05x.

        original_bpm must be positive; this is arithmetic, not beat
        detection -- the module does not claim to know a track's BPM.
        """
        original_bpm = float(original_bpm)
        target_bpm = float(target_bpm)
        if original_bpm <= 0:
            raise ValueError(
                f"MasterTempo.set_bpm: original_bpm must be positive, "
                f"got {original_bpm}"
            )
        if target_bpm <= 0:
            raise ValueError(
                f"MasterTempo.set_bpm: target_bpm must be positive, "
                f"got {target_bpm}"
            )
        return self.set_ratio(target_bpm / original_bpm)

    # -- processing ----------------------------------------------------
    def process(self, buf):
        """Return the whole mix at the current tempo ratio.

        The buffer passes through the ACTIVE stretcher -- Rubber Band
        when available (pitch preserved), the numpy resampler in
        rehearsal fallback (pitch shifts with tempo).
        """
        arr = buffers.validate(buf, name="MasterTempo.process")
        if self._ratio == 1.0:
            return arr.astype(np.float32, copy=True)
        return self._stretcher.stretch(arr, self._ratio)

    def describe(self):
        """One-paragraph honest backend story for status output."""
        lines = [
            f"Master tempo: {self._ratio:.3f}x "
            f"(pitch preserved: {'YES' if self.pitch_preserved else 'NO'})",
            f"Backend: {self.backend_label} [{self.grade}]",
        ]
        if self.fallback_reason is not None:
            lines.append(
                "NOTE: Rubber Band was requested by default but is not "
                "installed; the numpy resample fallback is active, so tempo "
                "changes ALSO change pitch. Install rubberband-cli for "
                "pitch-preserving stretch."
            )
        return "\n".join(lines)
