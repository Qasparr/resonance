# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/stems/mixer.py -- StemMixer: per-stem volume/mute/solo and
per-stem tempo.

Hypothesis: a stem mixer is honest only if (a) the level math is
  exactly reproducible -- mute, solo, and gain as literal
  multiplications, no hidden curves -- and (b) the time-stretch behind
  per-stem tempo is a real stretcher, probed at runtime, with its
  quality caveats written down instead of smoothed over.
Method:     StemMixer holds named stems (float32, one sample rate,
  stereo (2, N) with mono accepted as dual-mono).  mix() renders the
  summed mix: each audible stem contributes gain * audio, exactly, then
  per-stem tempo is applied through a TimeStretch adapter, the results
  are zero-padded to the longest stem, and summed in insertion order.
  TimeStretch is an interface; RubberBandAdapter drives the real
  ``rubberband`` CLI (pitch-preserving), ResampleStretch is a pure-numpy
  interpolation stretcher labeled rehearsal-grade (pitch NOT preserved).
  Both are probed at runtime; a missing ``rubberband`` binary raises
  TimeStretchUnavailable naming the tool and how to install it -- never
  a silent no-op.
Observation: muting, soloing, and gain are multiplications by 0, by a
  selector, and by a float32 scalar -- the mix is therefore bit-exact
  against a hand computation of the same expression, and the test suite
  asserts np.array_equal, not approximate equality.
Result:     a sample-true stem mixer with honest per-stem tempo.

DRIFT -- READ THIS, IT IS BY DESIGN
------------------------------------
Per-stem tempo changes a stem's LENGTH.  A stem at tempo 2.0 is half as
long; a stem at tempo 0.5 is twice as long.  mix() pads the shorter
stems with silence and renders to the LONGEST stem -- so a retimed stem
DRIFTS OUT OF SYNC with the others.  That is the remixer's choice, not
a bug: stretching one stem while the band keeps playing is inherently
unsynchronized, and any mixer that pretended otherwise would be lying
about time.  If you want the whole band to stay locked, stretch the
whole MIX (the v0.2.0 master BPM slider does exactly that) -- not one
stem.

MUTE WINS OVER SOLO
--------------------
Solo isolates: when any stem is soloed, only soloed stems sound.  But a
muted stem NEVER sounds, even when soloed -- mute is absolute, solo is
a selector among the unmuted.  This matches console convention and is
asserted in tests.

NO MEDICAL CLAIMS.  This module moves audio samples around.  It makes
no therapeutic, neurological, or health claims of any kind.
"""
import abc
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from resonance.core import buffers, io

# -- loud errors -----------------------------------------------------------------
RUBBERBAND_MISSING = (
    "Per-stem tempo with pitch preserved needs the `rubberband` command-line "
    "tool, and it is not on PATH.\n"
    "  Install it:\n"
    "    Debian/Ubuntu:  sudo apt install rubberband-cli\n"
    "    macOS:          brew install rubberband\n"
    "    Fedora:         sudo dnf install rubberband\n"
    "  Then call set_tempo() again.\n"
    "Refusing rather than faking: no time-stretch will be silently skipped, "
    "and no fake 'preserved pitch' will be claimed for the numpy fallback."
)


class TimeStretchUnavailable(Exception):
    """Raised loudly when real time-stretching is requested but the
    ``rubberband`` tool is absent.  Names the tool and how to install it."""


# -- the time-stretch interface -----------------------------------------------------
class TimeStretch(abc.ABC):
    """Interface for per-stem tempo.

    Contract: stretch(audio, ratio) -> float32 stereo (2, M) where the
    tempo factor *ratio* means: ratio > 1.0 plays FASTER (shorter
    output, M ~= N / ratio), ratio < 1.0 plays SLOWER (longer output),
    ratio == 1.0 is the identity (exact copy).  ratio must be a finite
    positive number.  Whether pitch is preserved is the adapter's
    documented property -- ask, don't assume.
    """

    #: True when the adapter preserves pitch (Rubber Band), False when it
    #: does not (numpy resample).  Checked by callers that care.
    preserves_pitch = False

    @abc.abstractmethod
    def stretch(self, audio, ratio):
        """Return the retimed audio as float32 stereo (2, M)."""

    def output_length(self, n, ratio):
        """Expected frame count of the stretched output for *n* input
        frames.  The default -- round(n / ratio) -- is exact for
        ResampleStretch and a close approximation for real stretchers;
        mix() always takes the max with the ACTUAL contribution lengths,
        so an approximation here can never truncate audio."""
        ratio = RubberBandAdapter._check_ratio(ratio)
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f"n must be a non-negative int, got {n!r}")
        return max(1, int(round(n / ratio)))


class RubberBandAdapter(TimeStretch):
    """Real time-stretch via the Rubber Band CLI (pitch-preserving).

    Hypothesis: the honest way to ship "per-stem tempo, pitch preserved"
      without vendoring DSP is to drive the actual Rubber Band tool --
      the same library behind Audacity's and Ardour's stretch -- as a
      subprocess, and to refuse loudly when it is missing.
    Method:     __init__ probes shutil.which("rubberband"); absent ->
      TimeStretchUnavailable with install instructions.  stretch()
      round-trips through 16-bit WAV in a temp dir and runs
      ``rubberband -t <ratio> in.wav out.wav`` (-t is the tempo ratio).
    Observation: the 16-bit round-trip is a caveat -- the stretched
      result is bit-exact to what rubberband wrote, but the input is
      quantized to 16-bit PCM on the way in, so this adapter is NOT
      sample-transparent the way the gain math is.  Documented here,
      not hidden.
    Result:     genuine pitch-preserving time-stretch when rubberband is
      installed; a loud named error when it is not.
    """

    preserves_pitch = True
    BIN = "rubberband"

    def __init__(self):
        """Probe for the rubberband binary.  Raises
        TimeStretchUnavailable (naming the tool + install instructions)
        when absent."""
        found = shutil.which(self.BIN)
        if not found:
            raise TimeStretchUnavailable(RUBBERBAND_MISSING)
        self.binary = found

    def stretch(self, audio, ratio):
        """Tempo-stretch via ``rubberband -t <ratio>``.  Caveat, stated
        loudly: audio passes through 16-bit PCM WAV files, so the
        result is exact *rubberband output*, not a bit-transparent
        transform of the float32 input."""
        ratio = self._check_ratio(ratio)
        audio = buffers.validate(audio, name="stretch input")
        if audio.ndim == 1:
            audio = np.stack([audio, audio])
        audio = np.asarray(audio, dtype=np.float32)
        if ratio == 1.0:
            return audio.copy()  # identity: no round-trip, exact
        with tempfile.TemporaryDirectory(prefix="resonance-stretch-") as tmp:
            src = Path(tmp) / "in.wav"
            dst = Path(tmp) / "out.wav"
            io.write_wav(src, audio, 44100)  # rate is a carrier here; see below
            # NOTE: the WAV sample rate is only a carrier for the CLI --
            # rubberband stretches by ratio, not by rate, so the carrier
            # rate does not change the math.  What matters is that the
            # caller treats the output as the same rate as the input.
            proc = subprocess.run(
                [self.binary, "-t", str(float(ratio)), str(src), str(dst)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if proc.returncode != 0:
                raise TimeStretchUnavailable(
                    "rubberband ran and failed "
                    f"(exit {proc.returncode}). Stderr:\n{proc.stderr}"
                )
            out, _ = io.read_wav(dst)
            out = buffers.validate(out, name="rubberband output")
            if out.ndim == 1:
                out = np.stack([out, out])
            return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _check_ratio(ratio):
        try:
            r = float(ratio)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tempo ratio must be a number, got {ratio!r}") from exc
        if not np.isfinite(r) or r <= 0:
            raise ValueError(f"tempo ratio must be finite and > 0, got {ratio!r}")
        return r


class ResampleStretch(TimeStretch):
    """Pure-numpy tempo change: linear-interpolation resampling.

    REHEARSAL-GRADE, PITCH NOT PRESERVED -- read that twice.  This
    adapter changes tempo by resampling, which is exactly what speeding
    up or slowing down a tape does: faster means higher-pitched
    (chipmunk), slower means lower-pitched.  It exists so tempo can be
    *rehearsed* and auditioned without installing anything -- sketching
    an arrangement, checking whether a stem works at half speed -- NOT
    for final renders.  For pitch-preserving stretch, install
    rubberband and use RubberBandAdapter.  This adapter never claims
    otherwise, in code or in docs.

    Hypothesis: a labeled, limited fallback beats both silence and lies.
    Method:     np.interp over a resampled index grid, per channel.
    Observation: at ratio 2.0 a 440 Hz sine comes back at ~880 Hz --
      the test suite asserts the length change; the pitch change is the
      documented physics of resampling, not a defect.
    Result:     honest rehearsal tempo, free of dependencies.
    """

    preserves_pitch = False

    def stretch(self, audio, ratio):
        """Resample to N/ratio frames.  Pitch is NOT preserved -- this is
        tape-speed physics, documented on the class."""
        ratio = RubberBandAdapter._check_ratio(ratio)
        audio = buffers.validate(audio, name="stretch input")
        stereo = audio if audio.ndim == 2 else np.stack([audio, audio])
        stereo = np.asarray(stereo, dtype=np.float32)
        n = stereo.shape[1]
        if ratio == 1.0:
            return stereo.copy()  # identity: exact
        new_n = max(1, int(round(n / ratio)))
        idx = np.linspace(0.0, n - 1, new_n)
        grid = np.arange(n, dtype=np.float64)
        out = np.stack(
            [np.interp(idx, grid, ch.astype(np.float64)) for ch in stereo]
        ).astype(np.float32)
        return out


# -- the mixer ----------------------------------------------------------------------
class StemMixer:
    """Sample-true stem mixer with per-stem volume/mute/solo/tempo.

    Hypothesis: if level handling is literally gain * audio -- no
      curves, no hidden normalization -- the mix is reproducible to the
      bit, and per-stem tempo is honest exactly when the stretcher
      behind it is real and labeled.
    Method:     named stems are stored as validated float32 stereo
      (2, N); mono is accepted and stored dual-mono.  mix() computes,
      per stem in insertion order: skip if muted; skip if any solo is
      active and this stem is not soloed (mute wins over solo);
      contribution = audio * float32(gain); apply the TimeStretch
      adapter when tempo != 1.0; zero-pad every contribution to the
      longest stem (silent stems still own timeline -- mute/solo/gain
      never change the mix length); sum.  All stems must share the mixer's sample rate --
      a mismatched rate raises instead of being silently resampled.
    Observation: the test suite asserts np.array_equal between mix()
      and a hand computation of the same expression -- exact, not
      approximate.
    Result:     a mixer whose math you can verify by reading it.
    """

    def __init__(self, sample_rate=44100, stretch=None):
        """sample_rate: int, the one rate every stem must share.
        stretch: a TimeStretch adapter, or None for auto-select --
          RubberBandAdapter when the ``rubberband`` binary is present,
          else ResampleStretch (rehearsal-grade, pitch NOT preserved)
          with a printed notice saying exactly that.  The notice is
          loud on purpose: a silent downgrade would be a lie about
          pitch."""
        if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
            raise ValueError(f"sample_rate must be a positive int, got {sample_rate!r}")
        self.sample_rate = int(sample_rate)
        if stretch is None:
            try:
                stretch = RubberBandAdapter()
            except TimeStretchUnavailable:
                print(
                    "StemMixer: `rubberband` not found -- per-stem tempo "
                    "falls back to ResampleStretch (REHEARSAL-GRADE, pitch "
                    "NOT preserved). Install rubberband-cli for "
                    "pitch-preserving stretch."
                )
                stretch = ResampleStretch()
        if not isinstance(stretch, TimeStretch):
            raise TypeError(
                f"stretch must be a TimeStretch adapter, got {type(stretch).__name__}"
            )
        self._stretch = stretch
        self._stems = {}  # name -> dict(audio, gain, muted, soloed, tempo)

    # -- introspection ---------------------------------------------------------------
    @property
    def stretch(self):
        """The TimeStretch adapter in use (read-only)."""
        return self._stretch

    @property
    def names(self):
        """Stem names in insertion order."""
        return list(self._stems.keys())

    def _get(self, name):
        try:
            return self._stems[name]
        except KeyError:
            raise KeyError(
                f"no stem named {name!r}; have: {self.names}"
            ) from None

    # -- stem management ----------------------------------------------------------------
    def add_stem(self, name, audio):
        """Add (or replace) a stem.  audio: float32 mono (N,) or stereo
        (2, N); mono is stored dual-mono.  Defaults: gain 1.0, unmuted,
        unsoloed, tempo 1.0.  Empty names and duplicate handling are
        loud: empty/blank names raise; re-adding a name REPLACES it
        (documented, not silent -- the old settings are discarded)."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"stem name must be a non-blank string, got {name!r}")
        buf = buffers.validate(audio, name=f"stem {name!r}")
        stereo = buf if buf.ndim == 2 else np.stack([buf, buf])
        self._stems[name] = {
            "audio": np.asarray(stereo, dtype=np.float32),
            "gain": 1.0,
            "muted": False,
            "soloed": False,
            "tempo": 1.0,
        }

    def remove_stem(self, name):
        """Remove a stem.  Unknown names raise -- no silent no-op."""
        self._get(name)
        del self._stems[name]

    # -- level handling -------------------------------------------------------------------
    def set_volume(self, name, gain):
        """Per-stem volume.  gain: finite float >= 0 (values > 1 are
        allowed -- no hidden limiter; clipping is the mixer's honest
        output, and core.edit.soft_limiter exists for taming it)."""
        try:
            g = float(gain)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"gain must be a number, got {gain!r}") from exc
        if not np.isfinite(g) or g < 0:
            raise ValueError(f"gain must be finite and >= 0, got {gain!r}")
        self._get(name)["gain"] = g

    def mute(self, name):
        """Mute a stem.  Absolute: a muted stem never sounds, even
        under solo."""
        self._get(name)["muted"] = True

    def unmute(self, name):
        self._get(name)["muted"] = False

    def solo(self, name):
        """Solo a stem.  When any stem is soloed, only soloed (and
        unmuted) stems sound."""
        self._get(name)["soloed"] = True

    def unsolo(self, name):
        self._get(name)["soloed"] = False

    # -- tempo ------------------------------------------------------------------------------
    def set_tempo(self, name, ratio):
        """Per-stem tempo via the configured TimeStretch adapter.

        ratio > 1.0: faster (shorter); ratio < 1.0: slower (longer);
        ratio == 1.0: identity.  DRIFT, BY DESIGN: a retimed stem
        changes length and therefore drifts out of sync with the other
        stems -- mix() pads with silence to the longest stem.  That is
        the remixer's choice, not a bug; see the module docstring."""
        ratio = RubberBandAdapter._check_ratio(ratio)
        self._get(name)["tempo"] = ratio

    # -- rendering ----------------------------------------------------------------------------
    def _contribution(self, name):
        """The exact signal one stem contributes to the mix (or None if
        silent).  Gain is a literal float32 multiplication; tempo goes
        through the TimeStretch adapter."""
        s = self._get(name)
        if s["muted"]:
            return None
        if self._any_solo and not s["soloed"]:
            return None
        contrib = s["audio"] * np.float32(s["gain"])  # exact, sample-true
        if s["tempo"] != 1.0:
            contrib = self._stretch.stretch(contrib, s["tempo"])
        return np.asarray(contrib, dtype=np.float32)

    @property
    def _any_solo(self):
        return any(s["soloed"] for s in self._stems.values())

    def mix(self):
        """Render the summed mix -> float32 stereo (2, M).

        M is the longest (possibly retimed) stem -- audible or not.
        Mute, solo, and gain NEVER change the mix length; only the stem
        set and the tempo settings do.  (Rationale: muting everything
        must still render a full-length silence, not a zero-length
        buffer -- the timeline is a property of the material, not of
        the faders.)  Audible contributions are gain * audio (exact),
        tempo-stretched when tempo != 1.0, zero-padded on the right to
        M, and summed in stem insertion order.  The result is bit-exact
        against a hand computation of gain * audio per audible stem --
        the test suite asserts np.array_equal.  An empty mixer renders
        (2, 0): no material, no timeline."""
        any_solo = self._any_solo
        audible = []
        lengths = []
        for name in self._stems:
            s = self._stems[name]
            n = s["audio"].shape[1]
            tempo = s["tempo"]
            sounds = not s["muted"] and (not any_solo or s["soloed"])
            if sounds:
                # exact, sample-true: literal float32 multiplication
                c = s["audio"] * np.float32(s["gain"])
                if tempo != 1.0:
                    c = self._stretch.stretch(c, tempo)
                c = np.asarray(c, dtype=np.float32)
                if c.ndim == 1:
                    c = np.stack([c, c])
                audible.append(c)
                lengths.append(c.shape[1])
            else:
                # silent, but its (retimed) length still owns timeline
                lengths.append(
                    self._stretch.output_length(n, tempo)
                    if tempo != 1.0 else n
                )
        m = max(lengths) if lengths else 0
        out = np.zeros((2, m), dtype=np.float32)
        for c in audible:
            out[:, : c.shape[1]] += c
        return out

    def render_stem(self, name):
        """Render one stem's contribution (gain/mute/solo/tempo
        applied) -> float32 stereo, or None if the stem is silent."""
        c = self._contribution(name)
        if c is None:
            return None
        if c.ndim == 1:
            c = np.stack([c, c])
        return np.asarray(c, dtype=np.float32)


__all__ = [
    "TimeStretchUnavailable",
    "TimeStretch",
    "RubberBandAdapter",
    "ResampleStretch",
    "StemMixer",
]
