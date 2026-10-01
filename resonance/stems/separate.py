# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/stems/separate.py -- stem-separation front door (Demucs adapter).

Hypothesis: RESONANCE can offer genuine, open-tooling stem separation --
  the 4-stem Demucs split (vocals / drums / bass / other) -- WITHOUT
  bundling gigabytes of model weights or hard-depending on torch, by
  treating the model as a *guest*: probed at runtime, used through the
  demucs project's own interfaces, and refusing loudly when absent.
Method:     Separator is the abstract contract (probe + separate).
  DemucsAdapter tries the demucs python API when ``demucs`` imports, else
  the ``demucs`` CLI via subprocess, else raises StemSeparationUnavailable
  with the missing package named and install instructions.  Inputs are
  resampled to 44.1 kHz for the model (numpy, utility-grade) and the
  resulting stems are resampled back to the input rate on the way out.
  All outputs are float32 stereo arrays in the workspace contract,
  shape (2, N).
Observation: torch and demucs are BOTH absent on most machines, including
  this build host.  That is the normal case, not an edge case, so the
  absence path is the most-tested path in this file.
Result:     real stems when Demucs is present; a loud, specific refusal --
  naming torch/demucs and how to install each -- when it is not.

THE HONEST FALLBACK, STATED PLAINLY
------------------------------------
There is no silent fallback in this module, and that is deliberate.  A
"fallback" that hands you EQ-filtered copies labeled as Demucs output
would be a lie -- high-passed treble is not a vocal stem, low-passed
rumble is not a bass stem, and pretending otherwise poisons every
downstream remix, analysis, and measurement.  So the honest fallback is
a documented refusal: StemSeparationUnavailable, raised with a message
that names the missing package, why it is needed, and exactly how to
install it.  What the module cannot do, it says.  Never a silent guess.

HONEST LIMIT, STATED UP FRONT (per the v0.2.0 roadmap contract)
--------------------------------------------------------------
Open stem models separate 4-6 stems -- vocals, drums, bass, "other"
(piano+everything-else in one bucket) -- NOT arbitrary individual
instruments.  "Isolate the guitar from the piano" is beyond current
open tooling.  Demucs does not separate guitar-from-piano, and this
module will never claim it does.

NO MEDICAL CLAIMS.  This module makes no therapeutic, neurological, or
health claims of any kind.  It splits audio files into stems.
"""
import abc
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from resonance.core import buffers, io

# -- the stem set --------------------------------------------------------------
# Demucs' own vocabulary.  Nothing else is promised here: asking for
# "guitar" or "piano" as a stem is outside what open tooling can do, and
# separate() raises on unknown stem requests rather than inventing them.
STEMS = ("vocals", "drums", "bass", "other")

DEMUCS_SAMPLE_RATE = 44100  # the rate every stock Demucs model expects

_TORCH_MISSING = (
    "Stem separation needs PyTorch, and torch is not importable in this "
    "Python environment.\n"
    "  Install it (CPU build is enough to run, slow):\n"
    "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
    "  Then install demucs itself (see the demucs message if that also "
    "fails).\n"
    "Refusing rather than faking: no EQ-filtered copies will be labeled "
    "as Demucs output."
)

_DEMUCS_MISSING = (
    "Stem separation needs the demucs package, and it is neither "
    "importable as a Python module nor present as the `demucs` CLI on "
    "PATH.\n"
    "  Install it:\n"
    "    pip install demucs\n"
    "  and download a model on first use (models are NOT bundled -- they "
    "are gigabytes of weights and live with the user, not in this repo).\n"
    "Refusing rather than faking: no EQ-filtered copies will be labeled "
    "as Demucs output."
)


class StemSeparationUnavailable(Exception):
    """Raised -- loudly, with install instructions -- when no real
    separation backend is available.

    This is the honest fallback.  It carries the missing piece's name
    and how to get it; it never carries fake stems.
    """


class Separator(abc.ABC):
    """Abstract contract every stem separator honors.

    Hypothesis: one interface -- probe() then separate() -- is enough to
      keep every backend (Demucs today, others later) interchangeable.
    Method:     probe() is a classmethod so availability can be checked
      without constructing anything; separate() takes a WAV path and
      returns a dict mapping each of STEMS to a float32 stereo array
      shape (2, N).
    """

    #: the stem vocabulary this contract speaks
    STEM_NAMES = STEMS

    @classmethod
    @abc.abstractmethod
    def probe(cls):
        """Return an availability report dict.

        Keys: "available" (bool), "backend" (str), plus backend-specific
        diagnostics ("torch", "demucs", "demucs_cli", "reason").  Never
        raises for missing dependencies -- absence is reported, not
        thrown.
        """

    @abc.abstractmethod
    def separate(self, path, *, model="htdemucs"):
        """Separate *path* (WAV) into stems.

        Returns dict {stem_name: float32 stereo (2, N)}.  All four STEMS
        keys are always present.  Raises StemSeparationUnavailable when
        no backend is usable.
        """


def _resample_utility_grade(audio, sr_from, sr_to):
    """Linear-interpolation resample, mono (N,) or stereo (2, N) -> float32.

    Honest label: utility-grade.  Good enough to feed a model its
    expected 44.1 kHz and to hand stems back at the input rate; NOT a
    mastering resampler.  np.interp is exact at integer ratios only.
    """
    if sr_from == sr_to:
        return np.asarray(audio, dtype=np.float32)
    mono = audio.ndim == 1
    a = np.atleast_2d(np.asarray(audio, dtype=np.float32))
    n = a.shape[1]
    new_n = max(1, int(round(n * sr_to / sr_from)))
    idx = np.linspace(0.0, n - 1, new_n)
    out = np.stack([np.interp(idx, np.arange(n), ch) for ch in a]).astype(np.float32)
    return out[0] if mono else out


class DemucsAdapter(Separator):
    """Separator backed by the real Demucs project (Meta AI Research).

    Hypothesis: Demucs is usable as a guest -- never imported at module
      load (numpy-only hard dep is preserved), probed lazily, driven
      through its own python API when importable and through its own
      ``demucs`` CLI subprocess otherwise.
    Method:     _require_backend() imports torch first (torch missing ->
      loud torch message), then demucs (importable -> python-API mode;
      else look for the CLI on PATH; neither -> loud demucs message).
      Python-API mode feeds a 44.1 kHz torch tensor through
      demucs.pretrained.get_model / demucs.apply.apply_model (the demucs
      4.x public surface, subject to their changes) and maps
      model.sources back onto STEMS.  CLI mode runs
      ``demucs -n <model> -o <tmpdir> <input>`` and reads the emitted
      vocals.wav / drums.wav / bass.wav / other.wav with resonance.core.io.
    Observation: on machines without torch this entire path is dead code
      by design -- probe() reports it, separate() refuses it.  That is
      the documented honest fallback, not a bug.
    Result:     real Demucs stems when Demucs is present; a refusal that
      names the missing package and the install command when it is not.
    """

    BACKEND = "demucs"

    def __init__(self, model="htdemucs", device=None):
        """model: demucs model name, e.g. "htdemucs" (default, hybrid
        transformer) or "mdx_extra".  device: "cpu"/"cuda"/None (None =
        cuda if available else cpu).  Nothing is imported here -- the
        laziness is the point."""
        self.model = model
        self.device = device

    # -- probing (never raises for missing deps) --------------------------------
    @classmethod
    def probe(cls):
        """Availability report.  Safe to call with nothing installed."""
        report = {
            "available": False,
            "backend": cls.BACKEND,
            "torch": None,
            "demucs": None,
            "demucs_cli": None,
            "reason": "",
        }
        try:
            import torch  # noqa: F401

            report["torch"] = getattr(torch, "__version__", "unknown")
        except ImportError:
            report["reason"] = (
                "torch is not importable. " + _TORCH_MISSING.split("\n")[0]
            )
            return report
        try:
            import demucs  # noqa: F401

            report["demucs"] = getattr(demucs, "__version__", "unknown")
            report["available"] = True
            report["reason"] = "demucs python API importable"
        except ImportError:
            cli = shutil.which("demucs")
            if cli:
                report["demucs_cli"] = cli
                report["available"] = True
                report["reason"] = f"demucs CLI found at {cli}"
            else:
                report["reason"] = (
                    "demucs is neither importable nor on PATH. "
                    + _DEMUCS_MISSING.split("\n")[0]
                )
        return report

    # -- the loud gate -----------------------------------------------------------
    def _require_backend(self):
        """Import torch, then demucs; return "api" or "cli".

        Raises StemSeparationUnavailable naming exactly what is missing
        and how to install it.  Called on every separate() so a machine
        that gains demucs mid-session starts working without a restart.
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise StemSeparationUnavailable(_TORCH_MISSING) from exc
        try:
            import demucs  # noqa: F401

            return "api"
        except ImportError:
            pass
        if shutil.which("demucs"):
            return "cli"
        raise StemSeparationUnavailable(_DEMUCS_MISSING)

    # -- the work -----------------------------------------------------------------
    def separate(self, path, *, model=None):
        """Separate *path* into {stem: float32 stereo (2, N)}.

        Raises:
            StemSeparationUnavailable: torch or demucs missing (the
                honest fallback -- message carries install instructions).
            FileNotFoundError: *path* does not exist.
            ValueError: unreadable/unsupported WAV (from core.io).
        """
        mode = self._require_backend()
        model = model or self.model
        audio, sr = io.read_wav(path)  # mono (N,) or stereo (2, N), float32
        audio = buffers.validate(audio, name="input")
        if mode == "api":
            return self._separate_api(audio, sr, model)
        return self._separate_cli(path, audio, sr, model)

    def _to_stereo(self, audio):
        """Mono (N,) -> dual-mono stereo (2, N); stereo passes through."""
        if audio.ndim == 1:
            return np.stack([audio, audio]).astype(np.float32)
        return np.asarray(audio, dtype=np.float32)

    def _separate_api(self, audio, sr, model_name):
        """Drive the demucs python API (demucs 4.x public surface).

        Feeds the model 44.1 kHz, maps model.sources onto STEMS, hands
        stems back at the input rate as float32 stereo.  Any runtime
        failure inside demucs is re-raised as StemSeparationUnavailable
        WITH the underlying error attached -- a failed real backend is
        still not an excuse to invent stems.
        """
        try:
            import torch
            from demucs.apply import apply_model
            from demucs.pretrained import get_model

            device = self.device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model = get_model(model_name)
            model.to(device)
            model.eval()
            stereo = self._to_stereo(
                _resample_utility_grade(audio, sr, DEMUCS_SAMPLE_RATE)
            )
            tensor = torch.from_numpy(stereo).to(device)
            with torch.no_grad():
                estimates = apply_model(model, tensor[None], device=device)[0]
            sources = list(model.sources)  # e.g. ["drums","bass","other","vocals"]
            n_frames = stereo.shape[1]
            out = {}
            for i, name in enumerate(sources):
                est = estimates[i].detach().to("cpu").numpy()
                if est.shape[1] > n_frames:
                    est = est[:, :n_frames]
                elif est.shape[1] < n_frames:
                    est = np.pad(est, ((0, 0), (0, n_frames - est.shape[1])))
                out[name] = self._to_stereo(
                    _resample_utility_grade(est, DEMUCS_SAMPLE_RATE, sr)
                )
            # Always return the full STEMS vocabulary; a model that drops
            # a source would otherwise silently change the contract.
            for name in STEMS:
                if name not in out:
                    raise StemSeparationUnavailable(
                        f"Demucs model {model_name!r} did not emit stem "
                        f"{name!r} (sources: {sources}). Refusing rather "
                        "than inventing it."
                    )
            return {name: out[name] for name in STEMS}
        except StemSeparationUnavailable:
            raise
        except Exception as exc:  # the demucs API moved or broke: say so
            raise StemSeparationUnavailable(
                "The demucs python API is importable but failed at runtime. "
                f"Underlying error: {type(exc).__name__}: {exc}. "
                "No fake stems were produced."
            ) from exc

    def _separate_cli(self, path, audio, sr, model_name):
        """Drive the ``demucs`` CLI as a subprocess.

        Runs ``demucs -n <model> -o <tmpdir> <input>`` (the CLI handles
        its own resampling internally) and reads the four emitted WAVs
        back through resonance.core.io, resampling to the input rate
        when needed.  A nonzero CLI exit becomes a loud error carrying
        the CLI's own stderr -- never silence, never fakes.
        """
        with tempfile.TemporaryDirectory(prefix="resonance-stems-") as tmp:
            cmd = [
                "demucs",
                "-n",
                model_name,
                "-o",
                tmp,
                str(Path(path)),
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600
            )
            if proc.returncode != 0:
                raise StemSeparationUnavailable(
                    "The demucs CLI ran and failed "
                    f"(exit {proc.returncode}). Stderr:\n{proc.stderr}\n"
                    "No fake stems were produced."
                )
            # Layout: <tmp>/<model>/<track-name>/<stem>.wav
            candidates = list(Path(tmp).rglob("vocals.wav"))
            if not candidates:
                raise StemSeparationUnavailable(
                    "The demucs CLI exited 0 but emitted no vocals.wav -- "
                    f"unexpected output layout under {tmp}. Refusing "
                    "rather than guessing where the stems went."
                )
            stem_dir = candidates[0].parent
            out = {}
            for name in STEMS:
                wav_path = stem_dir / f"{name}.wav"
                if not wav_path.exists():
                    raise StemSeparationUnavailable(
                        f"demucs CLI output is missing {name}.wav under "
                        f"{stem_dir}. Refusing rather than inventing it."
                    )
                stem_audio, stem_sr = io.read_wav(wav_path)
                out[name] = self._to_stereo(
                    _resample_utility_grade(stem_audio, stem_sr, sr)
                )
            return out


__all__ = [
    "STEMS",
    "DEMUCS_SAMPLE_RATE",
    "StemSeparationUnavailable",
    "Separator",
    "DemucsAdapter",
]
