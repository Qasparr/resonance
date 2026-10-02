# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/ai/adapters.py -- local-model backend probes.

HYPOTHESIS
    The roadmap's honest hardware story -- models download on
    first use, GPU recommended, every adapter degrades to a LOUD
    skip when its model or torch is absent -- needs one shared
    mechanism, not per-module improvisation. So every local-model
    adapter probes through BackendProbe and raises
    ModelNotAvailable naming exactly what is missing and how to
    get it. A missing backend is exit code 3, the repo-wide
    standard for "not installed, not broken".

METHOD
    BackendProbe: importlib probes for torch / transformers /
    diffusers, CUDA visibility, and weight directories from env
    vars (RESONANCE_<MODEL>_DIR). report() is a plain dict --
    printable, testable, no side effects. LocalModelAdapter is
    the base: available() and require() (which raises); the
    generate/transcribe adapters subclass it.

RESULT
    On this machine (no torch, no weights) every adapter says
    exactly that. When the weights land, the same seam carries
    the real calls -- the probe is the contract, not a stub.
"""
import importlib.util
import os


class ModelNotAvailable(Exception):
    """A local model backend is missing. Carries the fix, not just
    the failure: what is absent and how to install or point at it."""


def _have(module_name):
    return importlib.util.find_spec(module_name) is not None


class BackendProbe:
    """What the local-model backends can see from here."""

    def __init__(self, torch_module="torch", extra_modules=(),
                 weights_env=None):
        self.torch_module = torch_module
        self.extra_modules = tuple(extra_modules)
        self.weights_env = weights_env

    def report(self):
        """Plain-dict availability report. No side effects."""
        mods = {self.torch_module: _have(self.torch_module)}
        mods.update({m: _have(m) for m in self.extra_modules})
        weights_dir = os.environ.get(self.weights_env) if self.weights_env else None
        weights_ok = bool(weights_dir) and os.path.isdir(weights_dir)
        cuda = False
        if mods.get(self.torch_module):
            try:
                import torch  # lazy: only when present
                cuda = bool(torch.cuda.is_available())
            except Exception:
                cuda = False
        return {
            "modules": mods,
            "weights_env": self.weights_env,
            "weights_dir": weights_dir,
            "weights_present": weights_ok,
            "cuda": cuda,
            "ready": all(mods.values()) and weights_ok,
        }

    def missing_pieces(self):
        """Human lines naming each absent piece and its fix."""
        rep = self.report()
        lines = []
        for mod, ok in rep["modules"].items():
            if not ok:
                lines.append(f"python module '{mod}' not installed "
                             f"(pip install {mod})")
        if not rep["weights_present"]:
            lines.append(
                f"model weights not found"
                + (f" (set {self.weights_env} to the weights directory)"
                   if self.weights_env else ""))
        if not rep["cuda"]:
            lines.append("no CUDA device visible (CPU fallback would be slow)")
        return lines


class LocalModelAdapter:
    """Base for torch/transformers-backed adapters.

    adapter_name: e.g. "musicgen". weights_env: env var pointing
    at the weights dir. Subclasses implement _run() -- the real
    model call -- which only ever executes after require().
    """

    def __init__(self, adapter_name, weights_env,
                 extra_modules=("transformers",)):
        self.adapter_name = adapter_name
        self.probe = BackendProbe(extra_modules=extra_modules,
                                  weights_env=weights_env)

    def available(self):
        return self.probe.report()["ready"]

    def require(self):
        """Raise ModelNotAvailable (exit-3 material) unless ready."""
        if self.available():
            return True
        missing = self.probe.missing_pieces()
        raise ModelNotAvailable(
            f"{self.adapter_name}: local model backend not available:\n"
            + "\n".join(f"  - {line}" for line in missing)
            + "\nRefusing to fake it -- install the pieces above and retry.")

    def status_text(self):
        rep = self.probe.report()
        lines = [f"{self.adapter_name} backend:"]
        for mod, ok in rep["modules"].items():
            lines.append(f"  module {mod}: {'present' if ok else 'MISSING'}")
        lines.append(
            f"  weights ({self.probe.weights_env}): "
            f"{rep['weights_dir'] or 'unset'} "
            f"{'present' if rep['weights_present'] else 'MISSING'}")
        lines.append(f"  cuda: {'yes' if rep['cuda'] else 'no'}")
        lines.append(f"  ready: {'YES' if rep['ready'] else 'NO'}")
        return "\n".join(lines)
