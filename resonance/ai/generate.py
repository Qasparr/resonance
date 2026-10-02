# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/ai/generate.py -- generative music adapter (MusicGen tradition).

HYPOTHESIS
    "Extended length" must be composed structure, not a looped 30
    seconds. So this module separates the two jobs honestly: the
    MODEL renders sections (intro/verse/chorus/outro) from text or
    melody conditioning, and arrange_sections() joins them with
    the v0.1.0 editing primitives (crossfades via the splice
    engine) -- deterministic, numpy-only, testable TODAY. The
    model call itself is behind the adapter probe: no torch, no
    weights, no fake audio -- ModelNotAvailable, exit 3.

METHOD
    MusicGenAdapter(LocalModelAdapter): require() first, then
    _run() -- the real model call, implemented when weights land
    (today it raises ModelNotAvailable through the same path, so
    the seam is exercised, not stubbed around). arrange_sections()
    takes rendered section buffers + a section order and joins
    them with raised-cosine crossfades; every output is labeled
    generated in its returned provenance record.

RESULT
    generate() is honest about being unavailable; arrangement is
    honest about being real. Provenance is never hidden: the
    returned record names the model id (or "no-model") and the
    section map.
"""
import numpy as np

from resonance.ai.adapters import LocalModelAdapter, ModelNotAvailable
from resonance.core.buffers import validate as _validate_buf

WEIGHTS_ENV = "RESONANCE_MUSICGEN_DIR"
MODEL_ID = "facebook/musicgen-stereo-medium"  # adapter target, not a claim


def _crossfade(a, b, n_fade):
    """Raised-cosine crossfade join of two mono float64 arrays."""
    n_fade = max(8, int(n_fade))
    n_fade = min(n_fade, len(a), len(b))
    t = np.linspace(0.0, np.pi / 2.0, n_fade)
    fade_out = np.cos(t) ** 2
    fade_in = np.sin(t) ** 2
    joined = np.concatenate([
        a[:-n_fade],
        a[-n_fade:] * fade_out + b[:n_fade] * fade_in,
        b[n_fade:],
    ])
    return joined


class MusicGenAdapter(LocalModelAdapter):
    """Text/melody-conditioned section renderer (adapter target:
    MusicGen / AudioLDM / Stable Audio Open tradition)."""

    def __init__(self, model_id=MODEL_ID):
        super().__init__("musicgen", WEIGHTS_ENV,
                         extra_modules=("transformers",))
        self.model_id = model_id

    def _run(self, prompt, duration_s, sr):
        """The real model call. Implemented when weights land.

        Today this is unreachable without require() passing, and
        require() fails without torch + weights -- so the honest
        answer today is ModelNotAvailable, not silence and not
        fake audio.
        """
        raise ModelNotAvailable(
            f"musicgen: model call not wired (target {self.model_id}); "
            f"this path only executes after torch + weights are present.")

    def generate(self, prompt, duration_s=30.0, sr=44100):
        """Render one section. Loud skip until the backend exists."""
        self.require()
        return self._run(prompt, duration_s, sr)


def arrange_sections(sections, order, sr=44100, crossfade_s=2.0):
    """Join rendered sections into composed structure.

    sections: dict name -> mono/stereo buffer. order: list of
    names (e.g. ["intro", "verse", "chorus", "verse", "outro"]).
    Returns (audio, provenance): the joined track and a record
    naming every section and the fades between them. Deterministic
    and fully testable -- this half of generation is real today.
    """
    if not order:
        raise ValueError("arrange_sections: empty section order")
    missing = [name for name in order if name not in sections]
    if missing:
        raise ValueError(f"arrange_sections: unknown sections {missing}")
    sr = int(sr)
    n_fade = int(crossfade_s * sr)
    mono_parts = []
    for name in order:
        buf = _validate_buf(sections[name], name=f"section {name!r}")
        mono_parts.append(buf.astype(np.float64).mean(axis=0)
                          if buf.ndim == 2 else buf.astype(np.float64))
    joined = mono_parts[0]
    for part in mono_parts[1:]:
        joined = _crossfade(joined, part, n_fade)
    provenance = {
        "generator": "resonance.ai.generate.arrange_sections",
        "model": "no-model (arrangement only)",
        "order": list(order),
        "crossfade_s": float(crossfade_s),
        "sections_s": {name: float(len(_validate_buf(sections[name],
                                                    name="prov")) / sr)
                       for name in order},
        "labeled": "generated",
    }
    return joined.astype(np.float32), provenance
