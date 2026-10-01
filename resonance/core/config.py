# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/core/config.py -- session configuration dataclass.

Hypothesis: every render in the engine should take one plain config object
  (sample rate, duration, fades) so generators share defaults and sessions
  are reproducible from a single record.
Method:     a frozen dataclass with validation in __post_init__: positive
  sample rate, positive duration, non-negative fades that fit the duration.
Observation: invalid configs fail at construction, not mid-render; the
  n_frames/n_fade_samples helpers keep sample math in one place.
Result:     generators and sessions accept a SessionConfig and behave
  identically for identical configs.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionConfig:
    """Render configuration shared by every generator.

    sample_rate : int, Hz (default 44100)
    duration    : float, seconds (default 60.0)
    fade_in     : float, seconds of fade-in applied by renderers (default 2.0)
    fade_out    : float, seconds of fade-out applied by renderers (default 2.0)
    """
    sample_rate: int = 44100
    duration: float = 60.0
    fade_in: float = 2.0
    fade_out: float = 2.0

    def __post_init__(self):
        # Frozen dataclass: validate loudly at construction so no renderer
        # ever has to guess what a zero sample rate or a 90-second fade on
        # a 60-second session means.
        if int(self.sample_rate) <= 0:
            raise ValueError(f"SessionConfig: sample_rate must be positive, "
                             f"got {self.sample_rate}")
        if float(self.duration) <= 0:
            raise ValueError(f"SessionConfig: duration must be positive, "
                             f"got {self.duration}")
        for name in ("fade_in", "fade_out"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"SessionConfig: {name} must be >= 0, "
                                 f"got {getattr(self, name)}")
        if float(self.fade_in) + float(self.fade_out) > float(self.duration):
            raise ValueError("SessionConfig: fade_in + fade_out exceeds duration")

    @property
    def n_frames(self):
        """Total samples per channel for this config."""
        return int(round(self.sample_rate * self.duration))

    def fade_samples(self, which):
        """Fade length in samples for 'in' or 'out'."""
        secs = self.fade_in if which == "in" else self.fade_out
        return int(round(self.sample_rate * secs))
