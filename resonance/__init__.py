# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/__init__.py -- RESONANCE v0.3.0, a generative audio engine.

Audio convention (workspace-wide contract):
  * numpy float32 arrays
  * mono shape (N,), stereo shape (2, N)
  * sample_rate is an int, default 44100

This file stays light: version strings only. Heavy imports live in the
subpackages so importing resonance never pays for numpy up front.
"""

__version__ = "0.3.0"
API_VERSION = "1.0"

__all__ = ["__version__", "API_VERSION"]
