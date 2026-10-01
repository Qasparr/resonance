# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/diagnostics/__init__.py -- measurement and verification tools.

Public surface:
  measure : real throughput benchmarking (time.perf_counter)
  verify  : FFT verification of rendered entrainment audio

Diagnostics may call other modules' PUBLIC APIs only -- never internals.
"""
from resonance.diagnostics import measure, verify

__all__ = ["measure", "verify"]
