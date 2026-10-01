# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance.plugins -- discovery, lifecycle, and hook registry.

Public surface: PluginManager, PluginAPIError, API_VERSION (the exact
contract value plugins must declare, currently "1.0").
"""

from .manager import API_VERSION, PluginAPIError, PluginManager  # noqa: F401
