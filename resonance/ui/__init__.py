# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/ui/__init__.py -- the skinnable player UI subpackage.

Hypothesis: skins (validation, fallbacks, hooks) and the terminal
  player (regions, ANSI rendering, karaoke hook point) are two halves
  of one promise -- the skin contract -- so they belong behind one
  import surface, sharing one plugin hook registry.
Method:     re-export the validator (SkinValidator, DEFAULT_SKIN,
  validate/load/apply helpers, the shared plugin_manager) from skin.py
  and the headless TUI (Player, KaraokeProtocolError, run_demo) from
  tui.py. The CLI entry point main() lives in cli.py.
Result:     `from resonance.ui import Player, validate_skin` just works.
"""
from resonance.ui.skin import (
    DEFAULT_SKIN,
    SkinResult,
    SkinValidator,
    apply_skin,
    current_skin,
    load_skin_file,
    plugin_manager,
    validate_skin,
)
from resonance.ui.tui import KaraokeProtocolError, Player, run_demo

__all__ = [
    "DEFAULT_SKIN",
    "SkinResult",
    "SkinValidator",
    "apply_skin",
    "current_skin",
    "load_skin_file",
    "plugin_manager",
    "validate_skin",
    "KaraokeProtocolError",
    "Player",
    "run_demo",
]
