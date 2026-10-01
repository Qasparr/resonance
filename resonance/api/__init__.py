# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance.api -- the render-job HTTP service.

Public surface:
  create_app()        -- build the FastAPI app (lazy fastapi import)
  fastapi_available() -- True when fastapi is importable
  serve               -- uvicorn entry point (see serve.py)

The service renders audio to WAV in background jobs. FastAPI is an
optional dependency: importing resonance.api never requires it, and
`resonance-serve` refuses loudly when it is missing. See service.py for
the endpoint contract and the PROTOTYPE-GRADE token note.
"""

from resonance.api.service import (  # noqa: F401
    TOKEN_ROLES,
    create_app,
    fastapi_available,
)

__all__ = ["TOKEN_ROLES", "create_app", "fastapi_available"]
