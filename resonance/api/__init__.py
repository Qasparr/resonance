# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""resonance.api -- the render-job HTTP service.

Public surface:
  create_app()        -- build the FastAPI app (lazy fastapi import)
  fastapi_available() -- True when fastapi is importable
  JobQueue            -- execution seam (in-process today, Celery later)
  LocalStorageBackend -- storage seam (local disk today, object store later)
  role_from_token     -- auth seam (prototype map today, JWT/OAuth2 later)
  TOKEN_ROLES         -- the prototype token->role map
  serve               -- uvicorn entry point (see serve.py)

See docs/saas-roadmap.md for the seam contracts and the SaaS milestone
list (explicitly LATER -- the seams are the v0.1.0 deliverable).
"""

from resonance.api.auth import TOKEN_ROLES, role_from_token  # noqa: F401
from resonance.api.jobs import JobQueue  # noqa: F401
from resonance.api.service import (  # noqa: F401
    RENDER_KINDS,
    create_app,
    fastapi_available,
)
from resonance.api.storage import LocalStorageBackend, StorageBackend  # noqa: F401

__all__ = [
    "TOKEN_ROLES",
    "RENDER_KINDS",
    "JobQueue",
    "StorageBackend",
    "LocalStorageBackend",
    "role_from_token",
    "create_app",
    "fastapi_available",
]
