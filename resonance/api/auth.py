# Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
# All Rights Reserved, Without Prejudice · CashApp $axoneme
# SPDX-License-Identifier: AGPL-3.0-only
"""
resonance/api/auth.py -- the API auth seam (PROTOTYPE-GRADE today).

Hypothesis: auth should be one replaceable function, not auth-shaped
  code scattered through the routes. If the routes only ever ask
  "what role does this request carry?", then swapping the token map
  for JWT/OAuth2 later is mechanical: replace ONE function body.
Method:     TOKEN_ROLES is the prototype map. role_from_token() parses
  the Authorization header and returns a role string or None. The
  routes (see service.py) never read TOKEN_ROLES directly -- they call
  role_from_token(), and a 401 follows a None mechanically. Nothing
  here imports fastapi: the FastAPI dependency is a thin wrapper built
  in service.create_app() so the swap stays in this file.
Observation: tests pin the map's shape (test_api.py) so a future real-
  auth swap cannot silently inherit the prototype tokens.
Result:     the documented swap point: reimplement role_from_token()
  against a real identity provider and the whole service follows.

Honesty, repeated because it matters: static tokens in source, no
hashing, no expiry, no per-user isolation. This proves the enforcement
point (Bearer -> role -> gate), not production security. SaaS-grade
auth is a LATER milestone (docs/saas-roadmap.md).
"""

# PROTOTYPE-GRADE token map. Both tokens are deliberately obvious. Real
# auth (JWT/OAuth2) replaces this whole map -- see role_from_token.
TOKEN_ROLES = {
    "dev-token-001": "dev",    # full: submit, inspect, download
    "user-token-002": "user",  # submit, inspect, download (same, for now)
}


def role_from_token(authorization_header):
    """Return the role for an Authorization header, or None.

    THE SWAP POINT. Today: "Bearer <token>" looked up in TOKEN_ROLES.
    Tomorrow (SaaS milestone): verify a JWT / introspect an OAuth2
    token here and return the role from its claims. The routes must not
    change -- they only see a role string or None.

    authorization_header -- the raw Authorization header value (a str).
      Missing/empty/malformed headers return None; None means "reject
      with 401", a decision made in exactly one other place (the
      dependency in service.py). No exceptions for bad credentials --
      rejection is data.
    """
    if not authorization_header:
        return None
    scheme, _, token = str(authorization_header).partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return TOKEN_ROLES.get(token)


__all__ = ["TOKEN_ROLES", "role_from_token"]
