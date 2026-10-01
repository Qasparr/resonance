# RESONANCE — SaaS Roadmap: seams today, SaaS later

Johnathan 'Qasparr' (Κασπάρρ) Monroe, Keeper of the Secret Treasure
All Rights Reserved, Without Prejudice · CashApp $axoneme
SPDX-License-Identifier: AGPL-3.0-only

> The v0.1.0 deliverable is the SEAMS -- isolated boundaries where the
> service can grow without surgery. SaaS itself (billing, tenancy,
> managed workers) is a LATER milestone and is NOT built here. Nothing
> in this document claims the SaaS exists.

## The three seams

Each seam names: what it is, the v0.1.0 implementation, the future
swap, and what stays stable across the swap. The stability column is
the promise.

### 1. Execution — `resonance/api/jobs.py` (`JobQueue`)

- **What:** who runs the render work and where its state lives. The
  routes call `create()` / `submit()` / `get()` / `mark_done()` /
  `mark_error()`; they never touch `threading` themselves.
- **Today:** in-process `JobQueue` — `submit()` starts a daemon thread
  per job, records `"rendering"` → `"done"` / `"error"` in memory.
  A restart loses job records (prototype scope, documented).
- **Future swap:** reimplement the class's internals against Celery +
  Redis (or any task broker): `submit()` serializes a task,
  `get()` polls the result backend. The method names, the status
  vocabulary (`rendering`/`done`/`error`), and the route code stay.
- **Stays stable:** the five methods, the status vocabulary, the rule
  that a failed job records `error: "Type: message"` instead of
  crashing the process.

### 2. Storage — `resonance/api/storage.py` (`StorageBackend`)

- **What:** where rendered artifacts live and how routes reach them.
  The worker calls `backend.save(job_id, audio, sr)`; the download
  route asks `artifact_path()` / `artifact_size()`. Nobody outside
  this module constructs a file path.
- **Today:** `LocalStorageBackend` — `<root>/<job_id>/job.wav` on
  local disk; `root` defaults to a temp dir per app instance.
- **Future swap:** an `ObjectStorageBackend` (S3/GCS/compatible):
  `save()` uploads and returns the key, `artifact_path()` returns a
  presigned URL (or a download proxy route streams it). The three
  method names and the "save returns {path, bytes}" shape stay.
- **Stays stable:** `save()` / `artifact_path()` / `artifact_size()`,
  the rule that missing artifacts yield `None` (which the download
  route turns into the honest 409, never a 500).

### 3. Auth — `resonance/api/auth.py` (`role_from_token`)

- **What:** the single question the routes ask about identity:
  "what role does this request carry, or None?" The FastAPI
  dependency in `service.create_app()` is a thin wrapper --
  `role_from_token(header)` → role, else 401.
- **Today:** PROTOTYPE-GRADE. `role_from_token()` parses
  `Authorization: Bearer <token>` against the static `TOKEN_ROLES`
  map. No hashing, no expiry, no per-user isolation — it proves the
  enforcement point (Bearer → role → gate), not production security.
  The map's shape is pinned by `test_api.py` so a future swap cannot
  silently inherit the prototype tokens.
- **Future swap:** replace the FUNCTION BODY ONLY — verify a JWT
  signature (or introspect an OAuth2 token), map claims to roles,
  keep returning a role string or `None`. No route changes. The 401
  on `None` stays in the dependency wrapper, untouched.
- **Stays stable:** the signature `role_from_token(header) -> str |
  None`, the `None`-means-401 rule, the role vocabulary (`dev`,
  `user`).

## What the seams do NOT do (explicit non-deliverables)

The seams isolate where these would attach, but none of them is
built, claimed, or half-built in v0.1.0:

- **Metered billing** — LATER. Would consume job records (duration,
  kind, bytes) through the queue seam's stable record shape.
- **Multi-tenancy** — LATER. Would scope the storage seam's keys per
  tenant (`<tenant>/<job_id>/job.wav`) and extend the auth seam's
  role vocabulary; the route code would not change shape.
- **Async managed workers** — LATER. The Celery swap of the queue
  seam (above); autoscaling, retries, and dead-letter queues belong
  to that milestone, not this one.
- **Persistent job ledger** — LATER. The in-memory queue records are
  the shape a Postgres/SQLite ledger would persist; the seam does not
  persist them today.

## SaaS milestone list (explicitly LATER)

1. **Auth hardening** — JWT/OAuth2 through the auth seam; token
   expiry, rotation, per-user isolation.
2. **Managed workers** — Celery/Redis queue swap; horizontal render
   workers; retry + dead-letter policy.
3. **Object storage** — storage seam swap; presigned-URL downloads;
   lifecycle/retention policy.
4. **Multi-tenancy** — tenant-scoped storage keys, tenant roles,
   per-tenant quotas.
5. **Metered billing** — usage events from job records; invoicing
   integration; free-tier rate limits.

Each milestone is a swap behind an existing seam — that is the whole
point of building the seams now. Until a milestone lands, any claim
that RESONANCE "is SaaS" is false.
