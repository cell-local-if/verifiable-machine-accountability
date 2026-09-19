# Verifiable Machine Accountability

Backend foundation for recording, constraining, verifying, and auditing actions performed by automated systems. The service is designed to grow around machine identities, policy decisions, authorization, tamper-evident evidence, incident handling, responsibility attribution, audit queries, and compliance exports.

## Development

Requires Python 3.12 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest -q
uvicorn accountability.app:app --reload
```

The initial API exposes `GET /health`, which returns a JSON readiness result.

## Authorization event integrity chain

Each machine's authorization decision events form a per-machine, tamper-evident
hash chain. Every event returned by the create and list endpoints carries:

- `previous_event_id` — `null` for the machine's first event, otherwise the id
  of the preceding event in `(created_at, id)` order;
- `content_hash` — `SHA-256(UTF-8(compact key-sorted JSON of {id, machine_id,
  action_type, resource, allowed, reason, created_at}))`;
- `chain_hash` — `SHA-256(UTF-8("" + ":" + content_hash))` for the first event
  and `SHA-256(UTF-8(previous_chain_hash + ":" + content_hash))` thereafter.

All hashes are 64-character lowercase hexadecimal strings. New events are
appended to the chain tail inside a single write transaction, so concurrent
writes cannot lose events, fork, or break the chain. On startup the service
adds the new columns to pre-existing databases and backfills missing chain
data in `(created_at, id)` order; the recomputation is deterministic, so
restarting with an already complete database performs no writes.

`GET /machines/{machine_id}/authorization-decision-events/integrity` verifies
the chain read-only and returns `{valid, checked_count, broken_event_id}`:
a complete or empty chain reports `true`, the total count, and `null`;
otherwise it reports `false`, the total count, and the first event whose
content hash, link, or chain hash does not verify. A missing machine returns
`404 not_found`.

## Read-only evidence integrity audit

`GET /machines/{machine_id}/authorization-decision-events/evidence/integrity`
audits every evidence record owned by one machine, in `(created_at, id)`
order, and returns `{valid, checked_count, broken_evidence_id}`. A missing
machine returns `404 {"error":{"code":"not_found"}}`; a machine with no
evidence returns `200` with `true`, `0`, and `null`.

Each record passes only when:

- its `event_id` resolves to an existing authorization decision event that
  belongs to the same machine (missing or foreign-owned events fail);
- `evidence_type` is a string that stays non-empty after trimming surrounding
  whitespace;
- `content_hash`, compared exactly as stored with no case folding, is exactly
  64 lowercase hexadecimal characters (`^[0-9a-f]{64}$`).

The first failing record sets `valid` to `false` and `broken_evidence_id` to
its id; `checked_count` is always the machine's total evidence count, and it
is `null` when every record passes. Only the path machine's records are
checked, so damaged evidence under another machine never fails this audit.
The query is strictly read-only — it never writes, repairs, deletes, or
normalizes evidence, events, hash chains, or causal links — gives identical
results on repeat calls against unchanged data, and audits data persisted
across application restarts.

## Bounded causal traces

`GET /machines/{machine_id}/authorization-decision-events/{event_id}/causal-trace`
runs a read-only, bounded trace over the causal links created via the
causal-links endpoints. Both query parameters are required and validated
before any lookup, so any missing or invalid parameter returns `422`:

- `direction` — exactly `downstream` (existing `cause_event_id` to
  `effect_event_id`) or `upstream` (the reverse);
- `max_depth` — an integer from `1` to `20`. Boolean strings (`true`,
  `false`) and non-integer forms (`1.5`, `3.0`) are rejected.

After validation, the start event must exist and belong to `machine_id`;
otherwise the endpoint returns `404 {"error":{"code":"not_found"}}`.

The response is `{event_id, direction, max_depth, events}`. Each entry in
`events` is `{event_id, depth}` for an event reachable within `max_depth`
edges: the start event is never included, an event reachable by several paths
appears once at its smallest depth, and entries are ordered by `depth`
ascending, then by event `created_at` and `id`. The traversal stays inside the
machine, ignores links whose target event no longer exists, terminates even
when links form a cycle, and returns an empty array when nothing is
reachable. The query never writes or modifies any record.

## Read-only compliance export

`GET /machines/{machine_id}/authorization-decision-events/compliance-export`
returns a deterministic, read-only compliance slice for one machine. Both
query parameters are required and validated before the machine is looked up,
so a missing, malformed, or inverted parameter returns `422` even when the
machine does not exist:

- `from_created_at`, `to_created_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; offset forms such as `+00:00` are rejected);
  `from_created_at` must not be later than `to_created_at` (equal bounds are
  allowed).

After validation, a missing machine returns `404 {"error":{"code":"not_found"}}`.

The response is `{machine_id, from_created_at, to_created_at, events,
causal_links}`:

- `events` contains only the machine's authorization decision events whose
  `created_at` falls inside the closed interval `[from_created_at,
  to_created_at]`. Each item has exactly the same fields as the event list
  endpoint, ordered by `created_at`, then `id`.
- `causal_links` contains only the machine's causal links whose
  `cause_event_id` and `effect_event_id` both refer to exported events. Each
  item has exactly the same fields as the causal-link list endpoint, ordered by
  `created_at`, then `id`.
- Either array is empty (never omitted) when the window contains nothing.

The endpoint issues no writes, repairs, or deletions, never returns another
machine's events or links, produces identical output for identical data and
parameters on repeat calls, and reads data persisted across application
restarts.


