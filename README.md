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

## Causal links between decision events

`POST /machines/{machine_id}/authorization-decision-events/{cause_event_id}/causal-links`
records a directed "cause led to effect" edge between two authorization
decision events of the same machine. The body is `{"effect_event_id": "..."}`
(a non-empty string after trimming whitespace; otherwise `422`). The path
cause event and the body effect event must both exist and belong to the
machine (`404 not_found` otherwise). A self-link is rejected with
`422 self_causal_link`, an already existing edge in the same direction with
`409 duplicate_causal_link`, and an edge that would close a directed cycle
with `409 causal_cycle`; none of these failures writes anything. A successful
create returns `201` with `{id, machine_id, cause_event_id, effect_event_id,
created_at}` and never modifies either event or the hash chain.

`GET` on the same path returns the links whose cause is the path event,
ordered by `(created_at, id)` ascending (possibly empty); if the cause event
does not exist on the machine it returns `404 not_found`. Links persist
across restarts; the table is created automatically at startup.

