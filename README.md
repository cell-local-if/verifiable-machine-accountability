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

All hashes are 64-character lowercase hexadecimal strings. For a decision-event
request, the machine/status lookup, the suspended-or-declaration/policy
decision, and the chain-tail append all run inside one locked write
transaction — the same lock the machine status-change endpoint takes — so
concurrent writes cannot lose events, fork, or break the chain, and a status
change and an event append always have one definite serial order. On startup
the service adds the new columns to pre-existing databases and backfills
missing chain data in `(created_at, id)` order; the recomputation is
deterministic, so restarting with an already complete database performs no
writes.

`GET /machines/{machine_id}/authorization-decision-events/integrity` verifies
the chain read-only and returns `{valid, checked_count, broken_event_id}`:
a complete or empty chain reports `true`, the total count, and `null`;
otherwise it reports `false`, the total count, and the first event whose
content hash, link, or chain hash does not verify. A missing machine returns
`404 not_found`.

## Machine suspension and reactivation

A machine is persistently either `active` (the status assigned at creation) or
`suspended`. `POST /machines/{machine_id}/status` with body
`{"status": "active"}` or `{"status": "suspended"}` changes it. The body must
be an object carrying a string `status` whose value is exactly `active` or
`suspended`; a missing body, non-object body, missing or non-string `status`,
or any other value returns `422` before the machine is looked up (so the same
payload against a non-existent machine is still `422`). A missing machine
returns `404 {"error":{"code":"not_found"}}`. Requesting the machine's current
status returns `409 {"error":{"code":"invalid_status_transition"}}` and writes
nothing, so concurrent requests for the same target status have at most one
success: the lookup, same-status check, and update run in one locked write
transaction. On success only `status` and `updated_at` change — `version`,
`public_key`, `created_at`, and every other record are untouched — and the
full updated machine object is returned with `200`. The status is stored in
the machines table and survives restarts.

While a machine is `suspended`, both
`POST /machines/{machine_id}/authorization-evaluations` and
`POST /machines/{machine_id}/authorization-decision-events` return
`{"allowed": false, "reason": "machine_suspended"}` without consulting its
behavior declarations or the policy rules. Decision-event requests still
persist one event under the usual rules, linked into the machine's event hash
chain. After reactivation the normal declaration/policy evaluation resumes;
events recorded earlier (including while suspended) keep their stored result.

The status determination, declaration/policy computation, and event append
for a decision-event request are one indivisible authorization write,
serialized against status changes by the same locked transaction, so two
concurrent requests (one status change and one event append) have a single
definite serial order and at most one can "win" a given target transition:

- when the status change commits first, the later event reads the new status
  and uses its result — after a suspension the event is `machine_suspended`,
  never a result computed from the pre-change `active` state;
- when the event append commits first, it keeps the pre-change authorization
  result and the later status change never rewrites, recomputes, or
  reinterprets the committed event; once a status update completes, later
  event requests can no longer write a result based on the old `active`
  state.

Concurrent execution never loses events, forks or breaks the hash chain, or
leaves a half-finished status or event record: the lookup, decision, and
append either commit together or leave no trace. Body validation (`422`) and
the missing-machine/path-resource (`404 not_found`) outcomes are unchanged by
this serialization and never flip error types under a status race.

## Read-only joint-write transaction diagnostics

`GET /machines/{machine_id}/diag` exposes read-only observability over the
joint write transaction shared by the status-change and decision-event
entries above. Every attempt at `POST /machines/{machine_id}/status` or
`POST /machines/{machine_id}/authorization-decision-events` produces exactly
one diagnostic record; the diagnostics feature adds no new write entry, and
the two README-documented endpoints remain the only way to change a machine's
status or create a decision event.

Both query parameters are required and validated before the machine is
looked up, so an invalid query against a non-existent machine is still `422`:

- `from`, `to` — UTC RFC 3339 date-times ending in `Z` (fractional seconds
  optional; offset forms such as `+00:00`, a missing suffix, surrounding
  whitespace, and out-of-range calendar/time values are rejected); `from`
  must not be later than `to` (equal bounds are allowed). A missing, blank,
  malformed, offset, or inverted bound returns
  `422 {"error":{"code":"bad_time"}}`.
- Any other query parameter returns `422 {"error":{"code":"invalid_query"}}`.

After validation, a missing machine returns `404 {"error":{"code":"not_found"}}`.

Success returns `200` with `{id, from, to, records, check}`: `id` is the path
machine id, `from`/`to` echo the bounds verbatim, and `check` is the machine's
current authorization-event integrity audit in the existing
`{valid, checked_count, broken_event_id}` shape. Each item in `records` is:

- `tid` — the diagnostic record id;
- `at` — the terminal timestamp of the attempt, a UTC RFC 3339 date-time
  ending in `Z`;
- `op` — `change` for a status change, `event` for a decision-event creation;
- `phase` — `started-commit` (the joint write committed) or
  `started-rollback` (it rolled back; includes the attempt start and the
  terminal state);
- `fail` — `none` for every commit; a rollback carries one stable category:
  `race` (a concurrency conflict, including the losing request of a
  same-target status race), `io` (a persistence failure), or `other` (any
  other failure or a crash rollback);
- `flags` — `[]`, or the actually-experienced `lock_wait` and `retry` flags
  in that order. A lock wait and its retries collapse into this one record;
  a rollback is never recorded as a partial/fragmented entry;
- `status` — the attempt's **transaction terminal state**, exactly
  `committed` (the joint write landed) or `rolled_back` (it did not);
- `machine_status` — the machine's own terminal state for the attempt,
  `active` or `suspended` (the empty string for an attempt whose machine did
  not exist). It is deliberately separate from the transaction outcome, so a
  rolled-back attempt can still observe a `suspended` machine (for example,
  the losing request of a same-target suspension race reports
  `status = "rolled_back"` with `machine_status = "suspended"`);
- `event` — the created decision-event id for a committed `event` attempt,
  otherwise `null`;
- `count` — the machine's decision-event count at the terminal state (the
  event's chain position for a committed event attempt);
- `check` — the event hash-chain audit snapshot
  (`{valid, checked_count, broken_event_id}`) at the terminal state.

Records contain only this operational metadata — never keys, secrets, policy
text, or identity material. `records` includes only finalized records owned by
the path machine whose `at` falls inside the closed interval `[from, to]`,
ordered by the actual UTC instant of `at` and then by `tid` (ISO-8601 text is
not chronological across the fractional-second boundary, so instants are
parsed first). If the process dies mid-attempt, the residual record is
finalized on the next startup from committed evidence — a decision event at
or after the attempt start for an event attempt, a fresh machine
`updated_at` for a status change — and classified as a commit or a crash
rollback; nothing is ever left half-finished.

The query is strictly read-only: it never writes, repairs, recomputes, or
deletes diagnostics or any business record, gives identical results on repeat
calls against unchanged data, is stable across restarts (the startup recovery
pass finalizes residuals once and an already-complete database performs no
writes), and reads data persisted across restarts. Existing databases gain
the diagnostics table safely on startup, and an empty database is fully
usable. Databases created before the terminal-state split are migrated once
on startup: the old machine-state `status` value moves to `machine_status`
and `status` becomes the `committed`/`rolled_back` outcome derived from the
record's phase, so a second restart performs no writes. `422`/`404`/`409`,
the `active`/`suspended` semantics, and `GET /health` are unchanged.

## Key rotation integrity chain

Each machine's key rotation history forms its own per-machine, tamper-evident
hash chain, following the same rules as the authorization event integrity
chain. Every record returned by the key-rotation list endpoint carries:

- `previous_rotation_id` — `null` for the machine's first rotation, otherwise
  the id of the preceding record in `(created_at, id)` order;
- `content_hash` — `SHA-256(UTF-8(compact key-sorted JSON of {id, machine_id,
  old_public_key, new_public_key, version, created_at}))`;
- `chain_hash` — `SHA-256(UTF-8("" + ":" + content_hash))` for the first
  record and `SHA-256(UTF-8(previous_chain_hash + ":" + content_hash))`
  thereafter.

All hashes are 64-character lowercase hexadecimal strings. A successful
rotation updates the machine and appends the record to the chain tail inside
a single write transaction, so concurrent rotations cannot lose records,
fork, or break the chain. On startup the service adds the new columns to
pre-existing databases and backfills missing chain data in `(created_at, id)`
order; the recomputation is deterministic, so restarting with an already
complete database performs no writes.

`GET /machines/{machine_id}/key-rotation-events/integrity` verifies the chain
read-only and returns `{valid, checked_count, broken_rotation_id}`: a complete
or empty chain reports `true`, the total count, and `null`; otherwise it
reports `false`, the total count, and the first record whose content hash,
link, or chain hash does not verify. The check never writes, repairs, or
deletes records, is stable across repeat calls and restarts, and only
examines the path machine's records. A missing machine returns
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

## Persistent exception-handling incident registrations

`POST /machines/{machine_id}/authorization-decision-events/{event_id}/incidents`
registers a persistent exception-handling incident on one authorization
decision event. The body requires:

- `incident_type`, `summary` — strings that stay non-empty after trimming
  surrounding whitespace; missing, non-string, or blank values return `422`.
  Body validation runs before any path lookup, so a malformed payload is `422`
  even when the machine or event does not exist.

After validation, the machine and event must both exist and the event must
belong to the machine; otherwise the endpoint returns
`404 {"error":{"code":"not_found"}}`. Success returns `201` with
`{id, machine_id, event_id, incident_type, summary, status, created_at}`: a
fresh UUID `id`, `status` always `"open"`, and `created_at` a UTC RFC 3339
date-time ending in `Z`. The trimmed `incident_type` and `summary` are stored
as supplied. A repeat registration with the same `incident_type` and
`summary` on the same event returns
`409 {"error":{"code":"duplicate_incident"}}` and writes nothing; the same
combination is allowed on a different event.

`GET` on the same path returns the event's incidents in `created_at`, then
`id` order (`[]` when none), with the same fields as the create response; a
missing machine, missing event, or an event owned by another machine returns
`404 {"error":{"code":"not_found"}}`. Incidents live in their own table, are
strictly isolated by machine (another machine can never read or register
against an event it does not own), survive application restarts, and neither
endpoint ever writes or modifies events, evidence, hash chains, or causal
links.

## Incident status transitions and immutable history

An incident moves through a fixed state machine: `open` (the status assigned
at registration) → `acknowledged` → `resolved`.

`POST /machines/{machine_id}/authorization-decision-events/{event_id}/incidents/{incident_id}/status`
advances one incident. The body is `{"status": "..."}` where `status` must be
the string `"acknowledged"` or `"resolved"`; a missing field, a non-string
value, or any other string returns `422`, and body validation runs before any
path lookup (so a malformed payload is `422` even when nothing on the path
exists). After validation, the machine, event, and incident must all exist and
belong to one another; a missing one or an ownership mismatch returns
`404 {"error":{"code":"not_found"}}`. Only `open → acknowledged` and
`acknowledged → resolved` are legal; any other transition returns
`409 {"error":{"code":"invalid_status_transition"}}` and writes nothing. On
success the incident's `status` is updated and one history record is appended
inside a single transaction (so the two can never diverge, and concurrent
transitions cannot both observe the same prior status); the response is `200`
with the full updated incident (the same fields as the incident create/list
endpoints).

`GET .../incidents/{incident_id}/status-history` returns the incident's
immutable transition records in `created_at`, then `id` order (`[]` for an
incident that has never moved). The machine, event, and incident are validated
exactly as for the POST, including the same `404 not_found` outcomes. Each
entry contains exactly `{id, machine_id, event_id, incident_id, from_status,
to_status, created_at}`: a fresh UUID `id` and `created_at` a UTC RFC 3339
date-time ending in `Z`. History rows are append-only — they are never updated
or deleted, so a rejected transition leaves the incident status and the history
both untouched, and neither endpoint ever writes or modifies events, evidence,
hash chains, or causal links. History is strictly isolated to its own incident
(another incident, event, or machine can never read it) and persists across
application restarts.

## Incident responsibility assignments

`POST /machines/{machine_id}/authorization-decision-events/{event_id}/incidents/{incident_id}/responsibility-assignments`
attributes responsibility for one registered incident to a party in a role.
The body requires:

- `party`, `role` — strings that stay non-empty after trimming surrounding
  whitespace; missing, non-string, or blank values return `422`. Body
  validation runs before any path lookup, so a malformed payload is `422`
  even when the machine, event, or incident does not exist.

After validation, the machine, event, and incident must all exist and belong
to one another; a missing one or an ownership mismatch returns
`404 {"error":{"code":"not_found"}}`. Success returns `201` with
`{id, machine_id, event_id, incident_id, party, role, created_at}`: a fresh
UUID `id` and `created_at` a UTC RFC 3339 date-time ending in `Z`. The trimmed
`party` and `role` are stored as supplied. Assigning the same `party` and
`role` to the same incident twice returns
`409 {"error":{"code":"duplicate_assignment"}}` and writes nothing; the same
pair is allowed on a different incident.

`GET` on the same path returns the incident's assignments in `created_at`,
then `id` order (`[]` when none), with the same fields as the create response
and the same `404 not_found` semantics. Assignments live in their own table,
are strictly isolated by machine and incident (another machine, event, or
incident can never read them), survive application restarts, and neither
endpoint ever writes or modifies incidents, events, evidence, hash chains, or
causal links.

## Read-only incident lifecycle and responsibility-closure audit

`GET /machines/{machine_id}/authorization-decision-events/incidents/integrity`
audits every registered incident owned by one machine, in `(created_at, id)`
order, and returns `{valid, checked_count, broken_incident_id}`. A missing
machine returns `404 {"error":{"code":"not_found"}}`; a machine with no
incidents returns `200` with `true`, `0`, and `null`.

An incident passes only when:

- its `event_id` resolves to an existing authorization decision event that
  belongs to the same machine (missing or foreign-owned events fail);
- its `status` and its status history, ordered by `(created_at, id)`, correspond
  exactly: `open` has no history, `acknowledged` has exactly
  `open -> acknowledged`, and `resolved` has that edge followed by
  `acknowledged -> resolved`; every history record's `machine_id`, `event_id`,
  and `incident_id` must also match the incident;
- every responsibility assignment's `machine_id`, `event_id`, and
  `incident_id` matches the incident, its `party` and `role` each stay
  non-empty after trimming surrounding whitespace, and no two records share the
  same trimmed `(party, role)` pair;
- a `resolved` incident has at least one sound responsibility assignment
  (the responsibility closure is complete). Open and acknowledged incidents
  require none.

The first failing incident sets `valid` to `false` and `broken_incident_id` to
its id; `checked_count` is always the machine's total incident count, and
`broken_incident_id` is `null` when every incident passes. Only the path
machine's incidents are checked, so damage under another machine never fails
this audit. The query is strictly read-only — it never writes, repairs,
deletes, or normalizes incidents, history, assignments, events, evidence, hash
chains, or causal links — gives identical results on repeat calls against
unchanged data, and audits data persisted across application restarts.

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

## Read-only global policy rule listing

`GET /policy-rules` returns every global policy rule, or `[]` when none exist.
Each item contains exactly the persisted `{id, action_type, resource_pattern,
effect, priority, created_at, updated_at}` values, with no normalization or
repair of stored values. Rules are ordered by the actual UTC instant of
`created_at`, then by `id` ascending: because ISO-8601 text ordering is not
chronological across the fractional-second boundary (`...:00.5Z` sorts before
`...:00Z` as text since `.` precedes `Z`), stamps are parsed to UTC instants
first, so an exact-second record sorts before any fractional record of the same
second. The endpoint is strictly read-only — it never writes, updates, deletes,
normalizes, or repairs a rule — takes no part in authorization evaluation,
returns identical results on repeat calls against unchanged data, and reads
rules persisted across application restarts.

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

## Read-only causal-link compliance export

`GET /machines/{machine_id}/authorization-decision-events/causal-links/compliance-export`
returns a deterministic, read-only, machine-level compliance slice of causal
links, keyed on each link's own creation time rather than on an event window.
Both query parameters are required and validated before the machine is looked
up, so a missing, malformed, or inverted parameter returns `422` even when the
machine does not exist, and an invalid query never reads machine data:

- `from_created_at`, `to_created_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; surrounding whitespace, offset forms such as
  `+00:00`, a missing suffix, and out-of-range calendar/time values are
  rejected); `from_created_at` must not be later than `to_created_at` (equal
  bounds are allowed). A missing, blank, malformed, or inverted bound returns
  `422 {"error":{"code":"bad_time"}}`.
- Any other query parameter returns
  `422 {"error":{"code":"invalid_query"}}`.
- The path accepts `GET` only; other methods return `405`.

After validation, a missing machine returns
`404 {"error":{"code":"not_found"}}`.

The response is `{machine_id, from_created_at, to_created_at, causal_links}`;
the bounds are echoed verbatim and `causal_links` is always present, an empty
array when the window contains nothing. The array contains only links whose
stored `machine_id` is the path machine and whose own `created_at` falls inside
the closed interval `[from_created_at, to_created_at]` — independently of when
either endpoint event was created and of whether the endpoint events fall in
any event window. Each item has exactly the causal-link list endpoint fields
`{id, machine_id, cause_event_id, effect_event_id, created_at}`, ordered by the
actual UTC instant of `created_at` and then by id, so an exact-second link
sorts before any fractional-second link of the same second.

Links are exported exactly as stored: a cause or effect event that is missing,
owned by another machine, duplicated, or otherwise damaged is included
verbatim — the events table is never consulted and the link is never rewritten,
filtered out, or repaired. Another machine's links can never enter the result.
The endpoint issues no writes, repairs, deletions, recomputations, or
normalizations, never creates causal links or performs cycle detection or path
traversal, produces identical output for identical data and parameters on
repeat calls, reads links persisted across application restarts, and needs no
migration on an empty or old database (it adds no schema).

## Read-only evidence compliance export

`GET /machines/{machine_id}/authorization-decision-events/evidence/compliance-export`
returns a deterministic, read-only compliance slice of one machine's evidence
records. Both query parameters are required and validated before the machine
is looked up, so a missing, malformed, or inverted parameter returns `422`
even when the machine does not exist:

- `from_created_at`, `to_created_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; offset forms such as `+00:00` are rejected);
  `from_created_at` must not be later than `to_created_at` (equal bounds are
  allowed).

After validation, a missing machine returns `404 {"error":{"code":"not_found"}}`.

The response is `{machine_id, from_created_at, to_created_at, evidence}`.
`evidence` contains only the machine's evidence records whose `created_at`
falls inside the closed interval `[from_created_at, to_created_at]`. Each item
has exactly the same fields as the evidence list endpoint, ordered by
`created_at`, then `id`, and the array is empty (never omitted) when the
window contains nothing. Records are exported exactly as stored: a record
whose `event_id` is damaged or points at another machine's event is included
verbatim, never rewritten, filtered out, or repaired.

The endpoint issues no writes, repairs, or deletions, never returns another
machine's evidence, produces identical output for identical data and
parameters on repeat calls, and reads data persisted across application
restarts.



## Read-only incident status history compliance export

`GET /machines/{machine_id}/incident-status-events/compliance-export`
returns a deterministic, read-only, machine-level compliance slice of one
machine's incident status transition history. Both query parameters are
required and validated before the machine is looked up, so a missing,
malformed, or inverted parameter returns `422` even when the machine does not
exist:

- `from_created_at`, `to_created_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; offset forms such as `+00:00` are rejected);
  `from_created_at` must not be later than `to_created_at` (equal bounds are
  allowed).

After validation, a missing machine returns `404 {"error":{"code":"not_found"}}`.

The response is `{machine_id, from_created_at, to_created_at, status_history}`.
`status_history` contains only status events owned by the path machine whose
`created_at` falls inside the closed interval `[from_created_at, to_created_at]`,
aggregated across all of the machine's incidents. Each item has exactly
`{id, machine_id, event_id, incident_id, from_status, to_status, created_at}`,
ordered by `created_at`, then `id`, and the array is empty (never omitted)
when the window contains nothing.

Records are exported exactly as stored: a missing or foreign-owned incident or
event reference, or a corrupt status edge, is included verbatim — never
rewritten, filtered out, or repaired. The endpoint issues no writes, repairs,
or deletions, never returns another machine's status history, produces
identical output for identical data and parameters on repeat calls, and reads
data persisted across application restarts.

## Read-only machine-level accountability compliance export

`GET /machines/{machine_id}/accountability/compliance-export` returns a
deterministic, read-only, machine-level closed-loop accountability slice in
one response. Both query parameters are required and validated before the
machine is looked up, so a missing, malformed, or inverted parameter returns
`422` even when the machine does not exist, and an invalid query never reads
machine data:

- `from_created_at`, `to_created_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; surrounding whitespace, offset forms such as
  `+00:00`, a missing suffix, and out-of-range calendar/time values are
  rejected); `from_created_at` must not be later than `to_created_at` (equal
  bounds are allowed). A missing, blank, malformed, or inverted bound returns
  `422 {"error":{"code":"bad_time"}}`.
- Any other query parameter returns
  `422 {"error":{"code":"invalid_query"}}`.

After validation, a missing machine returns
`404 {"error":{"code":"not_found"}}` with no partial closure data.

The response is `{machine_id, from_created_at, to_created_at, events,
evidence, incidents, status_history, responsibility_assignments}` — the
bounds are echoed verbatim and every group is present, an empty array when
the window contains nothing. Each group contains only records owned by the
path machine whose own `created_at` falls inside the closed interval, each
ordered by the actual UTC instant of `created_at` and then by record id, so
an exact-second record sorts before any fractional-second record of the same
second (ISO text alone is not chronological across that boundary).

- `events` — the machine's authorization decision events, with the
  authorization result (`allowed`, `reason`) and the full integrity-chain
  fields (`previous_event_id`, `content_hash`, `chain_hash`). Event
  association requires both endpoints to be events in this export's event
  set.
- `evidence` — evidence records with their raw `content_hash` fingerprint,
  following the existing machine/event ownership.
- `incidents` — registered incidents with the registration content
  (`incident_type`, `summary`) and the current lifecycle `status`.
- `status_history` — the machine's incident status transitions, each with
  its before/after (`from_status`, `to_status`) statuses, following the
  existing machine/incident ownership.
- `responsibility_assignments` — responsibility attributions with the
  responsible party, role, and the assignment chain fields
  (`previous_assignment_id`, `content_hash`, `chain_hash`), following the
  existing machine/incident ownership.

The evidence, status-history, and assignment groups follow the existing
machine/entity ownership relations of their owning incidents; whether the
referenced event itself falls in the event window never removes a record.
Every record is exported exactly as stored: when an associated object is
missing or belongs to another machine, the record is still included verbatim
— integrity failures never filter, rewrite, repair, or normalize it. The
endpoint issues no writes, repairs, deletions, recomputations, or
normalizations of the machine or any accountability record, produces
byte-identical output for identical data and parameters on repeat calls, and
reads records persisted across application restarts. `GET /health`, machine
start/stop, authorization evaluation, the event chain, the existing exports,
and the diagnostics error semantics are unchanged.

## Read-only privacy-preserving responsibility compliance export

`GET /machines/{machine_id}/privacy-responsibility/compliance-export` returns a
deterministic, read-only, privacy-preserving slice of one machine's
responsibility assignments. Both query parameters are required and validated
before the machine is looked up, so a missing, malformed, or inverted
parameter returns `422` even when the machine does not exist, and an invalid
query never reads machine data:

- `from_created_at`, `to_created_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; surrounding whitespace, offset forms such as
  `+00:00`, a missing suffix, and out-of-range calendar/time values are
  rejected); `from_created_at` must not be later than `to_created_at` (equal
  bounds are allowed). A missing, blank, malformed, or inverted bound returns
  `422 {"error":{"code":"bad_time"}}`.
- Any other query parameter returns
  `422 {"error":{"code":"invalid_query"}}`.
- The path accepts `GET` only; other methods return `405` without filtering or
  computing digests.

After validation, a missing machine returns
`404 {"error":{"code":"not_found"}}` with no responsibility data.

The response is `{machine_id, from_created_at, to_created_at,
responsibility_assignments}`; the bounds are echoed verbatim and
`responsibility_assignments` is always present, an empty array when the window
contains nothing. The array contains only assignments whose stored
`machine_id` is the path machine and whose own `created_at` falls inside the
closed interval `[from_created_at, to_created_at]`, ordered by the actual UTC
instant of `created_at` and then by id (an exact-second record sorts before
any fractional-second record of the same second).

Each item keeps the existing assignment fields, but `party` and `role` are
replaced in place by `party_ref` and `role_ref`:

- `party_ref` — `SHA-256(UTF-8("privacy:v1|party" + machine_id +
  trimmed_party))` as 64 lowercase hexadecimal characters, where
  `trimmed_party` is the stored party value with surrounding whitespace
  stripped;
- `role_ref` — the same computation with the `privacy:v1|role` prefix.

Binding the machine id into the digest means the same party or role name on
two machines yields two distinct references. The raw party and role text never
appears in the response; neither do public keys, secrets, policy text, or
identity material. When a stored value is not a string or is empty after
trimming, its reference is `null` and the record is still included. The record
id, `machine_id`, `event_id`, `incident_id`, `created_at`, and the three
responsibility-chain fields (`previous_assignment_id`, `content_hash`,
`chain_hash`) are emitted exactly as stored.

Assignments are exported without consulting the incidents or events tables: a
missing, misowned, duplicated, or otherwise damaged event or incident
reference, field value, or chain link is included verbatim — never rewritten,
filtered out, or repaired — and another machine's assignments can never enter
the result. The endpoint issues no writes, repairs, deletions,
recomputations, or normalizations, produces byte-identical output for
identical data and parameters on repeat calls, reads assignments persisted
across application restarts, and needs no migration on an empty or old
database (it adds no schema). The existing machine, event, evidence, incident,
status-history, and responsibility-assignment creation and query semantics —
including the event hash chain, diagnostic terminal state, integrity summary,
and the existing compliance exports — are unchanged.

## Read-only machine-level integrity summary

`GET /machines/{machine_id}/integrity-summary` returns a deterministic,
read-only aggregate of the four existing machine-level integrity audits in one
response. The caller submits only the path machine id; the endpoint accepts no
query parameters, and any extra parameter returns
`422 {"error":{"code":"invalid_query"}}` before the machine is looked up, so an
invalid query against a non-existent machine is still `422`. After validation,
a missing machine returns `404 {"error":{"code":"not_found"}}` with no summary
data. The path accepts `GET` only; other methods return `405`.

The response is `{machine_id, valid, events, rotations, evidence, incidents}`:

- `valid` — `true` only when all four blocks pass; the first anomaly found by
  any block makes it `false`;
- `events` — the authorization decision event chain audit in the existing
  `{valid, checked_count, broken_event_id}` shape (content hash, previous-event
  link, chain hash);
- `rotations` — the key rotation chain audit in the existing
  `{valid, checked_count, broken_rotation_id}` shape (raw rotation fields,
  previous-rotation link, chain hash);
- `evidence` — the evidence audit in the existing
  `{valid, checked_count, broken_evidence_id}` shape (event ownership, evidence
  type, exact lowercase-hex fingerprint, compared as stored with no case
  folding);
- `incidents` — the incident lifecycle and responsibility-closure audit in the
  existing `{valid, checked_count, broken_incident_id}` shape (event ownership,
  status history, responsibility closure), keeping the first incident id found.

Each block runs its existing audit in that audit's stable
`(created_at, id)` order and counts only records stored under the path machine
name, so damaged records owned by another machine never affect the result. A
machine with no records still returns the complete response: all four blocks
report `checked_count` `0`, `valid` `true`, and a `null` broken id. The summary
exposes only the existing integrity conclusions and audit ids — no privacy
fields, policy text, or key content. The query is strictly read-only: it never
creates, updates, deletes, repairs, recomputes, or normalizes a machine or any
accountability record, produces identical results on repeat calls against
unchanged data, and reads data persisted across application restarts. It adds
no schema. `GET /health`, machine start/stop, authorization evaluation, the
event chain, incident handling, and the existing compliance exports are
unchanged.
