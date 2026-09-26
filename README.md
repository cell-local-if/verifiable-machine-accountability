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

## Machine status history

Every accepted status change appends exactly one immutable record to the
`machine_status_events` table inside the same locked write transaction that
updates the machine, so the status update and its history record commit
together or not at all. History records are never updated, deleted, or
recomputed afterwards, and the table is created automatically at startup on
databases that predate the feature, without touching existing records.

`GET /machines/{machine_id}/status-history` returns the machine's own
transition records as a JSON array — empty when the machine has never changed
status. The query accepts no parameters: any query parameter is a
`422 {"error":{"code":"invalid_query"}}` raised before the machine is looked
up, a missing machine is `404 {"error":{"code":"not_found"}}`, and only `GET`
is routed (other methods return `405`). Each item carries exactly
`{id, machine_id, from_status, to_status, created_at}` in this fixed field
order, with `created_at` the UTC commit-moment stamp ending in `Z`. Records
are ordered by the actual UTC instant of `created_at`, then by `id`. The
response body is compact UTF-8 JSON terminated by a single newline and
contains no floating-point or non-finite values. The query only reads: it
never creates, updates, deletes, repairs, or normalizes status, history, or
any other record, and another machine's records never enter the result.

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

## Evidence tamper-evident chain and read-only chain verification

In addition to the standalone evidence fingerprint and the read-only evidence
integrity audit above, each machine's evidence records form their own
per-machine, tamper-evident hash chain. The chain sits on the existing
machine-event evidence path: registering evidence keeps the same request
shape and, on success, appends the new record to the tail of **this machine's**
chain. Each record returned by the create, list, and evidence export
endpoints carries, in the same positions across all three:

- `previous_evidence_id` — `null` for the machine's first evidence record,
  otherwise the id of the immediately preceding record in `(created_at, id)`
  order (the actual UTC instant, then id);
- `chain_hash` — the chain digest;
- the existing `content_hash` evidence fingerprint, unchanged.

The per-record content digest is
`SHA-256(UTF-8(compact key-sorted JSON of {id, machine_id, event_id,
evidence_type, content_hash, created_at}))`, covering the record identifier,
machine, associated event, evidence type, fingerprint, and creation time. The
`chain_hash` is `SHA-256(UTF-8("" + ":" + content_digest))` for the first
record and `SHA-256(UTF-8(previous_chain_hash + ":" + content_digest))`
thereafter. All hashes are 64-character lowercase hexadecimal strings. Each
machine is an independent chain: a record never points across machines, the
first record is rooted at the empty prefix, and no two records share a
predecessor.

The machine/event ownership lookup, the duplicate-fingerprint check, the
insert, and the chain-tail append run inside one locked write transaction —
the same lock the other per-machine chains take — so concurrent registrations
cannot lose records, skip a link, point two records at the same predecessor,
or fork the chain; the committed result is equivalent to one definite serial
order. Validation outcomes are unchanged: an illegal body is
`422` (before any path lookup), a missing machine/event or an event owned by
another machine is `404 {"error":{"code":"not_found"}}`, and a repeated
fingerprint on the same event is `409 {"error":{"code":"duplicate_evidence"}}`
with nothing written. On startup the service adds the new columns to
pre-existing databases and backfills missing chain data in stable
`(created_at, id)` order; the recomputation is deterministic, so restarting
with an already complete database performs no writes, and an empty database
can still register and query evidence.

The evidence list returns only the path machine/event's records in stable
order, an empty array (`[]`) when there are none, as compact UTF-8 JSON
terminated by a single newline; the body contains no floating-point, `-0.0`,
or non-finite value, counts stay JSON integers, and field order is stable
across calls.

`GET /machines/{machine_id}/authorization-decision-events/evidence-chain/integrity`
verifies the whole evidence chain read-only and returns the existing evidence
audit's three conclusions,
`{valid, checked_count, broken_evidence_id}`: it applies the existing checks
(event ownership, non-blank `evidence_type`, exact lowercase-hex fingerprint
compared as stored) **and** verifies each record's content digest, its
`previous_evidence_id` link, and its `chain_hash`. A complete or empty chain
reports `true`, the total count, and `null` (an empty chain reports `true`,
`0`, `null`); otherwise it reports `false`, the machine's total evidence
count, and the first record whose content, link, or chain digest does not
verify. A record whose `created_at` is corrupted still parses into the total
and is reported as broken; a missing associated record never removes the
evidence row; another machine's damaged records never affect this machine's
conclusion. Only the first error is reported and nothing is repaired. The
endpoint accepts no query parameters — any parameter is
`422 {"error":{"code":"invalid_query"}}` raised before the machine is looked
up — a missing machine returns `404 {"error":{"code":"not_found"}}` with no
partial chain conclusion, and only `GET` is routed (other methods return
`405` without reading records). The query produces no writes and repeated
calls against unchanged data return byte-identical results. Existing events,
key rotation, incidents, diagnostics, and export filtering semantics are
unchanged.

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

## Read-only global policy rule compliance export

`GET /policy-rules/compliance-export` returns a deterministic, read-only slice
of the global policy rules over a closed UTC creation-time window. It is a
separate audit entry point under the global policy rules: the policy rule
listing and the authorization evaluation entry point are unchanged, and this
endpoint never participates in an evaluation.

Both query parameters are required and validated before any rule is read, so
an invalid query never reads policy-rule data:

- `from_created_at`, `to_created_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; surrounding whitespace, offset forms such as
  `+00:00`, a missing suffix, and out-of-range calendar/time values are
  rejected); `from_created_at` must not be later than `to_created_at` (equal
  bounds are allowed). A missing, blank, malformed, or inverted bound returns
  `422 {"error":{"code":"bad_time"}}`.
- Any other query parameter returns
  `422 {"error":{"code":"invalid_query"}}`, rejected in the validation phase.
- The path accepts `GET` only; other methods return `405` without filtering,
  ordering, reading rule content, or writing anything.

The response is `{from_created_at, to_created_at, policy_rules}`; the bounds
are echoed verbatim and `policy_rules` is always present, an empty array when
the window contains nothing (including an empty database). The array contains
only global rules whose own `created_at` falls inside the closed interval
`[from_created_at, to_created_at]`. Each item has exactly the policy-rule list
endpoint fields `{id, action_type, resource_pattern, effect, priority,
created_at, updated_at}`, emitted exactly as stored with no filtering,
repair, or normalization of missing, illegal, or duplicated data, and adds no
privacy field, key content, or policy text. Rules are ordered by the actual
UTC instant of `created_at` and then by id, so an exact-second rule sorts
before any fractional-second rule of the same second. A stored `created_at`
that no longer parses never crashes the export: it deterministically sorts
after every parseable instant and therefore never falls inside a finite
window, while its stored text is left untouched.

The endpoint issues no writes, repairs, recomputations, or normalizations,
never changes an authorization result, produces byte-identical compact UTF-8
JSON (terminated by a single newline, with no floating-point, `-0.0`, or
non-finite value and a stable field order) for identical data and parameters
on repeat calls, and reads rules persisted across application restarts.

## Global policy rule tamper-evident chain

Every global policy rule belongs to one hash chain spanning the whole table
(the chain covers global rules only; there is no per-machine rule chain).
Rules are ordered by the actual UTC instant of `created_at` and then by `id`,
so an exact-second rule precedes any fractional-second rule of the same
second. For each rule:

- `content_hash` is SHA-256 of the compact, key-sorted JSON document built
  from the rule's seven visible fields `{id, action_type, resource_pattern,
  effect, priority, created_at, updated_at}`; the chain fields are never part
  of the digest;
- `chain_hash` is SHA-256 of `<previous chain_hash>:<content_hash>`, using the
  empty string as the previous chain hash for the first rule (the same
  empty-prefix and colon convention as the event chain);
- `previous_rule_id` is `null` for the first rule and the immediately
  preceding rule's id otherwise.

All digests are 64 lowercase hexadecimal characters.

`POST /policy-rules` is the only creation entry point and is unchanged apart
from recording the chain: an invalid body is still `422`, a duplicate
`(action_type, resource_pattern, priority)` triple is still
`409 {"error":{"code":"duplicate_policy_rule"}}`, and a failed creation
writes neither a rule nor a chain link. On success the duplicate check, the
chain-tail read, and the rule insert commit in a single locked write
transaction, so concurrent creations cannot lose rules, skip a link, fork the
chain, or point two rules at the same predecessor. The `201` response carries
the visible fields together with `previous_rule_id`, `content_hash`, and
`chain_hash`; these are exactly the values later shown by the chain view.

`GET /policy-rules/chain` returns the complete chain as an array — empty when
no rules exist — in the chain order described above. Each item contains the
seven visible fields plus `previous_rule_id`, `content_hash`, and
`chain_hash`, identical to the creation response. A stored `created_at` that
no longer parses never crashes the query: the rule is taken in last order,
still returned, and its stored text is emitted untouched.

`GET /policy-rules/integrity` verifies the chain read-only and returns
`{valid, checked_count, broken_policy_rule_id}` — all three fields even for an
empty table, which reports `true`, `0`, `null`. Rules are examined in chain
order (an unparseable `created_at` sorts last but still counts toward the
total and is judged by its chain values rather than crashing the scan); the
first rule whose stored `content_hash`, `previous_rule_id` link, or
`chain_hash` does not recompute makes `valid` `false` and sets
`broken_policy_rule_id` to that rule's stored id, otherwise the id is `null`.

Both read-only endpoints accept `GET` only — other methods return `405`
without reading rule content — and accept no filter parameters: any query
parameter is `422 {"error":{"code":"invalid_query"}` during validation, before
any rule is read and identically against an empty database. The endpoints are
strictly read-only (never creating, updating, deleting, repairing,
recomputing, or normalizing a rule), return byte-identical compact UTF-8 JSON
on repeat calls, and read chains persisted across restarts. On startup an
older database has the three columns added and its rows backfilled in stable
chain order; a restart over an already complete chain issues no writes, and an
empty database can both create rules and run the queries. The plain rule
listing and compliance export keep returning only their existing fields, and
neither the chain fields nor the verification queries participate in
authorization evaluation.

## Read-only global policy rule decision preview

`POST /policy-rules/decision-preview` previews which global policy rules
would decide a hypothetical `(action, resource)` request, without consulting
machine status, behavior declarations, or authorization decision events and
without writing anything. The body is a JSON object carrying exactly two
string fields, `action` and `resource`; both must stay non-empty after
surrounding whitespace is stripped. Validation runs entirely before any rule
is read, so an illegal request is rejected identically against an empty rule
table:

- any query parameter is `422 {"error":{"code":"invalid_query"}}`, checked
  before the body is even parsed;
- a body that is not a JSON object, that is missing a field, that carries an
  extra field, or that types either field wrongly is
  `422 {"error":{"code":"invalid_request"}}`;
- an `action` or `resource` that is empty after trimming is
  `422 {"error":{"code":"invalid_value"}}`.

The path accepts `POST` only; other methods return `405` without reading
rules. A failure that prevents reading the rules returns
`500 {"error":{"code":"internal_error"}}` with no partial preview.

On success the response is `{action, resource, rules, conflicts, winners,
decision}` — every group is always present. `action` and `resource` echo the
trimmed request values. `rules` retains every stored rule verbatim (the seven
visible fields, never normalized or repaired) and appends a `relation`
annotation:

- `invalid` — a stored field is missing or of an illegal shape/value (a
  missing or mistyped field, a boolean or negative priority, an effect other
  than `allow`/`deny`); illegal fields never participate in the decision;
- `unmatched` — a valid rule whose action differs or whose resource pattern
  does not match the requested resource (the usual `*`-wildcard,
  literal-otherwise semantics);
- `winner` — one of the decisive minimum-priority matching candidates;
- `overridden` — a valid matching candidate at a larger numeric priority than
  the decisive minimum;
- `conflict` — a decisive minimum-priority candidate when the minimum tier
  mixes allows and denies; the mixed tier is decided as a denial.

Details are ordered by priority ascending, then by the actual UTC instant of
`created_at` and then by id (an exact-second stamp sorts before a fractional
stamp of the same second; a damaged stamp sorts after every parseable one).
When the minimum tier mixes allows and denies, `conflicts` reports that tier
as all unordered id pairs (each pair id-ascending, the list ordered by first
then second id) and `winners` is empty; otherwise `conflicts` is empty and
`winners` lists the decisive rules as `{id, effect, priority, created_at}`
ordered by `(created_at instant, id)`. `decision` is `{allowed, reason}`:
`allowed_by_policy` when the minimum tier is all allows, `denied_by_policy`
when it contains any deny (including a mixed tier), and
`{"allowed": false, "reason": "no_matching_policy"}` with empty `winners` and
`conflicts` when no legal candidate matches.

The preview is strictly read-only: it never creates, updates, deletes,
repairs, recomputes, or normalizes a rule, and the body is compact UTF-8 JSON
terminated by a single newline, byte-identical on repeat calls against
unchanged data, including rules persisted across application restarts. Policy
creation, the rule listing, the window export, authorization evaluation, the
machine interfaces, the event chain, and the existing compliance exports are
unchanged.

## Read-only batch global policy rule decision preview

`POST /policy-rules/decision-preview/batch` runs the single-preview analysis
for a whole batch of hypothetical `(action, resource)` requests against one
same-instant snapshot of the global rule table, read once. It is a pure
read-only audit entry: it consults no machine status, behavior declarations,
or authorization decision events, writes nothing, and no input can change
another input's result. The body is a JSON object carrying exactly one field,
`requests`, an array whose items each carry exactly the string fields
`action` and `resource`; the empty array is legal and yields an empty result.
Validation runs entirely before any rule is read, and any single illegal item
rejects the whole batch with no partial analysis:

- any query parameter is `422 {"error":{"code":"invalid_query"}}`, checked
  before the body is even parsed;
- a body that is not a JSON object, that lacks or adds a top-level field,
  whose `requests` is not an array, or whose items are not objects with
  exactly `action` and `resource` is
  `422 {"error":{"code":"invalid_batch"}}`;
- an item whose `action` or `resource` is not a string, or is empty after
  trimming, is `422 {"error":{"code":"invalid_value"}}`.

The path accepts `POST` only; other methods return `405` without reading
rules. A failure that prevents reading the rules returns
`500 {"error":{"code":"internal_error"}}` with no partial analysis.

On success the response is `{batch_count, analyses, summary, decisions}` in
this fixed field order. `batch_count` is the number of submitted requests.
`analyses` has one `{input, result}` entry per request in input order, where
`input` echoes the trimmed `{action, resource}` and `result` is exactly the
single-preview payload for that pair — the same matching, conflict, winner,
override, relation-annotation, and final-decision semantics, unchanged.
`summary` counts, in the fixed key order `no_match`, `allow`, `deny`,
`conflict`, `override`: inputs with no matching candidate, inputs decided
allow, inputs decided deny, inputs reporting a conflict group, and inputs
with at least one overridden candidate. `decisions` counts final decisions
under the three existing reason keys `no_matching_policy`,
`allowed_by_policy`, `denied_by_policy`; empty categories stay present with
the JSON integer zero. The body is compact UTF-8 JSON terminated by a single
newline, free of floating-point, `-0.0`, or non-finite values, byte-identical
on repeat calls against unchanged data, including data persisted across
application restarts. The single preview, rule creation, the rule listing,
the window export, the integrity chain, and authorization evaluation are
unchanged.

## Read-only batch machine authorization evaluation

`POST /machines/{machine_id}/authorization-evaluations/batch` evaluates a
whole batch of hypothetical `(action_type, resource)` requests for one machine
against one same-instant snapshot of that machine's enabled behavior
declarations and the global policy rules, read once. It is a pure read-only
audit entry: it consults machine status, that machine's declarations, and the
global rules only, writes nothing, appends no authorization decision event,
and no input can change another input's result. The body is a JSON object
carrying exactly one field, `requests`, an array whose items each carry
exactly the string fields `action_type` and `resource`; both stay non-empty
after surrounding whitespace is stripped. The empty array is legal and
returns a complete empty result. Validation runs entirely before any machine,
declaration, or rule is read, and any single illegal item rejects the whole
batch with no partial evaluation:

- any query parameter is `422 {"error":{"code":"invalid_query"}}`, checked
  before the body is even parsed;
- a missing body, a body that is not a JSON object, a body that lacks or adds
  a top-level field, or a `requests` that is not an array is
  `422 {"error":{"code":"invalid_batch"}}`;
- an item that is not an object, that lacks or adds a field, whose
  `action_type` or `resource` is not a string, or that is empty after
  trimming is `422 {"error":{"code":"invalid_value"}}`.

When the parameters and body are legal but the machine does not exist, the
response is `404 {"error":{"code":"not_found"}}` with no partial results, even
for an empty array. The path accepts `POST` only; other methods return `405`
without reading anything. A failure that prevents reading the machine's
declarations or the rules returns
`500 {"error":{"code":"internal_error"}}` with no partial analysis.

Every item uses the single-evaluation semantics unchanged, with the existing
reason codes and no new synonyms:

- a `suspended` machine returns `{"allowed": false,
  "reason": "machine_suspended"}` for every item without reading declarations
  or policy rules;
- an active machine with no enabled declaration matching the action/resource
  returns `no_enabled_declaration`;
- otherwise a request with no matching policy rule returns
  `no_matching_policy`;
- among the matching rules the lowest priority decides, a deny at that
  priority wins over same-priority allows (`denied_by_policy`), otherwise the
  result is `allowed_by_policy`.

On success the response is `{batch_count, results, summary, decisions}` in
this fixed field order. `batch_count` is the number of submitted requests.
`results` has one `{action_type, resource, allowed, reason}` entry per
request, in the fixed order of the input array, echoing the trimmed pair.
`summary` counts the five outcome categories under the keys
`machine_suspended`, `no_enabled_declaration`, `no_matching_policy`,
`denied_by_policy`, `allowed_by_policy`; empty categories stay present with
the JSON integer zero. `decisions` counts the final outcomes under `allowed`
and `denied`, and the two always sum to `batch_count`. Only the path
machine's data enters the result. The body is compact UTF-8 JSON terminated
by a single newline, free of floating-point, `-0.0`, or non-finite values,
byte-identical on repeat calls against unchanged data, evaluable against an
empty database, and valid against data persisted across application restarts.
The single evaluation, behavior-declaration creation and listing, the rule
previews, and decision-event accounting are unchanged.

## Read-only global policy rule conflict and override audit

`GET /policy-rules/conflicts` is a separate, strictly read-only audit entry
over the global policy rules: it never participates in an authorization
evaluation and changes neither policy creation, the rule listing, the window
export, the decision previews, the machine interfaces, nor any other semantic.
The query takes no business filter parameters — any query parameter is
`422 {"error":{"code":"invalid_query"}}` during the validation phase, before
any rule is read and without database access, identically against an empty
database. The path accepts `GET` only; other methods return `405` without
reading rules, computing matches, or producing a write. A failure while
reading the rules returns `500 {"error":{"code":"internal_error"}}` with no
partial analysis.

The response always contains exactly the three arrays `{rules, conflicts,
overrides}` in this fixed field order; an empty rule table returns three empty
arrays. `rules` keeps every stored global rule with the seven visible fields
`{id, action_type, resource_pattern, effect, priority, created_at,
updated_at}` emitted exactly as stored — corrupted fields are neither repaired
nor deleted — plus a `relation` annotation:

- `invalid` — the stored action type or resource pattern is not a string, the
  effect is not exactly `allow`/`deny`, or the priority is a boolean,
  non-integer, or negative; such a record never takes part in any matching
  judgement (a damaged `id` or `created_at` neither invalidates the rule nor is
  repaired, because matching never depends on them);
- `unmatched` — a valid rule with no same-action counterpart whose resource
  pattern has a common matching scope;
- `conflict` — a valid rule in a same-action pair with intersecting resource
  patterns, equal priority, and opposite effects;
- `override` — a valid rule in a same-action pair with intersecting resource
  patterns and different priorities; the direction is carried by the
  `overrides` array.

Two valid rules become candidates only when their actions are equal and their
resource patterns can match some common resource, under the existing pattern
semantics (`*` matches any text, every other segment is literal). A common
resource exists exactly when the prefix literals are prefix-comparable (one
starts with the other), the suffix literals are suffix-comparable, and the
middle literal runs can hold inside one resource: each run pins the relative
order of its distinct literal fragments, so runs demanding contradictory
orders for the same fragments (such as `*a*b*` against `*b*a*`) never form a
candidate and produce neither a conflict nor an override. The intersecting
scope is reported as a glob under the same semantics: the longer of the two
prefix literals, the merged middle literal runs, and the longer of the two
suffix literals joined by stars. The merged run keeps every literal fragment
of both patterns in an order both can satisfy and is canonical, so the
reported intersection never depends on the order a pair is examined in; an
exact pattern's intersection with a matching glob is the exact pattern
itself.

Each conflict entry is
`{rule_ids, intersection, reason}` with the two rule ids sorted ascending and
`reason` `"same_priority_opposite_effect"`. Each override entry is
`{overriding_rule_id, overridden_rule_id, intersection, reason}`: the rule
with the smaller numeric priority covers the same intersecting scope of the
larger-priority rule, with `reason` `"lower_priority_overrides"`. The
`conflicts` list sorts by `(first id, second id)` and `overrides` by
`(overriding rule id, overridden rule id)`, so both lists are stable for the
same data. Rule details order by the actual UTC instant of `created_at` and
then by id, so an exact-second record sorts before a fractional-second record
of the same second and a damaged stamp sorts after every parseable one.

The audit is strictly read-only — it never creates, updates, deletes,
repairs, recomputes, or normalizes a rule. The body is compact UTF-8 JSON
terminated by a single newline, contains no floating-point, `-0.0`, or
non-finite value, is byte-identical on repeat calls against unchanged data,
and reads rules persisted across application restarts; old and empty databases
work without migration.

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

## Read-only desensitized privacy responsibility export

`GET /machines/{machine_id}/authorization-decision-events/privacy-responsibility/compliance-export`
returns a deterministic, read-only, machine-level privacy view of the machine's
responsibility attributions. It submits only the machine id and the time
window; both query parameters are required and validated before the machine or
any responsibility record is read, so an invalid query never touches machine or
responsibility data and still reports `422` when the machine does not exist:

- `from_created_at`, `to_created_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; surrounding whitespace, offset forms such as
  `+00:00`, a missing suffix, and out-of-range calendar/time values are
  rejected); `from_created_at` must not be later than `to_created_at` (equal
  bounds are allowed). A missing, blank, offset, malformed, or inverted bound
  returns `422 {"error":{"code":"bad_time"}}`.
- Any other query parameter returns
  `422 {"error":{"code":"invalid_query"}}`.
- The path accepts `GET` only; other methods return `405`.

After validation, a missing machine returns
`404 {"error":{"code":"not_found"}}` with no responsibility data.

The response is `{machine_id, from_created_at, to_created_at,
responsibility_assignments}`; the bounds are echoed verbatim and
`responsibility_assignments` is always present, an empty array when the window
contains nothing. The array contains only assignments whose stored
`machine_id` is the path machine and whose own `created_at` falls inside the
closed interval `[from_created_at, to_created_at]` — independently of whether
the referenced event or incident exists, is owned by another machine, or is
duplicated. Records are ordered by the actual UTC instant of `created_at` and
then by record id, so an exact-second record sorts before any
fractional-second record of the same second.

Each item keeps `{id, machine_id, event_id, incident_id, created_at,
previous_assignment_id, content_hash, chain_hash}` exactly as stored. The raw
`party` and `role` values are never emitted; their positions carry
`party_ref` and `role_ref` instead:

- `party_ref` = lowercase-hex `SHA-256(UTF-8("privacy:v1|party" + machine_id
  + party_with_surrounding_whitespace_removed))`;
- `role_ref` follows the same order and digest with the `privacy:v1|role`
  prefix.

When the stored value is not a string or is empty after trimming surrounding
whitespace, the corresponding ref is `null`; the record is still included.
Missing, misowned, duplicated, or chain-damaged related objects likewise never
cause a record to be filtered out, rewritten, or repaired, and another
machine's assignments can never enter the result. The endpoint issues no
writes, repairs, deletions, recomputations, or normalizations, produces
byte-identical output for identical data and parameters on repeat calls, reads
assignments persisted across application restarts, and adds no schema — old
and empty databases need no migration. The existing create queries, hash
chains, diagnostics terminal state, integrity summary, and other compliance
exports are unchanged.

## Machine-level privacy access registration and read-only query

Two machine-scoped entries provide a machine-level audit of privacy data
access: `privacy-accesses` registration and its `compliance-export` query. They
follow the existing machine event paths and never change the desensitized
responsibility export (its response, digest, and filtering semantics are
unchanged), `GET /health`, the hash chains, diagnostics, or the other
compliance exports.

### Registering an access

`POST /machines/{machine_id}/privacy-accesses` registers one privacy data
access. The body must be a JSON object carrying exactly:

- `accessed_at` — when the access happened, a UTC RFC 3339 date-time ending in
  `Z`;
- `window_start`, `window_end` — the desensitized export window actually used,
  both UTC RFC 3339 date-times ending in `Z`; `window_start` must not be later
  than `window_end` (equal bounds are allowed);
- `result` — exactly the string `success` or `failed`;
- `matches_count` — a non-negative integer number of records hit; a `failed`
  access returns no data and must carry `0`.

Fractional seconds are optional; offset forms such as `+00:00`, surrounding
whitespace, a missing suffix, and out-of-range calendar/time values are
rejected. A missing field, a non-object body, an illegal time, window, result,
or hit count returns `422`. Body validation runs before any path lookup, so the
same malformed payload against a non-existent machine is still `422` and leaves
no record. Responsible-party rawtext and key material are not fields: they are
never stored or echoed.

After validation, a missing machine returns
`404 {"error":{"code":"not_found"}}` and writes nothing. Success returns `201`
with `{id, machine_id, accessed_at, window_start, window_end, result,
matches_count}`: a fresh UUID `id`, the path machine id, the access time and
window echoed exactly as submitted, the result, and the hit count. A repeat
registration with the same `(accessed_at, window_start, window_end, result)`
for the same machine returns `409 {"error":{"code":"duplicate_access"}}` and
writes nothing — the hit count is not part of the access identity, so even a
different count is a duplicate; a different access time, window, or result is a
distinct record. The path accepts `POST` only; other methods return `405`.

### Registering a batch of accesses

`POST /machines/{machine_id}/privacy-accesses/batch` registers a whole batch
of privacy data accesses in one request. The body must be a JSON object
carrying a `privacy_accesses` array; each item carries the same five fields as
one single registration (`accessed_at`, `window_start`, `window_end`,
`result`, `matches_count`), under the same timestamp, window, result, and
hit-count rules. The batch is validated in full and before any machine
lookup:

- a body that is not an object, a missing or non-array `privacy_accesses`, an
  array item that is not an object, a missing item field, or a business field
  of the wrong type returns `422 {"error":{"code":"invalid_batch"}}`;
- a malformed, offset, whitespace-bearing, or out-of-range timestamp, or an
  inverted window, returns `422 {"error":{"code":"bad_time"}}`;
- a `result` other than `success`/`failed`, a negative integer hit count, or a
  `failed` access with a non-zero count returns
  `422 {"error":{"code":"invalid_value"}}`;
- any query parameter returns `422 {"error":{"code":"invalid_query"}}`.

Validation is all-or-nothing: any invalid item rejects the whole request and
writes nothing. An empty array is legal and returns
`200 {"results":[]}` without touching the database. After validation, a
missing machine returns `404 {"error":{"code":"not_found"}}` and writes
nothing.

On success the status is `200` with a `results` array aligned by position
with the request array. The whole batch enters one locked write transaction —
the same lock the single registration takes, so batches and single
registrations are serialized against each other — and items reuse the existing
duplicate check and insert in array order. The first item of a given access
identity `(accessed_at, window_start, window_end, result)` for the machine
(the hit count never participates) registers; an item that repeats a
previously registered access or an earlier item in the same batch comes back
as a duplicate without a new row and without aborting the other items. A
success result is `{"outcome":"success", id, machine_id, accessed_at,
window_start, window_end, result, matches_count}` — one full record with a
fresh UUID; a duplicate result is
`{"outcome":"duplicate_access", accessed_at, window_start, window_end,
result, matches_count}`, echoing only the submitted fields. Any persistence
failure returns `500 {"error":{"code":"internal_error"}}`; the single
transaction rolls back and leaves no partial records. The path accepts `POST`
only; other methods return `405`. Batch-written records carry the same
per-machine hash chain fields and persist across restarts.

### Querying accesses

`GET /machines/{machine_id}/privacy-accesses/compliance-export` returns a
deterministic, read-only slice of the machine's registered accesses. It
submits only the machine id and a closed window over access time; validation
completes before the machine or any access record is read:

- `from_accessed_at`, `to_accessed_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; surrounding whitespace, offset forms such as
  `+00:00`, a missing suffix, and out-of-range calendar/time values are
  rejected); `from_accessed_at` must not be later than `to_accessed_at` (equal
  bounds are allowed). A missing, blank, offset, malformed, or inverted bound
  returns `422 {"error":{"code":"bad_time"}}`.
- Any other query parameter returns
  `422 {"error":{"code":"invalid_query"}}`.
- The path accepts `GET` only; other methods return `405`.

After validation, a missing machine returns
`404 {"error":{"code":"not_found"}}` with no access data.

The response is `{machine_id, from_accessed_at, to_accessed_at,
privacy_accesses}`; the bounds are echoed verbatim and `privacy_accesses` is
always present, an empty array when the window contains nothing. The array
contains only records whose stored `machine_id` is the path machine and whose
own `accessed_at` falls inside the closed interval
`[from_accessed_at, to_accessed_at]` — membership follows the access time, not
the recorded `window_start`/`window_end`. Records are ordered by the actual UTC
instant of `accessed_at` and then by record id ascending, so an exact-second
record sorts before any fractional-second record of the same second. Each item
exposes exactly `{id, machine_id, accessed_at, window_start, window_end,
result, matches_count}` — never a responsible party or key rawtext.

Records live in their own append-only table, are strictly isolated by machine
(another machine's records can never enter a result), and are registered
automatically as a table on startup, so an empty database is fully usable and
records persist across application restarts. The query is strictly read-only:
it never creates, updates, deletes, repairs, or normalizes a record, gives
identical results on repeat calls against unchanged data, and reads records
persisted across restarts.

### Incrementally querying changes

`GET /machines/{machine_id}/privacy-accesses/changes` is a stable,
keyset-paginated incremental view of one machine's accesses. It submits only
the machine id, a page size, and an optional cursor; validation completes
before the machine or any access record is read:

- `limit` — required, a non-boolean integer from `1` to `100`. A missing,
  blank, non-integer (e.g. `3.0`, `abc`), boolean (`true`/`false`), or
  out-of-range value returns `422 {"error":{"code":"bad_limit"}}`.
- `cursor` — optional, an opaque string shaped
  `<accessed_at original text>|<record uuid>` and returned by a previous page.
  An empty, non-string, damaged, or shape-mismatching value returns
  `422 {"error":{"code":"invalid_cursor"}}` without querying the machine.
- Any other query parameter returns
  `422 {"error":{"code":"invalid_query"}}`; all three checks precede the
  machine lookup, so an invalid query against a non-existent machine is still
  `422`.
- After validation, a missing machine returns
  `404 {"error":{"code":"not_found"}}` with no access records.
- The path accepts `GET` only; other methods return `405` without reading
  records, computing a page, or writing anything.

The response is `{machine_id, limit, records, next_cursor, has_more}`;
`records` contains only records owned by the path machine, ordered by the
actual UTC instant of `accessed_at` and then by record id ascending (an
exact-second record sorts before any fractional-second record of the same
second), and is an empty array on an empty page. Each record exposes exactly
the seven registered fields `{id, machine_id, accessed_at, window_start,
window_end, result, matches_count}` — never a responsible-party rawtext, key,
policy text, or identity material.

The cursor is an exclusive position over `(accessed_at, record id)`: it points
just after a page's last record, so a page returns only records strictly after
the cursor. Repeating the same cursor against unchanged data returns the
byte-identical next page, and a record inserted (even at an earlier access
time) never causes an already-returned record to be read back. `next_cursor`
is the position after the page's last record when at least one record follows,
and `null` on the last page; `has_more` is `true` exactly when a record exists
after the current position and `false` otherwise (including on an empty page).
Cursors are stateless and add no schema: the endpoint issues only reads, never
writes or repairs, keeps strict machine isolation on empty pages, and keeps
reading records persisted across application restarts. Single and batch
registration, duplicate detection, the integrity audit, the summary, buckets,
and the existing compliance exports are unchanged.

### Summarizing accesses

`GET /machines/{machine_id}/privacy-accesses/summary` is a read-only aggregate
view of one machine's accesses, sitting alongside the audit chain rather than
inside it. It takes the same machine id and closed access-time window as the
compliance export, under the same validation rules and order:

- `from_accessed_at`, `to_accessed_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; surrounding whitespace, offset forms such as
  `+00:00`, a missing suffix, and out-of-range calendar/time values are
  rejected); the lower bound must not be later than the upper bound (equal
  bounds are allowed). A missing, blank, offset, missing-`Z`, malformed, or
  inverted bound returns `422 {"error":{"code":"bad_time"}}`.
- Any other query parameter returns
  `422 {"error":{"code":"invalid_query"}}`.
- Both checks complete before the machine or any access record is read, so an
  invalid query against a non-existent machine is still `422`.
- After validation, a missing machine returns
  `404 {"error":{"code":"not_found"}}` with no summary data.
- The path accepts `GET` only; other methods return `405` without filtering,
  counting, writing, or reading machine records.

Success returns `{machine_id, from_accessed_at, to_accessed_at, success_count,
failed_count, matches_count}`; the bounds are echoed verbatim. Totals are
computed from stored values over exactly the records owned by the path machine
whose own `accessed_at` falls inside the closed interval — membership follows
the access time, not the recorded `window_start`/`window_end`:

- `success_count` — number of in-window records whose `result` is `success`;
- `failed_count` — number of in-window records whose `result` is `failed`,
  counted separately;
- `matches_count` — sum of every in-window record's stored `matches_count`,
  totaled independently of the result (a success with zero hits contributes
  nothing; registered failed records carry zero hits).

An empty window still returns the complete envelope with all three totals at
zero. The summary is strictly machine-isolated (another machine's records can
never enter a total), strictly read-only — it never creates, updates, deletes,
repairs, or normalizes an access record or any chain field — returns
byte-identical results on repeat calls against unchanged data, and reads
records persisted across application restarts. It exposes only the machine id,
the time bounds, and the three totals — never a responsible-party rawtext or
key material — and changes nothing about single registration, batch
registration, duplicate detection, the integrity audits, the compliance
exports, or query ordering.


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
