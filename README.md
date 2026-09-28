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

## Immutable authorization decision basis snapshots

Every new authorization decision event saves, in the same locked write
transaction that appends the event, an immutable snapshot of the basis the
decision was made on — the machine status, the enabled behavior declarations
and their matching scope, the policy candidate rules and their relations, and
the committed result, reason, and capture moment. The snapshot and the event
commit atomically or leave no trace; snapshots are append-only and are never
updated, recomputed, or reconstructed from later data. The real write entry
stays the public authorization-decision-event creation endpoint — no
bypassing business write entry is added. Only visible business fields are
stored: never machine/public keys or policy text beyond the rules' existing
visible columns.

`GET /machines/{machine_id}/authorization-decision-events/{event_id}/decision-basis`
is the read-only query under the machine authorization decision event path;
the caller submits only the path machine id and event id. Any query
parameter, a repeated parameter, or a request body is
`422 {"error":{"code":"invalid_query"}}` validated before the machine, event,
or snapshot is read (the same malformed request against a non-existent
machine is still `422`). After validation, a missing machine, a missing
event, or an event owned by another machine returns
`404 {"error":{"code":"not_found"}}`. An event that exists but was committed
before this feature existed (and so has no historical snapshot) returns
`404 {"error":{"code":"snapshot_not_found"}}` — the basis is never fabricated
from current declarations or rules. Only `GET` is routed; `HEAD` and every
other method return `405` without reading the snapshot, recomputing a
decision, or writing anything. A real failure while reading the event or
snapshot returns `500 {"error":{"code":"internal_error"}}` with no event
summary and no partial basis.

A successful response is exactly five groups in this fixed order:

- `event_summary` — one object with the event's committed result and reason
  (`allowed`, `reason`), its capture moment (`created_at`), and its chain
  fields (`previous_event_id`, `content_hash`, `chain_hash`);
- `status_basis` — `{machine_id, status, captured_at, declarations_read,
  policies_read}`: the machine status the decision used, the capture moment,
  and whether declarations and policy rules were read at all. A suspended
  machine's snapshot records `status "suspended"` with both read flags
  `false` (no declarations and no policy were consulted); an active machine
  has `declarations_read true`, and `policies_read` is `true` only when at
  least one enabled declaration matched (the decision stops at
  `no_enabled_declaration` before the rules otherwise);
- `declaration_basis` — `{read, declarations}`: the machine's enabled
  declarations for the requested action that participated in the judgement,
  each carrying its visible fields `{id, action_type, resource_pattern,
  enabled, created_at, updated_at}` plus a boolean `matched` for whether its
  pattern matched the requested resource. A suspended machine records `read
  false` and an empty array; a disabled or other-action declaration never
  appears;
- `policy_candidates` — `{read, candidates, winners, conflicts}`. `read` is
  true only when the declaration gate passed. Each candidate keeps the
  rule's seven visible fields plus a `relation` under the existing priority
  semantics: `winner` (a decisive minimum-priority candidate), `overridden`
  (a matching candidate at a larger numeric priority), `conflict` (a
  minimum-priority candidate when the decisive tier mixes allow and deny;
  the tier is then decided as a denial), or `unmatched` (a same-action rule
  whose pattern did not match). Candidates are ordered by priority
  ascending, then by the actual UTC instant of `created_at`, then by id.
  `winners` lists the decisive rules as `{id, effect, priority,
  created_at}` and is empty for a mixed tier; `conflicts` lists that tier as
  id-ascending unordered pairs. Rules for other actions never participate and
  never appear. Every collection is present even when empty;
- `decision` — `{allowed, reason}`, exactly the result committed on the
  event; the stored basis is emitted verbatim and the decision is never
  recomputed, so it can never disagree with the event's stored result.

Records are output in stable order as compact UTF-8 JSON terminated by a
single newline, with no floating-point, `-0.0`, or non-finite value. The
captured document is stored once and re-emitted byte-for-byte, so repeated
queries of one event are byte-identical; snapshots persist across application
restarts and are strictly isolated to the path machine (another machine's
event id on the path is a `404 not_found`). The query is strictly read-only:
it never creates, updates, deletes, repairs, or normalizes an event,
snapshot, declaration, or rule. On startup the snapshot table is created
safely on databases that predate the feature (an older database's existing
events simply answer `snapshot_not_found`), an empty database can both
create events and query their bases, and the event list, hash chain, evidence
chain, incident responsibility, privacy exports, diagnostics, and the health
check are unchanged.

### Independent read-only consistency audit

`GET /machines/{machine_id}/authorization-decision-events/{event_id}/decision-basis/integrity`
is the independent verification sub-entry under the snapshot path: it does not
re-emit the snapshot, it checks it, read-only, against the committed event. It
accepts only the path machine id and event id — any query parameter, a repeated
parameter, or a request body is
`422 {"error":{"code":"invalid_query"}}`, validated before the machine, event,
or snapshot is read (the same malformed request against a non-existent machine
is still `422`). After validation a missing machine, a missing event, or an
event owned by another machine returns
`404 {"error":{"code":"not_found"}}` with no audit conclusion. Only `GET` is
routed; `HEAD` and every other method return `405` without reading or writing.
A real failure while reading the event, snapshot, declarations, or rules
returns `500 {"error":{"code":"internal_error"}}` with no conclusion and no
partial result.

A successful response carries exactly four conclusions in this fixed order:

- `valid` — `true` only when the event summary, status basis, declaration
  basis, policy candidates, and final decision all agree with one another and
  with the committed event;
- `checked_count` — `0` when the event has no snapshot row and `1` when the one
  snapshot row exists (even if that snapshot is damaged). Another machine's
  snapshot is never counted, and there is at most one snapshot per event;
- `broken_basis_id` — the path event id whenever the conclusion is false,
  `null` on success;
- `reason` — `null` on success; the fixed `snapshot_not_found` for a missing
  row; otherwise the stable category of the first anomaly found.

The checks verify, in a fixed order: the document parses to the five expected
groups with sound field types and ordering; the event summary corresponds
verbatim to the event's committed `allowed` result, `reason`, capture moment
(`created_at`), and chain fields (`previous_event_id`, `content_hash`,
`chain_hash`) — nothing is recomputed or filled in; the machine status is
reconstructed **as of the event** from the machine's own
`machine_status_events` transition history rather than its current row; a
suspended machine's basis explicitly records that declarations and policy
were not read, while an active machine's read flags match its status and the
declaration gate; the declaration basis names exactly the enabled
declarations that participated in that action and resource judgement (a
missing, disabled, other-action, or superfluous record is an inconsistency,
and each `matched` scope flag must be correct); the policy candidates'
winner/overridden/conflict/unmatched (and invalid) relations, winner group,
conflict pairs, and priority-first ordering must follow the priorities as of
the event and support the final judgement; and the final decision equals the
event's committed `allowed` flag and `reason`.

The event-time status is rebuilt by an independent read of the machine's
transition history: only a transition whose actual creation instant is not
later than the capture instant (`created_at <= capture`) forms the state
then, same-instant transitions apply in ascending `id` order, and with no
qualifying transition the machine's creation default `active` carries
forward unchanged. A snapshot that merely agrees with the machine's current
status while contradicting that event-time state is rejected. The status
basis is judged history, then state, then read flags, giving three fixed
categories:

- `status_history_invalid` — the history is misattributed, illegal, or not
  rebuildable: a non-text or unparseable `created_at` (the as-of boundary is
  undecidable), a non-text `id` (same-instant ordering unstable), a status
  outside `active`/`suspended`, a self edge, or a `from_status` that does not
  continue the state the earlier transitions establish. Every stored
  transition of the machine is counted, including one committed after the
  capture (the history is one append-only chain whose edges must continue one
  another); a damaged record is reported, never crashed on, repaired,
  rewritten, or recomputed;
- `status_state_mismatch` — the history rebuilds soundly but the rebuilt
  event-time status differs from the snapshot's recorded `status`;
- `read_flags_mismatch` — the state agrees but the recorded
  `declarations_read`/`policies_read` flags (and the declaration group's own
  `read` flag) contradict the gate in force then: a suspended machine reads
  neither declarations nor policy, an active machine reads declarations, and
  policy is read only when at least one enabled declaration matched.

A real failure while reading the status history (like the event, snapshot,
declarations, or rules) returns `500 internal_error` with no conclusion and
no partial result, and no status history is ever fabricated. Relations are
audited against the rules and declarations that already existed when the
event committed, so later declarations or rules never retroactively break a
faithful historical snapshot; the audit never recomputes or repairs a record
and never fabricates a missing basis. An event without a snapshot row
answers `valid false`, `checked_count 0`, `broken_basis_id` equal to the
event id, and `reason "snapshot_not_found"`. A damaged or contradictory
snapshot answers `valid false`, `checked_count 1`, the event id, and the
stable first-anomaly category, using exactly these buckets for damaged
content:

- `malformed_document` — the document does not parse (invalid JSON or a
  non-finite constant), does not parse to an object, or has a document-level
  shape anomaly: a missing, extra, or reordered top-level group, a wrong
  field type on any group (the event summary, status basis, declaration
  basis, policy candidates, or decision), an anomalous collection shape, or
  an anomalous set/record ordering. Every such structural or ordering
  anomaly collapses to this one stable category instead of naming the group;
- `malformed_decision` — the structure is complete but the final decision
  field is illegal or contradicts the event's committed result (including a
  decision the rebuilt basis cannot support).

Content-level correspondence checks that are neither structural nor the
final decision keep their own mismatch category in examination order:
`event_summary_mismatch`, the status triplet (`status_history_invalid`,
`status_state_mismatch`, `read_flags_mismatch`, checked history then state
then read flags), `status_basis_mismatch`,
`declaration_basis_mismatch`/`declaration_match_mismatch`, and
`policy_candidate_mismatch`/`policy_winner_mismatch`/
`policy_conflict_mismatch`. The conclusion is compact
UTF-8 JSON in fixed field order terminated by one newline, with no
floating-point or non-finite value, so identical stored data audits
byte-for-byte identically, the result persists across restarts, and machine
isolation is enforced. Event creation, the event list, the hash chain,
evidence chain, incident responsibility, privacy exports, diagnostics, and
health-check semantics are unchanged.

### Incremental decision-basis snapshot changes query

`GET /machines/{machine_id}/authorization-decision-events/decision-basis/changes`
is the stable, keyset-paginated incremental view over one machine's
immutable decision-basis snapshots — the `changes` sub-entry under the
decision-basis path (a sibling of the single-snapshot query and its
integrity audit). It is a separate, strictly read-only audit entry point:
it changes neither event creation, the single-snapshot query, the
consistency audit, the hash chain, the privacy exports, the compliance
exports, diagnostics, the authorization evaluation, nor the health check,
and it adds no on-disk format (cursors are stateless), so old and empty
databases work unchanged. The caller submits only the path machine id, a
page size, and an optional cursor — no business filter parameters and no
request body. Validation completes before the machine or any snapshot is
read:

- `limit` — required, a non-boolean integer from `1` to `100`. A missing,
  blank, fractional (e.g. `1.0`), boolean (`true`/`false`), non-decimal, or
  out-of-range value returns `422 {"error":{"code":"bad_limit"}}`.
- `cursor` — optional, an opaque string shaped
  `<created_at original text>|<event id>` and returned by a previous page.
  An empty value, a missing separator, or an empty segment returns
  `422 {"error":{"code":"invalid_cursor"}}`; the timestamp segment is
  original stored text rather than re-parsed (the split is on the last
  separator), because a snapshot whose stored `created_at` no longer parses
  stays pageable. A well-shaped cursor whose `(created_at, event id)`
  position cannot be located among the path machine's stored snapshots —
  naming another machine's snapshot, a deleted snapshot, or an event that
  has no snapshot — is rejected as `422 {"error":{"code":"invalid_cursor"}}`
  after the snapshots are read, with no partial page, so a parameter error
  always takes priority over the machine lookup.
- Any other query parameter, a repeated `limit`/`cursor`, or a request
  carrying a body returns `422 {"error":{"code":"invalid_query"}}` in the
  validation phase; every validation error is answered without reading any
  machine or snapshot.
- The path accepts `GET` only; `HEAD` and every other method return `405`
  without reading a snapshot, computing a page, or writing anything.
- A valid query against a machine that does not exist returns
  `404 {"error":{"code":"not_found"}}` carrying no records.
- A failure that prevents reading the snapshots or the machine, or a stored
  snapshot document that is no longer decodable JSON, returns
  `500 {"error":{"code":"internal_error"}}` with no partial page — the row
  is neither repaired nor skipped.

The success object carries exactly `{machine_id, limit, records,
next_cursor, has_more}` in this fixed field order; the path machine id and
page size are echoed verbatim, and an empty database or a machine with no
snapshots returns the complete empty page. Only events that already have a
snapshot enter the page: an event committed before the basis feature (or
whose snapshot row is absent) is not fabricated, is not returned, and never
enters a cursor. Each record is exactly `{group, record}`: the fixed tag
`"decision_basis"` and the same five groups the single-snapshot query
exposes — `event_summary`, `status_basis`, `declaration_basis`,
`policy_candidates`, and `decision` — emitted exactly as captured, with no
recomputation, audit, repair, or shape normalization, so a JSON-valid but
tampered document is still re-emitted verbatim. Records are ordered by the
snapshot creation event's actual UTC instant of `created_at` and then by
event id ascending, so an exact-second record sorts before any
fractional-second record of the same second; a stored `created_at` that no
longer parses never crashes the query — its original text is kept and it
deterministically sorts after every parseable instant.

The cursor is an exclusive position over `(created_at, event id)`: it
points just after a page's last snapshot, so the page returns only
snapshots strictly after it. Repeating the same cursor against unchanged
data returns the byte-identical next page, and a newly captured snapshot
whose sort position is earlier never makes an already-returned record
resurface while the current page still follows the stable order.
`next_cursor` is the position after the page's last snapshot when at least
one follows and `null` otherwise; `has_more` is `true` exactly then and
`false` on both an empty and a last page. The query is strictly read-only
and machine-isolated — another machine's snapshots never enter the result
— and the body is compact UTF-8 JSON terminated by a single newline, free
of floating-point, `-0.0`, or non-finite values, byte-identical on repeat
calls against unchanged data, and readable for snapshots persisted across
application restarts.


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

## Read-only machine status-history compliance export

`GET /machines/{machine_id}/status-history/compliance-export` returns a
deterministic, read-only slice of one machine's status transition history
over a closed UTC time window. Only `GET` is routed (other methods,
including `HEAD`, return `405` without reading records). Both query
parameters are required and every check runs before the machine is looked
up, so a parameter error against a non-existent machine is still `422`:

- `from_created_at`, `to_created_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; offset forms such as `+00:00`, surrounding
  whitespace, and out-of-range calendar/time values are rejected);
  `from_created_at` must not be later than `to_created_at` (equal bounds are
  allowed). A missing, blank, malformed, or inverted bound returns
  `422 {"error":{"code":"bad_time"}}`.
- An unknown parameter, a repeated `from_created_at`/`to_created_at`, or a
  request body returns `422 {"error":{"code":"invalid_query"}}`.

After validation, a missing machine returns
`404 {"error":{"code":"not_found"}}`; a real read failure returns
`500 {"error":{"code":"internal_error"}}` with no partial records.

The response is `{machine_id, from_created_at, to_created_at,
status_history}` in this fixed key order; the machine id and the original
bound text are echoed verbatim, and `status_history` is an empty array
(never omitted) for an empty window, a machine with no history, or an empty
database. The array contains only the path machine's existing transition
records whose own `created_at` falls inside the closed interval, each with
exactly `{id, machine_id, from_status, to_status, created_at}` as the list
endpoint returns it, ordered by the actual UTC instant of `created_at` and
then by record id — an exact-second record sorts before any
fractional-second record of the same second. A stored `created_at` that no
longer parses sorts after every parseable instant and therefore never
enters a finite window; damaged field values inside in-window records are
kept exactly as stored, never filtered, repaired, or normalized. The query
adds no persistence surface, repeats byte-identically, reads across
restarts, and never returns another machine's records.

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

## Read-only stable machine key-rotation incremental query

`GET /machines/{machine_id}/key-rotation-events/changes` is the stable,
keyset-paginated incremental view over one machine's rotation records — the
`changes` sub-entry under the machine key-rotation record path. It is a
separate, strictly read-only audit entry point: it changes neither rotation
creation, the rotation list, the integrity audit, the compliance export, nor
any other machine interface, and it adds no on-disk format (cursors are
stateless), so old and empty databases work unchanged. The caller submits
only the path machine id, a page size, and an optional cursor — no business
filter parameters and no request body. Validation completes before the
machine or any rotation record is read:

- `limit` — required, a non-boolean integer from `1` to `100`. A missing,
  blank, fractional (e.g. `1.0`), boolean (`true`/`false`), non-decimal,
  or out-of-range value returns `422 {"error":{"code":"bad_limit"}}`.
- `cursor` — optional, an opaque string shaped
  `<created_at original text>|<rotation id>` and returned by a previous
  page. An empty, non-string, or shape-mismatching value returns
  `422 {"error":{"code":"invalid_cursor"}}`; the timestamp segment is
  original stored text rather than re-parsed, because a rotation whose
  stored `created_at` no longer parses stays pageable. A well-shaped cursor
  whose `(created_at, id)` position cannot be located among the path
  machine's stored rotations (including a position naming another
  machine's record) is rejected as
  `422 {"error":{"code":"invalid_cursor"}}` while the records are read, so
  a parameter error always takes priority over the machine lookup.
- Any other query parameter, a repeated `limit`/`cursor`, or a request
  carrying a body returns `422 {"error":{"code":"invalid_query"}}` in the
  validation phase; every validation error is answered without reading any
  machine or rotation record.
- The path accepts `GET` only; other methods return `405` without reading
  records, computing a page, or writing anything.
- A valid query against a machine that does not exist returns
  `404 {"error":{"code":"not_found"}}` carrying no rotation records.
- A failure that prevents reading the records or the machine returns
  `500 {"error":{"code":"internal_error"}}` with no partial page.

The success object carries exactly `{machine_id, limit, records,
next_cursor, has_more}` in this fixed field order; an empty database and a
machine with no rotations return the complete empty page. `records`
contains only rotations owned by the path machine, each with exactly the
complete fields of the rotation list view — the six visible fields `{id,
machine_id, old_public_key, new_public_key, version, created_at}` emitted
exactly as stored plus `previous_rotation_id`, `content_hash`, and
`chain_hash` — with no filtering, repair, or normalization. Records are
ordered by the actual UTC instant of `created_at` and then by record id
ascending, so an exact-second record sorts before any fractional-second
record of the same second. A stored `created_at` that no longer parses
never crashes the query: the record is kept with its stored value and
deterministically sorts after every parseable instant rather than being
deleted; damaged, missing, misowned, or duplicated chain values are
likewise emitted exactly as stored.

The cursor is an exclusive position over `(created_at, rotation id)`: it
points just after a page's last record, so the page returns only records
strictly after it. Repeating the same cursor against unchanged data returns
the byte-identical next page, and a newly inserted rotation whose sort
position is earlier never makes an already-returned record resurface while
the current page still follows the stable order. `next_cursor` is the
position after the page's last record when at least one record follows and
`null` otherwise; `has_more` is `true` exactly when a record exists after
the current position and `false` on both an empty and a last page.

The query is strictly read-only and machine-isolated: it never creates,
updates, deletes, repairs, recomputes, or normalizes a rotation, and another
machine's records never enter the result. The body is compact UTF-8 JSON
terminated by a single newline, free of floating-point, `-0.0`, or
non-finite values, byte-identical on repeat calls against unchanged data,
and readable for records persisted across application restarts.

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
transitions cannot both observe the same prior status); the record is linked
into the machine's incident-status-event hash chain in the same transaction,
so the status, the record, and its chain link commit together or not at all.
The response is `200` with the full updated incident (the same fields as the
incident create/list endpoints).

`GET .../incidents/{incident_id}/status-history` returns the incident's
immutable transition records in `created_at`, then `id` order (`[]` for an
incident that has never moved). The machine, event, and incident are validated
exactly as for the POST, including the same `404 not_found` outcomes. Each
entry contains exactly `{id, machine_id, event_id, incident_id, from_status,
to_status, created_at, previous_status_event_id, content_hash, chain_hash}`:
a fresh UUID `id`, `created_at` a UTC RFC 3339 date-time ending in `Z`, and
the per-machine chain fields described below. History rows are append-only —
they are never updated or deleted, so a rejected transition leaves the
incident status and the history both untouched, and neither endpoint ever
writes or modifies events, evidence, hash chains, or causal links. History is
strictly isolated to its own incident (another incident, event, or machine
can never read it) and persists across application restarts.

## Incident status event integrity chain

Each machine's incident status events form a per-machine, tamper-evident hash
chain. Every record returned by the status-history endpoint carries:

- `previous_status_event_id` — `null` for the machine's first record,
  otherwise the id of the preceding record in `(created_at, id)` order;
- `content_hash` — `SHA-256(UTF-8(compact key-sorted JSON of {id, machine_id,
  event_id, incident_id, from_status, to_status, created_at}))`;
- `chain_hash` — `SHA-256(UTF-8("" + ":" + content_hash))` for the first
  record and `SHA-256(UTF-8(previous_chain_hash + ":" + content_hash))`
  thereafter.

All hashes are 64-character lowercase hexadecimal strings. A successful
status transition updates the incident's `status`, appends the history
record, and links it to the machine's chain tail inside one locked write
transaction, so the status, the record, and its chain link commit together or
leave no trace, and concurrent transitions cannot lose records, fork the
chain, or break a link. On startup the service adds the new columns to
pre-existing databases and backfills missing chain data in `(created_at, id)`
order; the recomputation is deterministic, so restarting with an already
complete database performs no writes.

`GET /machines/{machine_id}/incident-status-events/integrity` verifies the
chain read-only and returns `{valid, checked_count,
broken_status_event_id}`: a complete or empty chain reports `true`, the total
count, and `null`; otherwise it reports `false`, the total count, and the
first record whose creation moment, id, previous-record link, content hash,
or chain hash does not verify. The endpoint accepts no query parameters and
no request body — either is a `422 {"error":{"code":"invalid_query"}}`
reported before the machine is looked up — and only `GET` is routed (other
methods return `405`). A missing machine returns
`404 {"error":{"code":"not_found"}}`, and a read failure returns
`500 {"error":{"code":"internal_error"}}`; none of these carries a partial
conclusion. Only the path machine's records are examined, so another
machine's damaged records never change the result.

## Read-only stable incident status-history incremental query

`GET /machines/{machine_id}/incident-status-events/changes` is the stable,
keyset-paginated incremental view over one machine's incident status
history — the `changes` sub-entry under the machine incident status
history path. It is a separate, strictly read-only audit entry point: it
changes neither incident creation, status transitions, the per-incident
history window export, the integrity chain, nor any other machine
interface, and it adds no on-disk format (cursors are stateless), so old
and empty databases work unchanged. The caller submits only the path
machine id, a page size, and an optional cursor — no business filter
parameters and no request body. Validation completes before the machine
or any status event is read:

- `limit` — required, a non-boolean integer from `1` to `100`. A missing,
  blank, fractional (e.g. `1.0`), boolean (`true`/`false`), non-decimal,
  or out-of-range value returns `422 {"error":{"code":"bad_limit"}}`.
- `cursor` — optional, an opaque string shaped
  `<created_at original text>|<status event id>` and returned by a
  previous page. An empty, non-string, or shape-mismatching value returns
  `422 {"error":{"code":"invalid_cursor"}}`; the timestamp segment is
  original stored text rather than re-parsed, because a record whose
  stored `created_at` no longer parses stays pageable. A well-shaped
  cursor whose `(created_at, id)` position cannot be located among the
  path machine's stored records (including a position naming another
  machine's record) is rejected as
  `422 {"error":{"code":"invalid_cursor"}}` while the records are read, so
  a parameter error always takes priority over the machine lookup.
- Any other query parameter, a repeated `limit`/`cursor`, or a request
  carrying a body returns `422 {"error":{"code":"invalid_query"}}` in the
  validation phase; every validation error is answered without reading
  any machine or status event.
- The path accepts `GET` only; `HEAD` and other methods return `405`
  without reading history, computing a page, or writing anything.
- A valid query against a machine that does not exist returns
  `404 {"error":{"code":"not_found"}}` carrying no records and no status.
- A failure that prevents reading the records or the machine returns
  `500 {"error":{"code":"internal_error"}}` with no partial page,
  conclusion, or status.

The success object carries exactly `{machine_id, limit, records,
next_cursor, has_more}` in this fixed field order; the path machine id and
the page size are echoed back, and an empty database or a machine with no
history returns the complete empty page. Each item of `records` is
exactly `{group, record}` with the fixed group tag `status_history`, and
`record` carries exactly the complete status-history fields — the seven
transition fields `{id, machine_id, event_id, incident_id, from_status,
to_status, created_at}` emitted exactly as stored plus
`previous_status_event_id`, `content_hash`, and `chain_hash` — with no
filtering, repair, or normalization. Items are ordered by the actual UTC
instant of `created_at`, then by the category tag, then by record id
ascending, so an exact-second record sorts before any fractional-second
record of the same second. A stored `created_at` that no longer parses
never crashes the query: the record is kept with its stored value and
deterministically sorts after every parseable instant rather than being
deleted; damaged, missing, misowned, or duplicated references and chain
values are likewise emitted exactly as stored.

The cursor is an exclusive position over `(created_at, status event id)`:
it points just after a page's last record, so the page returns only
records strictly after it. Repeating the same cursor against unchanged
data returns the byte-identical next page, and a newly inserted record
whose sort position is earlier never makes an already-returned record
resurface while the current page still follows the stable order.
`next_cursor` is the position after the page's last record when at least
one record follows and `null` otherwise; `has_more` is `true` exactly in
that case and `false` on both an empty and a last page.

The query is strictly read-only and machine-isolated: it never creates,
updates, deletes, repairs, recomputes, or normalizes a record, and another
machine's records never enter the result. The body is compact UTF-8 JSON
terminated by a single newline, free of floating-point, `-0.0`, or
non-finite values, byte-identical on repeat calls against unchanged data,
and readable for records persisted across application restarts.

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

## Read-only event-scoped responsibility chain verification

`GET /machines/{machine_id}/authorization-decision-events/{event_id}/responsibility-assignments/integrity`
verifies the complete responsibility-assignment hash chain owned by the path
machine, read-only. The caller submits only the machine id and event id in the
path — no query parameters and no request body; either is a
`422 {"error":{"code":"invalid_query"}}` raised before the machine is looked
up or any responsibility record is read, and non-`GET` methods return `405`.
The event id only confirms the attribution context: a missing machine, a
missing event, or an event owned by another machine is
`404 {"error":{"code":"not_found"}}` with no partial chain conclusion.

The response is `{valid, checked_count, broken_assignment_id}`. An empty chain
reports `true`, `0`, and `null`. `checked_count` always counts the machine's
full chain, damaged rows included. Records are examined in the order of the
actual UTC instant of `created_at` and then `id`; the first record's
`previous_assignment_id` must be empty, each later record's must point at the
immediately preceding record, and each record's stored `content_hash` and
`chain_hash` must match the digests recomputed under the creation-time rules.
The first mismatching record makes `valid` `false` and is reported by its
stored id verbatim; later records cannot change that attribution. A damaged
`created_at`, id, digest, or reference never crashes the query and is never
repaired, normalized, or recomputed for storage; another machine's damaged
records never affect this machine's result. A real read failure returns
`500 {"error":{"code":"internal_error"}}` with no partial conclusion. The
query never creates, updates, deletes, repairs, recomputes, or normalizes any
record, returns byte-identical results on repeat calls against unchanged data,
and reads records persisted across application restarts.

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

## Read-only stable machine incident-view incremental query

`GET /machines/{machine_id}/authorization-decision-events/incidents/changes`
is the stable, keyset-paginated incremental view over one machine's closed
incident loop — incident registrations, incident status history, and
responsibility assignments — the `changes` sub-entry under the machine
incident path. It is a separate, strictly read-only audit entry point: it
changes neither incident registration, status transitions, responsibility
assignments, the per-incident history and assignment listings, the integrity
audits, nor the compliance exports, and it adds no on-disk format (cursors
are stateless), so old and empty databases work unchanged. The caller
submits only the path machine id, a page size, and an optional cursor — no
business filter parameters and no request body. Validation completes before
the machine or any record is read:

- `limit` — required, a non-boolean integer from `1` to `100`. A missing,
  blank, fractional (e.g. `1.0`), boolean (`true`/`false`), non-decimal,
  or out-of-range value returns `422 {"error":{"code":"bad_limit"}}`.
- `cursor` — optional, an opaque string shaped
  `<created_at original text>|<group>|<record id>` and returned by a
  previous page. An empty, non-string, or shape-mismatching value returns
  `422 {"error":{"code":"invalid_cursor"}}`; the timestamp segment is
  original stored text rather than re-parsed, because a record whose
  stored `created_at` no longer parses stays pageable. A well-shaped
  cursor whose `(created_at, group, id)` position cannot be located among
  the path machine's stored records (including a position naming another
  machine's record, or the right `(created_at, id)` pair under the wrong
  group tag) is rejected as `422 {"error":{"code":"invalid_cursor"}}`
  while the records are read, so a parameter error always takes priority
  over the machine lookup.
- Any other query parameter, a repeated `limit`/`cursor`, or a request
  carrying a body returns `422 {"error":{"code":"invalid_query"}}` in the
  validation phase; every validation error is answered without reading any
  machine or record.
- The path accepts `GET` only; `HEAD` and other methods return `405`
  without reading records, computing a page, or writing anything.
- A valid query against a machine that does not exist returns
  `404 {"error":{"code":"not_found"}}` carrying no records.
- A failure that prevents reading the records or the machine returns
  `500 {"error":{"code":"internal_error"}}` with no partial page.

The success object carries exactly `{machine_id, limit, records,
next_cursor, has_more}` in this fixed field order; the path machine id and
the page size are echoed back, and an empty database or a machine with no
incident records returns the complete empty page. `records` merges the
three incident record groups into one page. Each item is exactly `{group,
record}` with the fixed group tag — `incidents`, `status_history`, or
`responsibility_assignments` — and `record` carries the complete record
exactly as the corresponding endpoint emits it: the seven incident fields
`{id, machine_id, event_id, incident_type, summary, status, created_at}`;
the seven transition fields plus `previous_status_event_id`,
`content_hash`, and `chain_hash` for status history; the seven attribution
fields plus `previous_assignment_id`, `content_hash`, and `chain_hash` for
assignments — all emitted exactly as stored, with no filtering, repair, or
normalization. Items are ordered by the actual UTC instant of
`created_at`, then by the group tag, then by record id ascending, so an
exact-second record sorts before any fractional-second record of the same
second. A stored `created_at` that no longer parses never crashes the
query: the record is kept with its stored value and deterministically
sorts after every parseable instant rather than being deleted; damaged,
missing, misowned, or duplicated references and chain values are likewise
emitted exactly as stored.

The cursor is the exclusive position `<created_at original text>|<group>|
<record id>` pointing just after a page's last record, so a page returns
only records strictly after it: repeating the same cursor against
unchanged data returns the byte-identical next page, and a newly inserted
record whose sort position is earlier never makes an already-returned
record resurface while the current page still follows the stable order.
`next_cursor` carries the position after the page's last record only when
a record follows (`null` on the last page), and `has_more` is true exactly
in that case — false on an empty or last page, including an empty
database. The query is strictly read-only and machine isolated: it never
creates, updates, deletes, repairs, recomputes, or normalizes a record,
and another machine's records never enter the page. The body is compact
UTF-8 JSON terminated by a single newline, free of floating-point, `-0.0`,
or non-finite values, byte-identical on repeat calls against unchanged
data, and readable across application restarts.

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

## Read-only stable global policy rule incremental query

`GET /policy-rules/changes` is the stable, keyset-paginated incremental
view over the global policy rules — the `changes` sub-entry under the
public global policy rule path. It is a separate, strictly read-only audit
entry point: it never participates in an authorization evaluation and
changes neither policy creation, the rule listing, the window export, the
chain verification, the decision previews, nor any other semantic. The
caller submits only a page size and an optional cursor — no business
filter parameters and no request body. Validation completes before any
rule is read:

- `limit` — required, a non-boolean integer from `1` to `100`. A missing,
  blank, fractional (e.g. `1.0`), boolean (`true`/`false`), non-decimal,
  or out-of-range value returns `422 {"error":{"code":"bad_limit"}}`.
- `cursor` — optional, an opaque string shaped
  `<created_at original text>|<rule id>` and returned by a previous page.
  An empty, non-string, or shape-mismatching value returns
  `422 {"error":{"code":"invalid_cursor"}}`; the timestamp segment is
  original stored text rather than re-parsed, because a rule whose stored
  `created_at` no longer parses stays pageable. A well-shaped cursor whose
  `(created_at, id)` position cannot be located among the stored rules is
  rejected as `422 {"error":{"code":"invalid_cursor"}}` while the rules are
  read.
- Any other query parameter, a repeated `limit`/`cursor`, or a request
  carrying a body returns `422 {"error":{"code":"invalid_query"}}` in the
  validation phase; every validation error is answered without reading
  any rule, identically against an empty database.
- The path accepts `GET` only; other methods return `405` without reading
  rules, computing a page, or writing anything.
- A failure that prevents reading the rules returns
  `500 {"error":{"code":"internal_error"}}` with no partial page.

The success object carries exactly `{limit, records, next_cursor,
has_more}` in this fixed field order; an empty database returns the
complete empty page. `records` contains only global policy rules, each
with exactly the complete fields of the rule-chain query — the seven
visible fields `{id, action_type, resource_pattern, effect, priority,
created_at, updated_at}` emitted exactly as stored plus
`previous_rule_id`, `content_hash`, and `chain_hash` — with no filtering,
repair, or normalization. Rules are ordered by the actual UTC instant of
`created_at` and then by rule id ascending, so an exact-second record
sorts before any fractional-second record of the same second. A stored
`created_at` that no longer parses never crashes the query: the record is
kept with its stored value and deterministically sorts after every
parseable instant rather than being deleted.

The cursor is an exclusive position over `(created_at, rule id)`: it
points just after a page's last record, so the page returns only records
strictly after it. Repeating the same cursor against unchanged data
returns the byte-identical next page, and a newly inserted rule whose
sort position is earlier never makes an already-returned record resurface
while the current page still follows the stable order. `next_cursor` is
the position after the page's last record when at least one record
follows and `null` otherwise; `has_more` is `true` exactly when a record
exists after the current position and `false` on both an empty and a
last page. Cursors are stateless and add no schema.

The query is strictly read-only: it never creates, updates, deletes,
repairs, recomputes, or normalizes a rule. The body is compact UTF-8 JSON
terminated by a single newline, free of floating-point, `-0.0`, or
non-finite values, byte-identical on repeat calls against unchanged data,
and readable for rules persisted across application restarts.

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

## Read-only stable authorization decision event incremental query

`GET /machines/{machine_id}/authorization-decision-events/changes` is the
stable, keyset-paginated incremental view over one machine's authorization
decision events — the `changes` sub-entry under the machine authorization
decision event path. It is a separate, strictly read-only audit entry point:
it changes neither event registration, the event list, the privacy export,
the compliance export, the hash chain, the integrity audit, nor the
authorization evaluation semantics, and it adds no on-disk format (cursors
are stateless), so old and empty databases work unchanged. The caller
submits only the path machine id, a page size, and an optional cursor — no
business filter parameters and no request body. Validation completes before
the machine or any event is read:

- `limit` — required, a non-boolean integer from `1` to `100`. A missing,
  blank, fractional (e.g. `1.0`), boolean (`true`/`false`), non-decimal,
  or out-of-range value returns `422 {"error":{"code":"bad_limit"}}`.
- `cursor` — optional, an opaque string shaped
  `<created_at original text>|<event id>` and returned by a previous page.
  An empty, non-string, or shape-mismatching value returns
  `422 {"error":{"code":"invalid_cursor"}}`; the timestamp segment is
  original stored text rather than re-parsed, because an event whose stored
  `created_at` no longer parses stays pageable. A well-shaped cursor whose
  `(created_at, id)` position cannot be located among the path machine's
  stored events (including a position naming another machine's record) is
  rejected as `422 {"error":{"code":"invalid_cursor"}}` after the events are
  read, with no partial page, so a parameter error always takes priority
  over the machine lookup.
- Any other query parameter, a repeated `limit`/`cursor`, or a request
  carrying a body returns `422 {"error":{"code":"invalid_query"}}` in the
  validation phase; every validation error is answered without reading any
  machine or event.
- The path accepts `GET` only; other methods return `405` without reading
  events, computing a page, or writing anything.
- A valid query against a machine that does not exist returns
  `404 {"error":{"code":"not_found"}}` carrying no events.
- A failure that prevents reading the events or the machine returns
  `500 {"error":{"code":"internal_error"}}` with no partial page.

The success object carries exactly `{machine_id, limit, records,
next_cursor, has_more}` in this fixed field order; the path machine id and
the page size are echoed verbatim, and an empty database or a machine with
no events returns the complete empty page. `records` contains only events
owned by the path machine, each with exactly the complete fields of the
event list view — the seven visible fields `{id, machine_id, action_type,
resource, allowed, reason, created_at}` emitted exactly as stored plus
`previous_event_id`, `content_hash`, and `chain_hash` — with no filtering,
repair, or normalization, so chain-damaged or misowned content is kept
exactly as stored. Records are ordered by the actual UTC instant of
`created_at` and then by event id ascending, so an exact-second record
sorts before any fractional-second record of the same second. A stored
`created_at` that no longer parses never crashes the query: the record is
kept with its stored value and deterministically sorts after every
parseable instant rather than being deleted, rewritten, or skipped.

The cursor is an exclusive position over `(created_at, event id)`: it
points just after a page's last record, so the page returns only records
strictly after it. Repeating the same cursor against unchanged data returns
the byte-identical next page, and a newly inserted event whose sort
position is earlier never makes an already-returned record resurface while
the current page still follows the stable order. `next_cursor` is the
position after the page's last record when at least one record follows and
`null` otherwise; `has_more` is `true` exactly when a record exists after
the current position and `false` on both an empty and a last page.

The query is strictly read-only and machine-isolated: it never creates,
updates, deletes, repairs, recomputes, or normalizes an event, and another
machine's records never enter the result. The body is compact UTF-8 JSON
terminated by a single newline, free of floating-point, `-0.0`, or
non-finite values, byte-identical on repeat calls against unchanged data,
and readable for records persisted across application restarts.

## Read-only desensitized authorization decision event privacy export

`GET /machines/{machine_id}/authorization-decision-events/privacy-export`
returns a deterministic, read-only, desensitized privacy view of one machine's
authorization decision events. It is a separate compliance sub-entry under the
machine authorization decision event path; the event registration, list,
compliance export, hash chain, integrity audit, and authorization evaluation
semantics are unchanged, and this endpoint never participates in an
evaluation. The caller submits only the path machine id and the two required
bounds; no business filter parameters or request body are accepted. Validation
completes before the machine or any event is read:

- `from_created_at`, `to_created_at` — UTC RFC 3339 date-times ending in `Z`
  (fractional seconds optional; surrounding whitespace, offset forms such as
  `+00:00`, a missing suffix, and out-of-range calendar/time values are
  rejected); the lower bound must not be later than the upper bound (equal
  bounds are allowed). A missing, blank, offset, missing-`Z`, malformed,
  out-of-range, or inverted bound returns `422 {"error":{"code":"bad_time"}}`.
- Any other query parameter returns
  `422 {"error":{"code":"invalid_query"}}`, rejected in the validation phase
  before the machine is looked up and without reading any event data.
- After the parameters validate, a missing machine returns
  `404 {"error":{"code":"not_found"}}` with no event data.
- The path accepts `GET` only; other methods return `405` without reading,
  digesting, or writing anything.

The response is `{machine_id, from_created_at, to_created_at, events}`; the
bounds are echoed verbatim and `events` is always present, an empty array when
the window contains nothing (including a machine with no events). The array
contains only events whose stored `machine_id` is the path machine and whose
own `created_at` falls inside the closed interval
`[from_created_at, to_created_at]`; another machine's records can never enter.
Events are ordered by the actual UTC instant of `created_at` and then by `id`
ascending, so an exact-second record sorts before any fractional-second record
of the same second.

Each item keeps the stored identifier, machine, decision result
(`allowed`, `reason`), creation time, previous-event link, and both chain
digests exactly as stored: `{id, machine_id, action_ref, resource_ref, allowed,
reason, created_at, previous_event_id, content_hash, chain_hash}`. The raw
`action` and `resource` values are never emitted; their positions carry
`action_ref` and `resource_ref` instead:

- `action_ref` = lowercase-hex `SHA-256(UTF-8("privacy:v1|action" + machine_id
  + action_with_surrounding_whitespace_removed))`;
- `resource_ref` follows the same connection order and digest, with only the
  prefix changed to `privacy:v1|resource`.

Both digests are 64 lowercase hexadecimal characters; when the stored value is
not a string or is empty after trimming surrounding whitespace, the
corresponding ref is `null` while the record is still included. Missing,
misowned, duplicated, or chain-damaged events are likewise exported verbatim —
never filtered out, repaired, recomputed, or normalized. The query is strictly
read-only: it never creates, updates, deletes, repairs, recomputes, or
normalizes any event, and identical data and parameters return byte-identical
compact UTF-8 JSON on repeat calls (terminated by a single newline, with no
floating-point, `-0.0`, or non-finite value and a stable field order),
including data persisted across application restarts; it adds no schema.

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

## Read-only stable causal-link incremental query

`GET /machines/{machine_id}/authorization-decision-events/causal-links/changes`
is the stable, keyset-paginated incremental view over one machine's causal
links — the `changes` sub-entry under the machine causal-link path. It is a
separate, strictly read-only audit entry point: it changes neither causal-link
creation, the single-event causal trace, the causal-link compliance export, the
causal-link integrity audit, nor the health check, and it adds no on-disk
format (cursors are stateless), so old and empty databases work unchanged. The
caller submits only the path machine id, a page size, and an optional cursor —
no business filter parameters and no request body. Validation completes before
the machine or any causal link is read:

- `limit` — required, a non-boolean integer from `1` to `100`. A missing,
  blank, fractional (e.g. `1.0`), boolean (`true`/`false`), non-decimal, or
  out-of-range value returns `422 {"error":{"code":"bad_limit"}}`.
- `cursor` — optional, an opaque string shaped
  `<created_at original text>|<causal link id>` and returned by a previous
  page. An empty, non-string, or shape-mismatching value returns
  `422 {"error":{"code":"invalid_cursor"}}`; the timestamp segment is original
  stored text rather than re-parsed, because a link whose stored `created_at`
  no longer parses stays pageable. A well-shaped cursor whose
  `(created_at, id)` position cannot be located among the path machine's
  stored links (including a position naming another machine's link) is
  rejected as `422 {"error":{"code":"invalid_cursor"}}` after the links are
  read, with no partial page, so a parameter error always takes priority over
  the machine lookup.
- Any other query parameter, a repeated `limit`/`cursor`, or a request
  carrying a body returns `422 {"error":{"code":"invalid_query"}}` in the
  validation phase; every validation error is answered without reading any
  machine or link.
- The path accepts `GET` only; `HEAD` and other methods return `405` without
  reading links, computing a page, or writing anything.
- A valid query against a machine that does not exist returns
  `404 {"error":{"code":"not_found"}}` carrying no link records.
- A failure that prevents reading the links or the machine returns
  `500 {"error":{"code":"internal_error"}}` with no partial page.

The success object carries exactly `{machine_id, limit, records, next_cursor,
has_more}` in this fixed field order; the path machine id and the page size are
echoed back, and an empty database or a machine with no causal links returns
the complete empty page. Each item of `records` is exactly `{group, record}`
with the fixed group tag `causal_links`, and `record` carries exactly the
complete causal-link fields — the five fields `{id, machine_id, cause_event_id,
effect_event_id, created_at}` emitted exactly as stored — with no filtering,
repair, or normalization. A dangling, duplicated, or otherwise damaged
endpoint reference is kept exactly as stored; the events table is never
consulted. Items are ordered by the actual UTC instant of `created_at`, then by
the category tag, then by link id ascending, so an exact-second link sorts
before any fractional-second link of the same second. A stored `created_at`
that no longer parses never crashes the query: the link is kept with its stored
value and deterministically sorts after every parseable instant rather than
being deleted.

The cursor is an exclusive position over `(created_at, causal link id)`: it
points just after a page's last link, so the page returns only records strictly
after it. Repeating the same cursor against unchanged data returns the
byte-identical next page, and a newly created link whose sort position is
earlier never makes an already-returned record resurface while the current page
still follows the stable order. `next_cursor` carries the position after the
page's last record only when a record follows and is `null` otherwise;
`has_more` is `true` exactly in that case and `false` on both an empty and a
last page.

The query is strictly read-only and machine-isolated: it never creates,
updates, deletes, repairs, recomputes, or normalizes a link, and another
machine's links never enter the result. The body is compact UTF-8 JSON
terminated by a single newline, free of floating-point, `-0.0`, or non-finite
values, byte-identical on repeat calls against unchanged data, and readable for
links persisted across application restarts.

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

## Read-only stable evidence compliance incremental query

`GET /machines/{machine_id}/authorization-decision-events/evidence/compliance-export/changes`
is the stable, keyset-paginated incremental view over one machine's evidence
records — the `changes` sub-entry under the evidence compliance-export path.
It is a separate, strictly read-only audit entry point: it changes neither
evidence registration, the evidence list, the window compliance export, the
event hash chain, nor the integrity audits, and it adds no on-disk format
(cursors are stateless), so old and empty databases work unchanged. The
caller submits only the path machine id, a page size, and an optional cursor
— no business filter parameters and no request body. Validation completes
before the machine or any evidence record is read:

- `limit` — required, a non-boolean integer from `1` to `100`. A missing,
  blank, fractional (e.g. `1.0`), boolean (`true`/`false`), non-decimal,
  or out-of-range value returns `422 {"error":{"code":"bad_limit"}}`.
- `cursor` — optional, an opaque string shaped
  `<created_at original text>|<record id>` and returned by a previous
  page. An empty, non-string, or shape-mismatching value returns
  `422 {"error":{"code":"invalid_cursor"}}`; the timestamp segment is
  original stored text rather than re-parsed, because an evidence record
  whose stored `created_at` no longer parses stays pageable. A well-shaped
  cursor whose `(created_at, id)` position cannot be located among the path
  machine's stored evidence (including a position naming another machine's
  record) is rejected as `422 {"error":{"code":"invalid_cursor"}}` while the
  records are read, so a parameter error always takes priority over the
  machine lookup.
- Any other query parameter, a repeated `limit`/`cursor`, or a request
  carrying a body returns `422 {"error":{"code":"invalid_query"}}` in the
  validation phase; every validation error is answered without reading any
  machine or evidence record.
- The path accepts `GET` only; other methods return `405` without reading
  evidence, computing a page, or writing anything.
- A valid query against a machine that does not exist returns
  `404 {"error":{"code":"not_found"}}` carrying no evidence records.
- A failure that prevents reading the records or the machine returns
  `500 {"error":{"code":"internal_error"}}` with no partial page.

The success object carries exactly `{machine_id, limit, records,
next_cursor, has_more}` in this fixed field order, echoing the path machine
id and page size; an empty database and a machine with no evidence return
the complete empty page. `records` contains only evidence owned by the path
machine, each with exactly the complete fields of the evidence window
export — the six visible fields `{id, machine_id, event_id, evidence_type,
content_hash, created_at}` emitted exactly as stored plus
`previous_evidence_id` and `chain_hash` — with no filtering, repair, or
normalization. Records are ordered by the actual UTC instant of
`created_at` and then by record id ascending, so an exact-second record
sorts before any fractional-second record of the same second. A stored
`created_at` that no longer parses never crashes the query: the record is
kept with its stored value and deterministically sorts after every
parseable instant rather than being deleted or skipped; damaged, missing,
misowned, or duplicated values are likewise emitted exactly as stored.

The cursor is an exclusive position over `(created_at, record id)`: it
points just after a page's last record, so the page returns only records
strictly after it. Repeating the same cursor against unchanged data returns
the byte-identical next page, and a newly registered evidence record whose
sort position is earlier never makes an already-returned record resurface
while the current page still follows the stable order. `next_cursor` is the
position after the page's last record when at least one record follows and
`null` otherwise; `has_more` is `true` exactly when a record exists after
the current position and `false` on both an empty and a last page.

The query is strictly read-only and machine-isolated: it never creates,
updates, deletes, repairs, recomputes, or normalizes an evidence record,
and another machine's evidence never enters the result. The body is
compact UTF-8 JSON terminated by a single newline, free of
floating-point, `-0.0`, or non-finite values, byte-identical on repeat
calls against unchanged data, and readable for records persisted across
application restarts.



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

## Read-only single-event accountability trace

`GET /machines/{machine_id}/authorization-decision-events/{event_id}/accountability-trace`
returns a deterministic, read-only, single-event closed-loop trace in one
response. It adds a new query only; event registration, listing, the hash
chain, exports, incident handling, responsibility attribution, and the
health check are unchanged and no write entry is added. The caller submits
only the path machine id and the path event id — no query parameters,
repeated parameters, request body, or business filter:

- any query string (including a repeated parameter), or a body carried on
  the GET, returns `422 {"error":{"code":"invalid_query"}}` before the
  machine or event is looked up and without reading any record;
- a missing machine, a missing event, or an event owned by another machine
  returns `404 {"error":{"code":"not_found"}}` with no closed-loop data;
- a failure while reading the event or any associated record returns
  `500 {"error":{"code":"internal_error"}}` with no event summary and none
  of the association arrays;
- the path accepts `GET` only; `HEAD` and every other method return `405`
  without reading records, computing a trace, or writing anything.

A successful response is exactly six groups in this fixed order. The first,
`event_summary`, is a single object (never an array) carrying the selected
event's result and reason (`allowed`, `reason`), its creation moment
(`created_at`), and its chain fields (`previous_event_id`, `content_hash`,
`chain_hash`). The other five groups are arrays: `evidence`, `incidents`,
`status_history`, `responsibility_assignments`, and `causal_links`.

The evidence, incidents, status-history, and responsibility-assignment
groups contain only records whose own `machine_id` is the path machine and
whose own `event_id` is the selected event; a damaged parent object never
filters a child (for example, a status transition or assignment is still
traced when its stored incident no longer exists). `causal_links` contains
the machine-owned associations whose stored `cause_event_id` OR
`effect_event_id` points at the selected event; the other endpoint is
emitted exactly as stored, even when it dangles or repeats. Every record is
output with its complete stored fields — evidence keeps its
`content_digest` and evidence chain fields, status transitions keep their
chain fields, and assignments keep theirs — with no repair, recomputation,
normalization, or dropping of a damaged, duplicated, or dangling value.

Each array is ordered by the actual UTC instant of its own `created_at` and
then by record id ascending, so an exact-second record precedes a
fractional-second record of the same second; a `created_at` whose original
text no longer parses is kept verbatim and sorts after every parseable
instant instead of crashing. Every array is present, an empty array when
the event has no record of that kind. The query is strictly read-only and
machine isolated (another machine's records never enter a group), and the
body is compact UTF-8 JSON in a fixed field order terminated by a single
newline, free of floating-point, `-0.0`, or non-finite values, and
byte-identical on repeat calls and across application restarts.

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

## Read-only stable incremental privacy responsibility query

`GET /machines/{machine_id}/authorization-decision-events/privacy-responsibility/compliance-export/changes`
is the stable, keyset-paginated incremental view of the desensitized privacy
responsibility export — the `changes` sub-entry under that export path. It
submits only the machine id, a page size, and an optional cursor; validation
completes before the machine or any responsibility record is read, so an
invalid query never touches machine or responsibility data:

- `limit` — required, a non-boolean integer from `1` to `100`. A missing,
  blank, fractional, boolean (`true`/`false`), non-decimal, or out-of-range
  value returns `422 {"error":{"code":"bad_limit"}}`.
- `cursor` — optional, an opaque exclusive position shaped
  `<created_at original text>|<record id>` and returned by a previous page.
  An empty, non-string, or shape-mismatching value, or a well-shaped cursor
  that names no stored assignment of the path machine, returns
  `422 {"error":{"code":"invalid_cursor"}}`. The timestamp segment is the
  original stored text (it is not required to parse), so an assignment with
  an unparseable `created_at` stays pageable.
- Any other query parameter, a repeated `limit`/`cursor`, or a request that
  carries a body returns `422 {"error":{"code":"invalid_query"}}`; every
  parameter check precedes the machine lookup, so an invalid query against a
  non-existent machine is still `422`.
- After validation, a missing machine returns
  `404 {"error":{"code":"not_found"}}` with no responsibility records.
- The path accepts `GET` only; other methods return `405` without reading
  records, computing a page, or writing anything.

The response is `{machine_id, limit, records, next_cursor, has_more}` in this
fixed key order; `records` is an empty array on an empty page or an empty
database. Each record carries exactly the ten desensitized
privacy-responsibility fields `{id, machine_id, event_id, incident_id,
created_at, previous_assignment_id, content_hash, chain_hash, party_ref,
role_ref}` — the eight stored fields exactly as stored plus the same
machine-scoped `party_ref`/`role_ref` digests the compliance export emits;
the raw party and role never appear. Records are ordered by the actual UTC
instant of `created_at` and then by record id ascending (an exact-second
record sorts before any fractional-second record of the same second). A
stored `created_at` that no longer parses is kept verbatim and
deterministically sorts after every parseable record.

The cursor is exclusive: a page returns only records strictly after the
cursor, so repeating the same cursor against unchanged data returns the
byte-identical page, and a newly inserted assignment with an earlier sort
position never makes an already-returned record resurface. `next_cursor` is
non-null only when at least one record follows the page (it is `null` on the
last page), and `has_more` is `true` exactly in that case. A failure while
reading the assignments returns `500 {"error":{"code":"internal_error"}}`
with no partial page. Cursors are stateless and add no schema: the query is
strictly read-only and machine-isolated, corrupted, missing, misowned, or
duplicated records are kept exactly as stored, the body is compact UTF-8 JSON
terminated by a single newline with no floating-point, `-0.0`, or non-finite
value, repeated calls against identical data and parameters are
byte-identical, and records persisted across application restarts remain
readable; old and empty databases work unchanged. Responsibility creation,
listing, chain verification, the existing desensitized export, diagnostics,
health checks, and all other endpoints are unchanged.

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

### Per-record chain diagnostics

`GET /machines/{machine_id}/privacy-accesses/diagnostics` is the read-only,
per-record diagnostic entry for one machine's privacy access chain. It adds a
new query only; registration, batch registration, the changes query, the
summary, buckets, the export, and the existing audits are unchanged. The caller
submits only the path machine id — no filter parameters, time bounds, business
conditions, or request body:

- any query parameter, or a body carried on the GET, returns
  `422 {"error":{"code":"invalid_query"}}` before the machine is looked up and
  without reading any access record;
- a missing machine returns `404 {"error":{"code":"not_found"}}` with no
  diagnostic data and no partial chain conclusion;
- a failure while reading the machine or its access records returns
  `500 {"error":{"code":"internal_error"}}` with no partial diagnosis;
- the path accepts `GET` only; other methods return `405` without reading
  records, computing positions, or summarizing data.

A successful response is exactly `{machine_id, valid, checked_count,
records}` in this field order. Records are ordered by the actual UTC instant
of `accessed_at` and then by record id (an exact-second record precedes any
fractional-second record of the same second), with `position` numbered from 1
with no gaps; a damaged `accessed_at` that no longer parses sorts after every
parseable instant instead of crashing. An empty machine returns `valid`
`true`, `checked_count` `0`, and an empty `records` array.

Every stored row is attributed to exactly one machine, in decreasing order of
evidence strength: the content digest (which covers the ownership field) is
authoritative when it verifies under some machine id; when the digest cannot
verify under any machine because the ownership field and another business
field are both damaged, the surviving chain links attribute the row — a row
whose stored predecessor link names an attributed row, or whose own id is
named by an attributed row's stored predecessor link, belongs to the same
machine — so the original machine still counts the record in
`checked_count` and reports it with `bad_ownership`, while the machine the
corrupted ownership column now names never gains it; and when neither
evidence survives, the stored ownership column is trusted, so a record whose
ownership column is intact but whose other business fields are damaged stays
in its stored machine's total. The two link directions are weighed
separately when a duplicated identifier or a conflicting link produces more
than one candidate owner: an owner corroborated from both directions (the
row's own predecessor and an incoming link) wins over a single foreign
pointer, and a candidate attested uniquely from one direction attributes the
row, so a conflicting chain link keeps the original machine's record
flagged `bad_ownership` and listed with the records after it rather than
moving it; a duplicated identifier is non-evidence in either direction
(it cannot identify one predecessor or one target), so a foreign row that
shares an identifier never enters the original machine's result.

Each record is exactly `{id, position, previous_access_id, content_hash,
chain_hash, errors}`: the actual stored record id, its position, its stored
predecessor id (`null` on the first record and the immediately preceding
record otherwise, emitted verbatim even when a link is damaged), the stored
content digest, the stored chain digest, and an array of anomaly codes. The seven fixed codes, in order, are
`missing_previous`, `bad_previous`, `bad_content_hash`, `bad_chain_hash`,
`bad_time`, `bad_id`, and `bad_ownership`; a sound record carries an empty
array and each record keeps every problem found on it. The first anomalous
record makes `valid` `false`; later records are still listed exactly as
stored. Content and chain digests follow the existing privacy access chain
rules. The query never creates, updates, deletes, repairs, recomputes, or
normalizes a record, damaged values never crash it or are rewritten, and
another machine's damaged records never affect this machine's result. The body
is compact UTF-8 JSON terminated by a single newline, contains no
floating-point or non-finite value, and is byte-identical on repeat calls and
across application restarts.


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
