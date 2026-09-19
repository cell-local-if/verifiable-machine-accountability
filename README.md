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

## API overview

- `GET /health` — readiness check.
- `POST /machines` — register a machine.
- `GET /machines/{machine_id}` — fetch a machine.
- `POST /machines/{machine_id}/rotate-key` — rotate a machine's public key with optimistic version control.
- `POST /machines/{machine_id}/behavior-declarations` / `GET ...` — record and list a machine's enabled/disabled behavior declarations.
- `POST /policy-rules` — create a global policy rule. Accepts `action_type`, `resource_pattern`, `effect` (`allow`/`deny`), and a non-negative integer `priority`. The tuple `(action_type, resource_pattern, priority)` is unique; duplicates return `409 duplicate_policy_rule`.
- `POST /machines/{machine_id}/authorization-evaluations` — evaluate whether a machine may perform `action_type` on `resource`. Returns `{allowed, reason}`:
  - `no_enabled_declaration` — the machine has no enabled declaration whose action and resource pattern match.
  - `no_matching_policy` — declared, but no policy rule matches the action and resource.
  - `denied_by_policy` — the lowest-priority matching group contains a `deny`.
  - `allowed_by_policy` — otherwise allowed.

  `*` in a pattern matches any string; all other characters are matched literally. Evaluation is read-only.


