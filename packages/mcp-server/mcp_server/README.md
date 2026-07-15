# reposynth-grounding — Deployment Grounding MCP server

Lets a coding agent **verify its work against the live backend** instead of
guessing. Five read-only tools over Supabase / any Postgres:

| Tool | Question it answers |
|------|--------------------|
| `infra_state` | What is actually deployed right now? |
| `infra_verify` | Do these expectations hold in reality? (pass/fail per assertion) |
| `infra_impact` | What would this destructive op break? (+ prod risk flag) |
| `infra_snapshot` | Record current state (redacted, content-addressed) |
| `infra_drift` | What changed between two states / environments? |

Read-only by construction: the connector protocol has no write method, and
Postgres sessions open with `default_transaction_read_only=on`. Secret values
are hashed at capture and never stored.

## Setup

1. Install (any Python ≥3.10 env; reposynth env recommended):
   ```bash
   pip install -e '/Users/janit/reposynth/packages/mcp-server[dev]'
   ```
2. In your project, create `.reposynth/targets.json`:
   ```json
   { "targets": {
       "dev":  { "connector": "supabase", "projectRef": "<ref>",
                 "tokenEnv": "SUPABASE_ACCESS_TOKEN", "risk": "dev" },
       "prod": { "connector": "postgres", "urlEnv": "PROD_PG_URL_READONLY", "risk": "prod" } } }
   ```
   Values are env-var *names*; set those vars in your shell. For Postgres use a
   read-only role:
   ```sql
   CREATE ROLE reposynth_ro LOGIN PASSWORD '...';
   GRANT CONNECT ON DATABASE app TO reposynth_ro;
   GRANT USAGE ON SCHEMA public TO reposynth_ro;
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO reposynth_ro;
   ```
3. Register with Claude Code (run inside your project so `.reposynth/` resolves):
   ```bash
   claude mcp add reposynth-grounding -- reposynth-mcp
   ```

## Smoke test

Ask the agent: *"Call infra_verify on target dev asserting table_exists for a
table you know exists, and one you know doesn't."* Expect one pass, one fail —
never a silent success.
