# Legibility benchmark

Measures whether `deployment_check` / `infra_verify` make live backend state
legible enough to change an agent's answer — and, critically, whether they cut
the **confidently-wrong** rate (the false-completion proxy).

- `cases/<name>/` — `truth.json` (question + ground-truth answer), `state.json`
  (recorded StateDoc served by the fixture connector), `repo/` (seeded source).
- `runner.py` — `load_cases`, `score` (pure, unit-tested), `tool_answer`
  (deterministic), and `main()` (agent-alone vs agent-with-tools; needs
  `ANTHROPIC_API_KEY`).

Run offline: `../.venv-mcp/bin/python runner.py`

Metrics: correctness rate and confidently-wrong rate = `wrong && confidence=high`.

All cases run on the fixture connector — offline and deterministic. This
validates tool *logic*, not the Postgres SQL constants (those are unvalidated
against a real database — see the M2 spec's deferred-risk note).
