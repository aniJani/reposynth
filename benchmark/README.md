# Legibility benchmark

Measures whether `deployment_check` / `infra_verify` make live backend state
legible enough to change an agent's answer — and, critically, whether they cut
the **confidently-wrong** rate (the false-completion proxy).

- `cases/<name>/` — `truth.json` (question + ground-truth answer), `state.json`
  (recorded StateDoc served by the fixture connector), `repo/` (seeded source).
- `runner.py` — `load_cases`, `score` (pure, unit-tested), `tool_answer`
  (deterministic), and `main()` (agent-alone vs agent-with-tools; needs
  `ANTHROPIC_API_KEY`).

`tool_answer` dispatches by question type: missing-resource questions ("does the
code reference a table absent from the deployment?") go through `deployment_check`
(code-vs-live extraction); state-property questions ("is RLS enabled?", "is the
bucket public?") carry a `verify` assertion in their `truth.json` and go through
`infra_verify`. The `verify` assertion is the affirmative of the question, so
pass→"yes", fail→"no".

Run offline: `../.venv-mcp/bin/python runner.py`

Metrics: correctness rate and confidently-wrong rate = `wrong && confidence=high`.

All cases run on the fixture connector — offline and deterministic. This
validates tool *logic*, not the Postgres SQL constants (those are unvalidated
against a real database — see the M2 spec's deferred-risk note).
