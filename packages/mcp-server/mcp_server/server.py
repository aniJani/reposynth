"""FastMCP server: RepoSynth Deployment Grounding over stdio."""
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import tools

mcp = FastMCP("reposynth-grounding")


@mcp.tool()
def infra_state(target: str, section: Optional[str] = None) -> dict:
    """Live backend state for a named target (read-only).

    section: optional — one of schema|rls|auth|storage|functions|config.
    Response always carries the target's risk tier; treat risk:"prod" with care.
    """
    return tools.infra_state(target=target, section=section)


@mcp.tool()
def infra_verify(target: str, assertions: list) -> dict:
    """Assert expectations against the LIVE backend; per-assertion pass/fail/unsupported.

    Assertion types: table_exists, column_matches, rls_enabled, policy_exists,
    index_exists, auth_provider_enabled, bucket_exists, function_deployed,
    env_name_present, collection_exists. Use this to check your work against
    reality before declaring a task done.
    """
    return tools.infra_verify(target=target, assertions=assertions)


@mcp.tool()
def infra_impact(target: str, op: dict) -> dict:
    """Blast-radius of a proposed destructive op BEFORE doing it.

    op: {"op": drop_table|delete_bucket|drop_policy|drop_role|delete_function, ...args}.
    Returns findings from captured state + the target's risk tier; ops outside
    the set return result:"unknown".
    """
    return tools.infra_impact(target=target, op=op)


@mcp.tool()
def infra_snapshot(target: str, label: Optional[str] = None) -> dict:
    """Capture the target's current state to .reposynth/snapshots/ (redacted, content-addressed)."""
    return tools.infra_snapshot(target=target, label=label)


@mcp.tool()
def infra_drift(ref_a: str, ref_b: str) -> dict:
    """Structural diff between two states. Refs: a snapshot id, or "live:<target>"."""
    return tools.infra_drift(ref_a=ref_a, ref_b=ref_b)


@mcp.tool()
def deployment_check(target: str, repo_path: Optional[str] = None) -> dict:
    """Verify the repo's backend assumptions against the LIVE target, with provenance.

    Extracts what the code assumes exists — tables (ORM/SQL/Supabase client calls),
    storage buckets, edge functions, auth providers, edge-function env vars — and
    checks each against reality. Each result carries file:line sites. `skipped`
    lists assumptions with non-literal args that could not be resolved; a non-empty
    `skipped` means "not everything was checked." Use before declaring infra work done.
    """
    return tools.deployment_check(target=target, repo_path=repo_path)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
