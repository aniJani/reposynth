import pytest

from mcp_server.server import mcp

EXPECTED = {"infra_state", "infra_verify", "infra_impact", "infra_snapshot", "infra_drift"}


@pytest.mark.asyncio
async def test_all_five_tools_registered():
    registered = await mcp.list_tools()
    assert EXPECTED <= {t.name for t in registered}


@pytest.mark.asyncio
async def test_tools_have_descriptions():
    registered = await mcp.list_tools()
    for t in registered:
        if t.name in EXPECTED:
            assert t.description
