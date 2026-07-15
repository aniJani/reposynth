def test_package_imports():
    import mcp_server
    assert mcp_server.__version__ == "0.1.0"


def test_mcp_sdk_available():
    from mcp.server.fastmcp import FastMCP
    assert FastMCP is not None


def test_orchestrator_infra_importable():
    import mcp_server._paths  # noqa: F401
    import orchestrator.infra  # resolves via the bootstrap
    assert orchestrator.infra is not None
