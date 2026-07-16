import mcp_server._paths  # noqa: F401
from orchestrator.infra.expectations import extract_python

SRC = '''
import os
class Order(Base):
    __tablename__ = "orders"
class Invoice(Base):
    __tablename__ = "invoices"
X = os.environ["STRIPE_KEY"]
Y = os.getenv("DB_URL")
TABLE = compute_name()
class Dyn(Base):
    __tablename__ = TABLE
'''


def test_tablename_becomes_table_exists():
    out = extract_python(SRC, "models.py")
    tables = [f["assertion"]["table"] for f in out["findings"]]
    assert tables == ["orders", "invoices"]
    assert out["findings"][0]["assertion"]["type"] == "table_exists"
    assert out["findings"][0]["site"] == {"file": "models.py", "line": 4}


def test_os_environ_goes_to_app_env_not_findings():
    out = extract_python(SRC, "models.py")
    envs = sorted(e["env"] for e in out["app_env"])
    assert envs == ["DB_URL", "STRIPE_KEY"]
    # never emitted as a checkable assertion:
    assert all(f["assertion"]["type"] == "table_exists" for f in out["findings"])


def test_dynamic_tablename_is_skipped():
    out = extract_python(SRC, "models.py")
    assert any(s["reason"] == "dynamic argument" for s in out["skipped"])


def test_syntax_error_returns_empty_not_raise():
    out = extract_python("def (:", "bad.py")
    assert out == {"findings": [], "app_env": [], "skipped": []}
