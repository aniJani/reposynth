import mcp_server._paths  # noqa: F401
from orchestrator.infra.connectors import postgres
from orchestrator.infra.connectors.base import get_connector

ROWS = {
    postgres.TABLES_SQL: [
        {"name": "users", "rls_enabled": True},
        {"name": "orders", "rls_enabled": False},
    ],
    postgres.COLUMNS_SQL: [
        {"table_name": "users", "column_name": "id", "data_type": "uuid", "is_nullable": "NO"},
        {"table_name": "orders", "column_name": "id", "data_type": "uuid", "is_nullable": "NO"},
        {"table_name": "orders", "column_name": "user_id", "data_type": "uuid", "is_nullable": "YES"},
    ],
    postgres.INDEXES_SQL: [{"table": "users", "name": "users_pkey"}],
    postgres.FOREIGN_KEYS_SQL: [
        {"name": "orders_user_fk", "from_table": "orders", "to_table": "users"},
    ],
    postgres.POLICIES_SQL: [
        {"table": "users", "name": "sel", "cmd": "SELECT",
         "roles": "{authenticated}", "using": "true", "with_check": None},
    ],
}


def fake_run_sql(sql):
    return ROWS[sql]


def test_introspect_schema_shape():
    sections = postgres.introspect(fake_run_sql)
    tables = {t["name"]: t for t in sections["schema"]["tables"]}
    assert tables["orders"]["columns"][1] == {"name": "user_id", "type": "uuid", "nullable": True}
    assert tables["users"]["indexes"] == ["users_pkey"]
    assert tables["orders"]["foreignKeys"] == [{"name": "orders_user_fk", "toTable": "users"}]


def test_introspect_rls_shape():
    sections = postgres.introspect(fake_run_sql)
    rls = {t["table"]: t for t in sections["rls"]["tables"]}
    assert rls["users"]["enabled"] is True
    assert rls["users"]["policies"] == [
        {"name": "sel", "cmd": "SELECT", "roles": ["authenticated"],
         "using": "true", "withCheck": None}
    ]
    assert rls["orders"]["enabled"] is False
    assert rls["orders"]["policies"] == []


def test_roles_array_normalization():
    assert postgres._parse_roles("{authenticated,anon}") == ["authenticated", "anon"]
    assert postgres._parse_roles(["service_role"]) == ["service_role"]
    assert postgres._parse_roles(None) == []


def test_connector_registered_and_readonly():
    connector = get_connector("postgres")
    assert connector.id == "postgres"
    assert not hasattr(connector, "apply") and not hasattr(connector, "execute")
