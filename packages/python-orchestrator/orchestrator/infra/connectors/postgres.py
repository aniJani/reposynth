"""Generic Postgres connector: pg_catalog introspection, public schema.

ponytail: public schema only — extend WHERE clauses with a schema list if
multi-schema demand appears.
"""
from typing import Callable, List

from ..state_doc import make_state_doc
from ..targets import resolve_env
from . import base

TABLES_SQL = """
SELECT c.relname AS name, c.relrowsecurity AS rls_enabled
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r' AND n.nspname = 'public' ORDER BY 1
"""

COLUMNS_SQL = """
SELECT table_name, column_name, data_type, is_nullable
FROM information_schema.columns WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position
"""

INDEXES_SQL = """
SELECT tablename AS table, indexname AS name
FROM pg_indexes WHERE schemaname = 'public' ORDER BY 1, 2
"""

FOREIGN_KEYS_SQL = """
SELECT conname AS name, conrelid::regclass::text AS from_table,
       confrelid::regclass::text AS to_table
FROM pg_constraint WHERE contype = 'f' ORDER BY 1
"""

POLICIES_SQL = """
SELECT tablename AS table, policyname AS name, cmd, roles,
       qual AS using, with_check
FROM pg_policies WHERE schemaname = 'public' ORDER BY 1, 2
"""


def _parse_roles(roles) -> List[str]:
    if roles is None:
        return []
    if isinstance(roles, str):
        return [r for r in roles.strip("{}").split(",") if r]
    return list(roles)


def introspect(run_sql: Callable) -> dict:
    table_rows = run_sql(TABLES_SQL)
    column_rows = run_sql(COLUMNS_SQL)
    index_rows = run_sql(INDEXES_SQL)
    fk_rows = run_sql(FOREIGN_KEYS_SQL)
    policy_rows = run_sql(POLICIES_SQL)

    columns_by_table, indexes_by_table, fks_by_table, policies_by_table = {}, {}, {}, {}
    for r in column_rows:
        columns_by_table.setdefault(r["table_name"], []).append(
            {"name": r["column_name"], "type": r["data_type"],
             "nullable": r["is_nullable"] == "YES"})
    for r in index_rows:
        indexes_by_table.setdefault(r["table"], []).append(r["name"])
    for r in fk_rows:
        fks_by_table.setdefault(r["from_table"], []).append(
            {"name": r["name"], "toTable": r["to_table"]})
    for r in policy_rows:
        policies_by_table.setdefault(r["table"], []).append(
            {"name": r["name"], "cmd": r["cmd"], "roles": _parse_roles(r["roles"]),
             "using": r["using"], "withCheck": r["with_check"]})

    schema_tables, rls_tables = [], []
    for row in table_rows:
        name = row["name"]
        schema_tables.append({
            "name": name,
            "columns": columns_by_table.get(name, []),
            "indexes": indexes_by_table.get(name, []),
            "foreignKeys": fks_by_table.get(name, []),
        })
        rls_tables.append({
            "table": name,
            "enabled": bool(row["rls_enabled"]),
            "policies": policies_by_table.get(name, []),
        })
    return {"schema": {"tables": schema_tables}, "rls": {"tables": rls_tables}}


class PostgresConnector:
    id = "postgres"

    def detect(self, project_dir: str) -> dict:
        return {"detected": False}  # ponytail: no reliable fs signal for a plain PG url

    def fetch_state(self, target: dict) -> dict:
        import psycopg2
        import psycopg2.extras

        url = resolve_env(target, "urlEnv")
        conn = psycopg2.connect(url, options="-c default_transaction_read_only=on")
        try:
            def run_sql(sql):
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql)
                    return [dict(r) for r in cur.fetchall()]
            sections = introspect(run_sql)
        finally:
            conn.close()
        return make_state_doc("postgres", target["name"], sections)


base.register(PostgresConnector())
