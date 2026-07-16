"""Extract a repo's backend assumptions as M1 assertions, with provenance.

Pure: reads source text, emits assertions in verify.py's closed vocabulary.
Imports nothing from the frozen reposynth layer.
"""
import ast


def _site(relpath, lineno):
    return {"file": relpath, "line": lineno}


def extract_python(source: str, relpath: str) -> dict:
    findings, app_env, skipped = [], [], []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"findings": [], "app_env": [], "skipped": []}

    # __tablename__ = "<literal>"  -> table_exists  (only inside a class body)
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for stmt in cls.body:
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign):
                targets = [stmt.target]
            else:
                continue
            if not any(isinstance(t, ast.Name) and t.id == "__tablename__" for t in targets):
                continue
            val = stmt.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                findings.append({"assertion": {"type": "table_exists", "table": val.value},
                                 "site": _site(relpath, stmt.lineno)})
            elif val is not None:
                skipped.append({"reason": "dynamic argument", "site": _site(relpath, stmt.lineno)})

    # os.environ["X"] read / os.getenv("X") / os.environ.get("X")  -> app-host env
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
           and node.value.attr == "environ" and isinstance(node.slice, ast.Constant) \
           and isinstance(node.slice.value, str) and isinstance(node.ctx, ast.Load):
            app_env.append({"env": node.slice.value, "site": _site(relpath, node.lineno)})
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
           and node.func.attr in ("getenv", "get") and node.args \
           and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            recv = node.func.value
            is_env = (isinstance(recv, ast.Name) and recv.id == "os") or \
                     (isinstance(recv, ast.Attribute) and recv.attr == "environ")
            if is_env:
                app_env.append({"env": node.args[0].value, "site": _site(relpath, node.lineno)})

    return {"findings": findings, "app_env": app_env, "skipped": skipped}


import tree_sitter_typescript as _tsts
from tree_sitter import Language, Parser, Query, QueryCursor

_TS = Language(_tsts.language_typescript())
_PARSER = Parser(_TS)

_Q_CLIENTS = Query(_TS, """
(variable_declarator
  name: (identifier) @name
  value: (call_expression function: (identifier) @fn (#eq? @fn "createClient")))
""")

# member-call with a first arg (literal or dynamic): <obj>.<method>(<arg>)
# The arg is captured generically and classified in Python (_literal_str).
_Q_MEMBER_CALL = Query(_TS, """
(call_expression
  function: (member_expression object: (_) @obj property: (property_identifier) @method)
  arguments: (arguments . (_) @arg))
""")

_Q_OAUTH = Query(_TS, """
(call_expression
  function: (member_expression property: (property_identifier) @m (#eq? @m "signInWithOAuth"))
  arguments: (arguments (object (pair
    key: (property_identifier) @k (#eq? @k "provider")
    value: (_) @prov))))
""")

_Q_PGTABLE = Query(_TS, """
(call_expression
  function: (identifier) @fn (#eq? @fn "pgTable")
  arguments: (arguments . (_) @t))
""")

# process.env.X  /  import.meta.env.X  /  Deno.env.get("X")
_Q_PROCESS_ENV = Query(_TS, """
(member_expression
  object: (member_expression) @envobj
  property: (property_identifier) @name) @full
""")
_Q_DENO_ENV = Query(_TS, """
(call_expression
  function: (member_expression
    object: (member_expression object: (identifier) @o (#eq? @o "Deno")
             property: (property_identifier) @p1 (#eq? @p1 "env"))
    property: (property_identifier) @p2 (#eq? @p2 "get"))
  arguments: (arguments (string (string_fragment) @env)))
""")


def _text(n):
    return n.text.decode()


def _literal_str(node):
    """String value if node is a string literal, else None (dynamic argument)."""
    if node.type != "string":
        return None
    frag = next((c for c in node.named_children if c.type == "string_fragment"), None)
    return frag.text.decode() if frag else ""


def collect_clients(sources):
    clients = set()
    for _relpath, src in sources:
        tree = _PARSER.parse(src.encode() if isinstance(src, str) else src)
        for name, nodes in QueryCursor(_Q_CLIENTS).captures(tree.root_node).items():
            if name == "name":
                clients.update(_text(n) for n in nodes)
    return clients


def _obj_is_client(obj_node, clients):
    """True if obj is `<client>` identifier bound to createClient."""
    return obj_node.type == "identifier" and _text(obj_node) in clients


def _obj_is_client_member(obj_node, clients, prop):
    """True if obj is `<client>.<prop>` (e.g. supabase.storage / supabase.functions)."""
    return (obj_node.type == "member_expression"
            and obj_node.child_by_field_name("property") is not None
            and _text(obj_node.child_by_field_name("property")) == prop
            and _obj_is_client(obj_node.child_by_field_name("object"), clients))


def extract_ts(source: str, relpath: str, clients) -> dict:
    findings, app_env, skipped = [], [], []
    src = source.encode()
    tree = _PARSER.parse(src)
    root = tree.root_node
    under_functions = "supabase/functions/" in relpath.replace("\\", "/")
    drizzle = "drizzle-orm/pg-core" in source

    def site(node):
        return _site(relpath, node.start_point[0] + 1)

    # member calls on a bound client: literal arg -> finding, dynamic -> skipped
    for _m, cap in QueryCursor(_Q_MEMBER_CALL).matches(root):
        if "arg" not in cap:
            continue
        obj = cap["obj"][0]; method = _text(cap["method"][0]); node = cap["method"][0]
        if method in ("from", "table") and _obj_is_client(obj, clients):
            kind, key = "table_exists", "table"
        elif method == "from" and _obj_is_client_member(obj, clients, "storage"):
            kind, key = "bucket_exists", "bucket"
        elif method == "invoke" and _obj_is_client_member(obj, clients, "functions"):
            kind, key = "function_deployed", "function"
        else:
            continue
        val = _literal_str(cap["arg"][0])
        if val is not None:
            findings.append({"assertion": {"type": kind, key: val}, "site": site(node)})
        else:
            skipped.append({"reason": "dynamic argument", "site": site(node)})

    # signInWithOAuth: literal provider -> finding, dynamic -> skipped
    for _m, cap in QueryCursor(_Q_OAUTH).matches(root):
        if "prov" not in cap:
            continue
        prov = cap["prov"][0]
        val = _literal_str(prov)
        if val is not None:
            findings.append({"assertion": {"type": "auth_provider_enabled", "provider": val}, "site": site(prov)})
        else:
            skipped.append({"reason": "dynamic argument", "site": site(prov)})

    # drizzle pgTable: literal -> finding, dynamic -> skipped
    if drizzle:
        for _m, cap in QueryCursor(_Q_PGTABLE).matches(root):
            if "t" not in cap:
                continue
            t = cap["t"][0]
            val = _literal_str(t)
            if val is not None:
                findings.append({"assertion": {"type": "table_exists", "table": val}, "site": site(t)})
            else:
                skipped.append({"reason": "dynamic argument", "site": site(t)})

    # Deno.env.get(...) -> emit if under functions dir, else app_env
    for name, nodes in QueryCursor(_Q_DENO_ENV).captures(root).items():
        if name == "env":
            for n in nodes:
                if under_functions:
                    findings.append({"assertion": {"type": "env_name_present", "env": _text(n)},
                                     "site": site(n)})
                else:
                    app_env.append({"env": _text(n), "site": site(n)})

    # process.env.X / import.meta.env.X
    for _m, cap in QueryCursor(_Q_PROCESS_ENV).matches(root):
        envobj = _text(cap["envobj"][0])
        if envobj in ("process.env", "import.meta.env"):
            n = cap["name"][0]
            entry = {"env": _text(n), "site": site(n)}
            if under_functions:
                findings.append({"assertion": {"type": "env_name_present", "env": _text(n)}, "site": site(n)})
            else:
                app_env.append(entry)

    return {"findings": findings, "app_env": app_env, "skipped": skipped}


import json as _json
from pathlib import Path

_SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".reposynth_cache", "dist", "build"}
_TS_EXT = {".ts", ".tsx", ".js", ".jsx"}


def _iter_files(repo):
    for p in Path(repo).rglob("*"):
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS or part.startswith(".venv") for part in p.parts):
            continue
        if p.suffix in _TS_EXT or p.suffix == ".py":
            yield p


def extract(repo_dir: str) -> dict:
    repo = Path(repo_dir)
    files = list(_iter_files(repo))

    ts_sources = []
    for p in files:
        if p.suffix in _TS_EXT:
            try:
                ts_sources.append((str(p.relative_to(repo)), p.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
    clients = collect_clients(ts_sources)

    all_findings, all_skipped, all_app_env = [], [], []
    for p in files:
        rel = str(p.relative_to(repo)).replace("\\", "/")
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if p.suffix == ".py":
            r = extract_python(src, rel)
        else:
            r = extract_ts(src, rel, clients)
        all_findings += r["findings"]; all_skipped += r["skipped"]; all_app_env += r["app_env"]

    # dedupe identical assertions -> one entry, many sites
    by_key = {}
    for f in all_findings:
        key = _json.dumps(f["assertion"], sort_keys=True)
        by_key.setdefault(key, {"assertion": f["assertion"], "sites": []})
        by_key[key]["sites"].append(f["site"])
    assertions = []
    for entry in by_key.values():
        entry["sites"] = sorted(entry["sites"], key=lambda s: (s["file"], s["line"]))
        assertions.append(entry)
    assertions.sort(key=lambda a: _json.dumps(a["assertion"], sort_keys=True))

    notes = ["RLS intent is not derived from code; verify explicitly with rls_enabled / policy_exists assertions."]
    if all_app_env:
        notes.append(f"{len(all_app_env)} app-level env references were not checked — "
                     f"no connector exists for the application host.")

    return {"assertions": assertions, "skipped": all_skipped, "notes": notes}
