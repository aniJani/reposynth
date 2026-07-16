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
