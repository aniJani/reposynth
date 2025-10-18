# --- FILE: packages/python-orchestrator/orchestrator/language_adapter.py (FINAL CORRECT VERSION) ---
from abc import ABC, abstractmethod
from pathlib import Path
from collections import deque


class AstNode:
    def __init__(self, node_data):
        self.id = node_data["id"]
        self.kind = node_data["kind"]
        self.start_byte = node_data["start_byte"]
        self.end_byte = node_data["end_byte"]
        self.children_ids = node_data["children"]
        self.children = []
        self.parent = None
        self.text = None

    def __repr__(self):
        return f"<Node id={self.id} kind='{self.kind}'>"


class LanguageAdapter(ABC):
    @abstractmethod
    def get_definitions(self, ast_nodes, source_code):
        pass

    @abstractmethod
    def get_imports(self, ast_nodes, source_code):
        pass

    @abstractmethod
    def get_variables(self, ast_nodes, source_code, definitions):
        pass

    def _build_tree(self, node_list):
        if not node_list:
            return None
        node_map = {node_data["id"]: AstNode(node_data) for node_data in node_list}
        root_node = None
        for node_data in node_list:
            node = node_map[node_data["id"]]
            if node.id == 0:
                root_node = node
            for child_id in node.children_ids:
                child_node = node_map.get(child_id)
                if child_node:
                    child_node.parent = node
                    node.children.append(child_node)
        return root_node

    def _get_node_text(self, node, source_code):
        return source_code[node.start_byte : node.end_byte]

    def _find_first_descendant_by_kind_bfs(self, node: AstNode, kind: str):
        """Performs a Breadth-First Search to find the shallowest descendant of a given kind."""
        if not node:
            return None

        queue = deque(node.children)
        while queue:
            child = queue.popleft()
            if child.kind == kind:
                return child

            # Add this node's children to the end of the queue to search them next
            if child.kind not in [
                "object_type",
                "function_body",
                "class_body",
                "statement_block",
            ]:
                queue.extend(child.children)
        return None


class PythonAdapter(LanguageAdapter):
    def get_definitions(self, ast_nodes, source_code):
        definitions = []
        root = self._build_tree(ast_nodes)
        if not root:
            return []

        def find_defs(node):
            if node.kind in ["function_definition", "class_definition"]:
                # In Tree-sitter Python grammar, the name is always the child right after "def"/"class"
                # Structure: [def/class keyword, name (identifier), parameters/body, ...]
                # So we look for the first identifier that appears before any colon
                name_node = None
                for child in node.children:
                    if child.kind == "identifier":
                        name_node = child
                        break
                    # Stop if we hit the parameters or body (after the name)
                    if child.kind in ["parameters", "argument_list", ":", "block", "type"]:
                        break

                if name_node:
                    func_name = self._get_node_text(name_node, source_code)
                    definitions.append(
                        {
                            "name": func_name,
                            "kind": node.kind,
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                            "is_public": not func_name.startswith("_"),
                        }
                    )
            for child in node.children:
                find_defs(child)

        find_defs(root)
        return definitions

    def get_imports(self, ast_nodes, source_code):
        imports, root = [], self._build_tree(ast_nodes)
        if not root:
            return []

        def find_imports(node):
            if node.kind == "import_statement":
                name_nodes = [c for c in node.children if c.kind == "dotted_name"]
                for name_node in name_nodes:
                    imports.append(self._get_node_text(name_node, source_code))
            if node.kind == "import_from_statement":
                # Check for both absolute imports (dotted_name) and relative imports (relative_import)
                module_name_node = next(
                    (c for c in node.children if c.kind in ["dotted_name", "relative_import"]), None
                )
                if module_name_node:
                    imports.append(self._get_node_text(module_name_node, source_code))
            for child in node.children:
                find_imports(child)

        find_imports(root)
        return imports

    def get_variables(self, ast_nodes, source_code, definitions):
        variables, root = [], self._build_tree(ast_nodes)
        if not root:
            return []
        func_scopes = {
            d["start_byte"]: d["name"]
            for d in definitions
            if d["kind"] == "function_definition"
        }

        def find_scope(node):
            curr = node.parent
            while curr:
                if (
                    curr.kind == "function_definition"
                    and curr.start_byte in func_scopes
                ):
                    return func_scopes[curr.start_byte]
                curr = curr.parent
            return "global"

        def find_vars(node):
            if node.kind == "assignment":
                identifier_node = next(
                    (c for c in node.children if c.kind == "identifier"), None
                )
                if identifier_node:
                    variables.append(
                        {
                            "name": self._get_node_text(identifier_node, source_code),
                            "scope": find_scope(node),
                            "start_byte": identifier_node.start_byte,
                            "end_byte": identifier_node.end_byte,
                        }
                    )
            for child in node.children:
                find_vars(child)

        find_vars(root)
        return variables


class JavaScriptAdapter(LanguageAdapter):
    def get_definitions(self, ast_nodes, source_code):
        definitions, root = [], self._build_tree(ast_nodes)
        if not root:
            return []

        def find_defs(node, is_exported=False):
            # Handle module.exports = function myFunction() { ... }
            # And module.exports = class MyClass { ... }
            if node.kind == "assignment_expression":
                left_node = node.children[0]
                right_node = node.children[2] # a = b, left is child 0, right is child 2
                
                left_text = self._get_node_text(left_node, source_code)
                if left_text == "module.exports":
                    is_exported = True
                    # The actual definition is the right-hand side of the assignment
                    node = right_node

            if node.kind == "export_statement":
                for child in node.children:
                    find_defs(child, is_exported=True)
                return
            
            if node.kind in ["function_declaration", "class_declaration", "method_definition", "function"]:
                # For `function_declaration`, the identifier is a direct child.
                # For anonymous `function`, there is no identifier.
                name_node = self._find_first_descendant_by_kind_bfs(node, "identifier")
                if name_node:
                    definitions.append({
                        "name": self._get_node_text(name_node, source_code),
                        "kind": node.kind,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                        "is_public": is_exported,
                    })

            # Recurse carefully
            for child in node.children:
                find_defs(child, is_exported)

        find_defs(root)
        return definitions

    def get_imports(self, ast_nodes, source_code):
        imports, root = [], self._build_tree(ast_nodes)
        if not root:
            return []

        def find_imports(node):
            # Find `require('module')`
            if node.kind == "call_expression":
                # Check if the function being called is an identifier named 'require'
                func_node = node.children[0]
                if func_node.kind == 'identifier' and self._get_node_text(func_node, source_code) == "require":
                    # The argument is usually the first child of 'arguments'
                    args_node = next((c for c in node.children if c.kind == 'arguments'), None)
                    if args_node:
                        string_node = self._find_first_descendant_by_kind_bfs(args_node, "string")
                        if string_node:
                            # Remove quotes from the module path
                            imports.append(self._get_node_text(string_node, source_code)[1:-1])
            
            # Find `import ... from 'module'`
            if node.kind == "import_statement":
                source_node = self._find_first_descendant_by_kind_bfs(node, "string")
                if source_node:
                    imports.append(self._get_node_text(source_node, source_code)[1:-1])
            
            for child in node.children:
                find_imports(child)

        find_imports(root)
        return imports

    def get_variables(self, ast_nodes, source_code, definitions):
        variables, root = [], self._build_tree(ast_nodes)
        if not root:
            return []

        # --- REFINED LOGIC ---
        def find_top_level_vars(node, is_exported=False):
            # Handle `module.exports = ...` or `exports.foo = ...`
            if node.kind == "assignment_expression":
                left_node = node.children[0]
                left_text = self._get_node_text(left_node, source_code)
                if left_text.startswith("module.exports") or left_text.startswith("exports."):
                     is_exported = True

            # Standard ES6 export
            if node.kind == "export_statement":
                for child in node.children:
                    find_top_level_vars(child, is_exported=True)
                return # Stop recursion here for this branch

            # Find `const app = express()` or `var app = express()`
            if node.kind in ["lexical_declaration", "variable_declaration"]:
                declarator = self._find_first_descendant_by_kind_bfs(node, "variable_declarator")
                if declarator:
                    id_node = self._find_first_descendant_by_kind_bfs(declarator, "identifier")
                    if id_node:
                        variables.append({
                            "name": self._get_node_text(id_node, source_code),
                            # Use `is_exported` flag determined by parent traversal
                            "scope": "export" if is_exported else "global",
                            "start_byte": node.start_byte, # Use the span of the whole declaration
                            "end_byte": node.end_byte,
                        })

        # --- CRITICAL CHANGE ---
        # Only iterate through the direct children of the root 'program' node.
        # This prevents us from capturing variables inside functions.
        if root and root.children:
            for child in root.children:
                find_top_level_vars(child)
            
        return variables


class TypeScriptAdapter(LanguageAdapter):
    def get_definitions(self, ast_nodes, source_code):
        definitions, root = [], self._build_tree(ast_nodes)
        if not root:
            return []

        def find_defs(node, is_exported=False):
            if node.kind == "export_statement":
                for child in node.children:
                    find_defs(child, is_exported=True)
                return
            if node.kind in [
                "function_declaration",
                "class_declaration",
                "interface_declaration",
                "method_definition",
            ]:
                name_node = self._find_first_descendant_by_kind_bfs(node, "identifier")
                if not name_node and node.kind == "interface_declaration":
                    name_node = self._find_first_descendant_by_kind_bfs(
                        node, "type_identifier"
                    )
                if name_node:
                    definitions.append(
                        {
                            "name": self._get_node_text(name_node, source_code),
                            "kind": node.kind,
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                            "is_public": is_exported,
                        }
                    )
            for child in node.children:
                find_defs(child, is_exported)

        find_defs(root)
        return definitions

    def get_imports(self, ast_nodes, source_code):
        imports, root = [], self._build_tree(ast_nodes)
        if not root:
            return []

        def find_imports(node):
            if node.kind == "import_statement":
                source_node = self._find_first_descendant_by_kind_bfs(node, "string")
                if source_node:
                    imports.append(self._get_node_text(source_node, source_code)[1:-1])
            for child in node.children:
                find_imports(child)

        find_imports(root)
        return imports

    def get_variables(self, ast_nodes, source_code, definitions):
        variables, root = [], self._build_tree(ast_nodes)
        if not root:
            return []

        def find_vars(node, is_exported=False):
            if node.kind == "export_statement":
                for child in node.children:
                    find_vars(child, is_exported=True)
                return
            if node.kind in ["type_alias_declaration", "interface_declaration"]:
                name_node = self._find_first_descendant_by_kind_bfs(
                    node, "type_identifier"
                )
                if name_node:
                    variables.append(
                        {
                            "name": self._get_node_text(name_node, source_code),
                            "scope": "export" if is_exported else "internal",
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                        }
                    )
            for child in node.children:
                find_vars(child, is_exported)

        find_vars(root)
        return variables


def get_adapter(file_path: Path) -> LanguageAdapter | None:
    if file_path.suffix == ".py":
        return PythonAdapter()
    elif file_path.suffix in [".ts", ".tsx"]:
        return TypeScriptAdapter()
    elif file_path.suffix in [".js", ".jsx"]:
        return JavaScriptAdapter()
    return None
