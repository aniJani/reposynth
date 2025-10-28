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

    def get_top_level_statements(self, ast_nodes, source_code):
        """
        Capture top-level executable code like prototype assignments, IIFE, etc.
        Default implementation returns empty list; language-specific adapters can override.
        """
        return []

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

        # IMPROVED: First pass - extract __all__ list if it exists
        all_exports = self._extract_all_list(root, source_code)

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

                    # IMPROVED: Use __all__ if available, otherwise fall back to underscore rule
                    if all_exports is not None:
                        is_public = func_name in all_exports
                    else:
                        is_public = not func_name.startswith("_")

                    definitions.append(
                        {
                            "name": func_name,
                            "kind": node.kind,
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                            "is_public": is_public,
                        }
                    )
            for child in node.children:
                find_defs(child)

        find_defs(root)
        return definitions

    def _extract_all_list(self, root, source_code):
        """
        Extract the __all__ list from a Python module if it exists.
        Returns a set of exported names, or None if __all__ is not defined.
        """
        if not root:
            return None

        def find_all_assignment(node):
            # Look for assignment: __all__ = [...]
            if node.kind == "assignment":
                # Check if the left side is "__all__"
                left_node = node.children[0] if len(node.children) > 0 else None
                if left_node and left_node.kind == "identifier":
                    name = self._get_node_text(left_node, source_code)
                    if name == "__all__":
                        # Find the list node on the right side
                        list_node = next((c for c in node.children if c.kind == "list"), None)
                        if list_node:
                            # Extract all string literals from the list
                            exports = set()
                            for child in list_node.children:
                                if child.kind == "string":
                                    # Remove quotes from string
                                    string_val = self._get_node_text(child, source_code)
                                    # Handle both single and double quotes, and raw strings
                                    cleaned = string_val.strip().strip('"\'').strip('r"\'').strip('b"\'')
                                    exports.add(cleaned)
                            return exports
            # Recurse
            for child in node.children:
                result = find_all_assignment(child)
                if result is not None:
                    return result
            return None

        return find_all_assignment(root)

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

        # Track all exported names from export clauses (export { foo, bar })
        exported_names = set()

        def collect_exported_names(node):
            """First pass: collect all names from export { ... } clauses"""
            if node.kind == "export_clause":
                for child in node.children:
                    if child.kind == "export_specifier":
                        name_node = self._find_first_descendant_by_kind_bfs(child, "identifier")
                        if name_node:
                            exported_names.add(self._get_node_text(name_node, source_code))
            for child in node.children:
                collect_exported_names(child)

        collect_exported_names(root)

        def find_defs(node, is_exported=False):
            # Handle module.exports = function myFunction() { ... }
            # And module.exports = class MyClass { ... }
            if node.kind == "assignment_expression":
                left_node = node.children[0]
                right_node = node.children[2] # a = b, left is child 0, right is child 2

                left_text = self._get_node_text(left_node, source_code)
                if left_text == "module.exports" or left_text.startswith("exports."):
                    is_exported = True
                    # The actual definition is the right-hand side of the assignment
                    node = right_node

            # Handle export statement wrapper
            if node.kind == "export_statement":
                # Check if there's an export_clause (export { foo, bar })
                has_export_clause = any(c.kind == "export_clause" for c in node.children)

                if has_export_clause:
                    # Don't recurse into export_clause itself
                    for child in node.children:
                        if child.kind != "export_clause":
                            find_defs(child, is_exported=False)
                    return
                else:
                    # Direct export like: export function foo() or export default class Bar
                    for child in node.children:
                        find_defs(child, is_exported=True)
                    return

            if node.kind in ["function_declaration", "class_declaration", "method_definition", "function"]:
                # For `function_declaration`, the identifier is a direct child.
                # For anonymous `function`, there is no identifier.
                name_node = self._find_first_descendant_by_kind_bfs(node, "identifier")
                if name_node:
                    name_text = self._get_node_text(name_node, source_code)
                    # Check if this name is in the exported_names set (from export { ... })
                    final_is_exported = is_exported or (name_text in exported_names)

                    definitions.append({
                        "name": name_text,
                        "kind": node.kind,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                        "is_public": final_is_exported,
                    })

            # Handle lexical declarations (const, let) and variable declarations (var)
            if node.kind in ["lexical_declaration", "variable_declaration"]:
                for child in node.children:
                    if child.kind == "variable_declarator":
                        name_node = self._find_first_descendant_by_kind_bfs(child, "identifier")
                        if name_node:
                            name_text = self._get_node_text(name_node, source_code)
                            final_is_exported = is_exported or (name_text in exported_names)
                            definitions.append(
                                {
                                    "name": name_text,
                                    "kind": "const_declaration" if node.kind == "lexical_declaration" else "variable_declaration",
                                    "start_byte": child.start_byte,
                                    "end_byte": child.end_byte,
                                    "is_public": final_is_exported,
                                }
                            )

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

    def get_top_level_statements(self, ast_nodes, source_code):
        """
        Capture important top-level executable statements like:
        - Prototype assignments: create.prototype = StringSchema.prototype
        - Property attachments: myFunc.prop = value
        - IIFE: (function() { ... })()
        """
        statements, root = [], self._build_tree(ast_nodes)
        if not root:
            return []

        def find_important_statements(node, depth=0):
            # Only capture top-level statements (direct children of program/root)
            if depth > 1:
                return

            if node.kind == "expression_statement":
                # Check if this contains an assignment or call expression
                for child in node.children:
                    if child.kind == "assignment_expression":
                        # Check for prototype assignments or property attachments
                        left_node = child.children[0] if len(child.children) > 0 else None
                        if left_node:
                            left_text = self._get_node_text(left_node, source_code)
                            # Capture prototype assignments and property attachments
                            if ".prototype" in left_text or ("." in left_text and not left_text.startswith("this.")):
                                statement_text = self._get_node_text(node, source_code)
                                statements.append({
                                    "type": "prototype_assignment" if ".prototype" in left_text else "property_attachment",
                                    "code": statement_text,
                                    "start_byte": node.start_byte,
                                    "end_byte": node.end_byte,
                                })
                    elif child.kind == "call_expression":
                        # Check for IIFE: (function() { ... })() or (() => {})()
                        func_node = child.children[0] if len(child.children) > 0 else None
                        if func_node and func_node.kind in ["parenthesized_expression", "arrow_function", "function"]:
                            statement_text = self._get_node_text(node, source_code)
                            statements.append({
                                "type": "iife",
                                "code": statement_text,
                                "start_byte": node.start_byte,
                                "end_byte": node.end_byte,
                            })

            # Only recurse into direct children of root
            if depth == 0:
                for child in node.children:
                    find_important_statements(child, depth + 1)

        find_important_statements(root)
        return statements


class TypeScriptAdapter(LanguageAdapter):
    def get_definitions(self, ast_nodes, source_code):
        definitions, root = [], self._build_tree(ast_nodes)
        if not root:
            return []

        # Track all exported names from export clauses (export { foo, bar })
        exported_names = set()

        def collect_exported_names(node):
            """First pass: collect all names from export { ... } clauses"""
            if node.kind == "export_clause":
                for child in node.children:
                    if child.kind == "export_specifier":
                        # export_specifier has "name" field for the exported identifier
                        name_node = self._find_first_descendant_by_kind_bfs(child, "identifier")
                        if name_node:
                            exported_names.add(self._get_node_text(name_node, source_code))
            for child in node.children:
                collect_exported_names(child)

        collect_exported_names(root)

        def find_defs(node, is_exported=False):
            # Handle export statement wrapper
            if node.kind == "export_statement":
                # Check for export default
                has_default = any(self._get_node_text(c, source_code) == "default" for c in node.children if c.kind == "identifier")

                # Check if there's an export_clause (export { foo, bar })
                has_export_clause = any(c.kind == "export_clause" for c in node.children)

                if has_export_clause:
                    # Don't recurse into export_clause itself, but mark all definitions as potentially exported
                    # They'll be matched against exported_names set
                    for child in node.children:
                        if child.kind != "export_clause":
                            find_defs(child, is_exported=False)
                    return
                else:
                    # Direct export like: export function foo() or export default class Bar
                    for child in node.children:
                        find_defs(child, is_exported=True)
                    return

            # Handle all definition types
            if node.kind in [
                "function_declaration",
                "class_declaration",
                "interface_declaration",
                "type_alias_declaration",  # Added: type definitions
                "enum_declaration",  # Added: enum definitions
                "method_definition",
                "function_signature",  # Added: function overload signatures
            ]:
                # Find the name node (different for different types)
                name_node = None
                if node.kind in ["interface_declaration", "type_alias_declaration", "enum_declaration"]:
                    name_node = self._find_first_descendant_by_kind_bfs(node, "type_identifier")
                else:
                    name_node = self._find_first_descendant_by_kind_bfs(node, "identifier")

                if name_node:
                    name_text = self._get_node_text(name_node, source_code)
                    # Check if this name is in the exported_names set (from export { ... })
                    final_is_exported = is_exported or (name_text in exported_names)

                    definitions.append(
                        {
                            "name": name_text,
                            "kind": node.kind,
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                            "is_public": final_is_exported,
                        }
                    )

            # Handle lexical declarations (const, let) and variable declarations (var)
            if node.kind in ["lexical_declaration", "variable_declaration"]:
                # These can contain multiple declarators: const a = 1, b = 2;
                for child in node.children:
                    if child.kind == "variable_declarator":
                        name_node = self._find_first_descendant_by_kind_bfs(child, "identifier")
                        if name_node:
                            name_text = self._get_node_text(name_node, source_code)
                            final_is_exported = is_exported or (name_text in exported_names)
                            definitions.append(
                                {
                                    "name": name_text,
                                    "kind": "const_declaration" if node.kind == "lexical_declaration" else "variable_declaration",
                                    "start_byte": child.start_byte,
                                    "end_byte": child.end_byte,
                                    "is_public": final_is_exported,
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

        # Track exported names from export { ... } clauses
        exported_names = set()

        def collect_exported_names(node):
            if node.kind == "export_clause":
                for child in node.children:
                    if child.kind == "export_specifier":
                        name_node = self._find_first_descendant_by_kind_bfs(child, "identifier")
                        if name_node:
                            exported_names.add(self._get_node_text(name_node, source_code))
            for child in node.children:
                collect_exported_names(child)

        collect_exported_names(root)

        def find_vars(node, is_exported=False):
            # Handle export statement wrapper
            if node.kind == "export_statement":
                has_export_clause = any(c.kind == "export_clause" for c in node.children)
                if has_export_clause:
                    for child in node.children:
                        if child.kind != "export_clause":
                            find_vars(child, is_exported=False)
                    return
                else:
                    for child in node.children:
                        find_vars(child, is_exported=True)
                    return

            # Capture type declarations (type aliases, interfaces, enums)
            if node.kind in ["type_alias_declaration", "interface_declaration", "enum_declaration"]:
                name_node = self._find_first_descendant_by_kind_bfs(node, "type_identifier")
                if name_node:
                    name_text = self._get_node_text(name_node, source_code)
                    final_is_exported = is_exported or (name_text in exported_names)
                    variables.append(
                        {
                            "name": name_text,
                            "kind": node.kind,  # Added: track the kind for better analysis
                            "scope": "export" if final_is_exported else "internal",
                            "start_byte": node.start_byte,
                            "end_byte": node.end_byte,
                        }
                    )

            # Capture const/let/var declarations at top level
            if node.kind in ["lexical_declaration", "variable_declaration"]:
                for child in node.children:
                    if child.kind == "variable_declarator":
                        # Handle regular identifiers
                        name_node = self._find_first_descendant_by_kind_bfs(child, "identifier")
                        if name_node:
                            name_text = self._get_node_text(name_node, source_code)
                            final_is_exported = is_exported or (name_text in exported_names)
                            variables.append(
                                {
                                    "name": name_text,
                                    "kind": node.kind,
                                    "scope": "export" if final_is_exported else "global",
                                    "start_byte": child.start_byte,
                                    "end_byte": child.end_byte,
                                }
                            )
                        else:
                            # Handle destructuring: const { a, b } = obj;
                            pattern_node = next((c for c in child.children if c.kind in ["object_pattern", "array_pattern"]), None)
                            if pattern_node:
                                self._extract_destructured_names(pattern_node, source_code, variables, is_exported, exported_names, child)

            for child in node.children:
                find_vars(child, is_exported)

        find_vars(root)
        return variables

    def _extract_destructured_names(self, pattern_node, source_code, variables, is_exported, exported_names, parent_node):
        """Helper to extract variable names from destructuring patterns"""
        if pattern_node.kind == "object_pattern":
            for child in pattern_node.children:
                if child.kind == "shorthand_property_identifier_pattern":
                    name_text = self._get_node_text(child, source_code)
                    final_is_exported = is_exported or (name_text in exported_names)
                    variables.append(
                        {
                            "name": name_text,
                            "kind": "destructured_variable",
                            "scope": "export" if final_is_exported else "global",
                            "start_byte": parent_node.start_byte,
                            "end_byte": parent_node.end_byte,
                        }
                    )
                elif child.kind == "pair_pattern":
                    # { a: b } - we want 'b'
                    value_node = self._find_first_descendant_by_kind_bfs(child, "identifier")
                    if value_node:
                        name_text = self._get_node_text(value_node, source_code)
                        final_is_exported = is_exported or (name_text in exported_names)
                        variables.append(
                            {
                                "name": name_text,
                                "kind": "destructured_variable",
                                "scope": "export" if final_is_exported else "global",
                                "start_byte": parent_node.start_byte,
                                "end_byte": parent_node.end_byte,
                            }
                        )
        elif pattern_node.kind == "array_pattern":
            for child in pattern_node.children:
                if child.kind == "identifier":
                    name_text = self._get_node_text(child, source_code)
                    final_is_exported = is_exported or (name_text in exported_names)
                    variables.append(
                        {
                            "name": name_text,
                            "kind": "destructured_variable",
                            "scope": "export" if final_is_exported else "global",
                            "start_byte": parent_node.start_byte,
                            "end_byte": parent_node.end_byte,
                        }
                    )

    def get_top_level_statements(self, ast_nodes, source_code):
        """
        Capture important top-level executable statements like:
        - Prototype assignments: create.prototype = StringSchema.prototype
        - Property attachments: myFunc.prop = value
        - IIFE: (function() { ... })()
        - Namespace declarations
        """
        statements, root = [], self._build_tree(ast_nodes)
        if not root:
            return []

        def find_important_statements(node, depth=0):
            # Only capture top-level statements (direct children of program/root)
            if depth > 1:
                return

            if node.kind == "expression_statement":
                # Check if this contains an assignment or call expression
                for child in node.children:
                    if child.kind == "assignment_expression":
                        # Check for prototype assignments or property attachments
                        left_node = child.children[0] if len(child.children) > 0 else None
                        if left_node:
                            left_text = self._get_node_text(left_node, source_code)
                            # Capture prototype assignments and property attachments
                            if ".prototype" in left_text or ("." in left_text and not left_text.startswith("this.")):
                                statement_text = self._get_node_text(node, source_code)
                                statements.append({
                                    "type": "prototype_assignment" if ".prototype" in left_text else "property_attachment",
                                    "code": statement_text,
                                    "start_byte": node.start_byte,
                                    "end_byte": node.end_byte,
                                })
                    elif child.kind == "call_expression":
                        # Check for IIFE: (function() { ... })() or (() => {})()
                        func_node = child.children[0] if len(child.children) > 0 else None
                        if func_node and func_node.kind in ["parenthesized_expression", "arrow_function", "function"]:
                            statement_text = self._get_node_text(node, source_code)
                            statements.append({
                                "type": "iife",
                                "code": statement_text,
                                "start_byte": node.start_byte,
                                "end_byte": node.end_byte,
                            })

            # Only recurse into direct children of root
            if depth == 0:
                for child in node.children:
                    find_important_statements(child, depth + 1)

        find_important_statements(root)
        return statements


def get_adapter(file_path: Path) -> LanguageAdapter | None:
    if file_path.suffix == ".py":
        return PythonAdapter()
    elif file_path.suffix in [".ts", ".tsx"]:
        return TypeScriptAdapter()
    elif file_path.suffix in [".js", ".jsx"]:
        return JavaScriptAdapter()
    return None
