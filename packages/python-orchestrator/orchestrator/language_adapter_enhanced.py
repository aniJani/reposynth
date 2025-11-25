# Enhanced language adapter methods to add inheritance tracking
# These methods should be integrated into the existing language_adapter.py

def enhanced_typescript_get_definitions_with_inheritance(self, ast_nodes, source_code):
    """
    Enhanced version that adds 'inherits_from' field to class definitions.
    Patch this into TypeScriptAdapter.get_definitions
    """
    definitions, root = [], self._build_tree(ast_nodes)
    if not root:
        return []

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

    def find_defs(node, is_exported=False, depth=0):
        captured_definition = False

        if node.kind == "export_statement":
            has_export_clause = any(c.kind == "export_clause" for c in node.children)
            if has_export_clause:
                for child in node.children:
                    if child.kind != "export_clause":
                        find_defs(child, is_exported=False, depth=depth)
                return
            else:
                for child in node.children:
                    find_defs(child, is_exported=True, depth=depth)
                return

        if node.kind in [
            "function_declaration",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "method_definition",
            "function_signature",
        ]:
            name_node = None
            if node.kind in ["interface_declaration", "type_alias_declaration", "enum_declaration"]:
                name_node = self._find_first_descendant_by_kind_bfs(node, "type_identifier")
            else:
                name_node = self._find_first_descendant_by_kind_bfs(node, "identifier")

            if name_node:
                name_text = self._get_node_text(name_node, source_code)
                final_is_exported = is_exported or (name_text in exported_names)

                definition = {
                    "name": name_text,
                    "kind": node.kind,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                    "is_public": final_is_exported,
                }

                # NEW: Extract inheritance for classes and interfaces
                if node.kind in ["class_declaration", "interface_declaration"]:
                    inherits_from = self._extract_inheritance(node, source_code)
                    if inherits_from:
                        definition["inherits_from"] = inherits_from

                definitions.append(definition)
                captured_definition = True

        if node.kind in ["lexical_declaration", "variable_declaration"] and depth <= 1:
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

        next_is_exported = False if captured_definition else is_exported
        for child in node.children:
            find_defs(child, next_is_exported, depth + 1)

    find_defs(root)
    return definitions


def extract_inheritance_typescript(self, node, source_code):
    """
    Extract base classes/interfaces from class or interface declarations.
    Returns a list of base class/interface names.
    """
    inherits_from = []

    # Look for "extends" clause in TypeScript/JavaScript
    for child in node.children:
        if child.kind == "class_heritage":
            # class_heritage contains "extends" or "implements" clauses
            for heritage_child in child.children:
                if heritage_child.kind == "extends_clause":
                    # Find the type/identifier being extended
                    type_node = self._find_first_descendant_by_kind_bfs(heritage_child, "identifier")
                    if not type_node:
                        type_node = self._find_first_descendant_by_kind_bfs(heritage_child, "type_identifier")
                    if type_node:
                        inherits_from.append(self._get_node_text(type_node, source_code))

    return inherits_from if inherits_from else None


def enhanced_python_get_definitions_with_inheritance(self, ast_nodes, source_code):
    """
    Enhanced version that adds 'inherits_from' field to class definitions.
    Patch this into PythonAdapter.get_definitions
    """
    definitions = []
    root = self._build_tree(ast_nodes)
    if not root:
        return []

    all_exports = self._extract_all_list(root, source_code)

    def find_defs(node):
        if node.kind in ["function_definition", "class_definition"]:
            name_node = None
            for child in node.children:
                if child.kind == "identifier":
                    name_node = child
                    break
                if child.kind in ["parameters", "argument_list", ":", "block", "type"]:
                    break

            if name_node:
                func_name = self._get_node_text(name_node, source_code)

                if all_exports is not None:
                    is_public = func_name in all_exports
                else:
                    is_public = not func_name.startswith("_")

                definition = {
                    "name": func_name,
                    "kind": node.kind,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                    "is_public": is_public,
                }

                # NEW: Extract inheritance for classes
                if node.kind == "class_definition":
                    inherits_from = self._extract_inheritance_python(node, source_code)
                    if inherits_from:
                        definition["inherits_from"] = inherits_from

                definitions.append(definition)

        for child in node.children:
            find_defs(child)

    find_defs(root)
    return definitions


def extract_inheritance_python(self, node, source_code):
    """
    Extract base classes from Python class definitions.
    Returns a list of base class names.
    """
    inherits_from = []

    # In Python AST, class bases are in argument_list node
    # class Foo(Bar, Baz): ...
    for child in node.children:
        if child.kind == "argument_list":
            # Extract all identifiers from the argument list (these are base classes)
            for arg_child in child.children:
                if arg_child.kind == "identifier":
                    inherits_from.append(self._get_node_text(arg_child, source_code))
                # Handle dotted names like "module.BaseClass"
                elif arg_child.kind == "attribute":
                    inherits_from.append(self._get_node_text(arg_child, source_code))

    return inherits_from if inherits_from else None
