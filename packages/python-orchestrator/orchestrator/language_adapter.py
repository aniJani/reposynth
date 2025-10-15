# --- FILE: packages/python-orchestrator/orchestrator/language_adapter.py ---

from abc import ABC, abstractmethod
from pathlib import Path

# A simple in-memory representation of our AST nodes
class AstNode:
    def __init__(self, node_data):
        self.id = node_data['id']
        self.kind = node_data['kind']
        self.start_byte = node_data['start_byte']
        self.end_byte = node_data['end_byte']
        self.children_ids = node_data['children']
        self.children = []
        self.parent = None
        self.text = None

    def __repr__(self):
        return f"<Node id={self.id} kind='{self.kind}'>"

class LanguageAdapter(ABC):
    @abstractmethod
    def get_definitions(self, ast_nodes, source_code):
        """Extract function and class definitions."""
        pass

    @abstractmethod
    def get_imports(self, ast_nodes, source_code):
        """Extract import statements."""
        pass

    def _build_tree(self, node_list):
        """Helper to reconstruct the tree from a flat list of nodes."""
        node_map = {node_data['id']: AstNode(node_data) for node_data in node_list}
        for node_data in node_list:
            node = node_map[node_data['id']]
            for child_id in node.children_ids:
                child_node = node_map.get(child_id)
                if child_node:
                    child_node.parent = node
                    node.children.append(child_node)
        return node_map.get(0) # Return the root node (ID 0)

    def _get_node_text(self, node, source_code):
        """Extracts the text of a node from the source code."""
        return source_code[node.start_byte:node.end_byte]

class PythonAdapter(LanguageAdapter):
    def get_definitions(self, ast_nodes, source_code):
        definitions = []
        root = self._build_tree(ast_nodes)
        if not root:
            return []

        def find_defs(node):
            if node.kind in ["function_definition", "class_definition"]:
                name_node = next((c for c in node.children if c.kind == 'identifier'), None)
                if name_node:
                    definitions.append({
                        "name": self._get_node_text(name_node, source_code),
                        "kind": node.kind,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                        "is_public": not self._get_node_text(name_node, source_code).startswith("_"),
                    })
            for child in node.children:
                find_defs(child)
        
        find_defs(root)
        return definitions

    def get_imports(self, ast_nodes, source_code):
        imports = []
        root = self._build_tree(ast_nodes)
        if not root:
            return []
            
        def find_imports(node):
            if node.kind == "import_statement":
                # e.g., `import os, sys`
                name_nodes = [c for c in node.children if c.kind == 'dotted_name']
                for name_node in name_nodes:
                    imports.append(self._get_node_text(name_node, source_code))

            if node.kind == "import_from_statement":
                # e.g., `from pathlib import Path`
                module_name_node = next((c for c in node.children if c.kind == 'dotted_name'), None)
                if module_name_node:
                    imports.append(self._get_node_text(module_name_node, source_code))

            for child in node.children:
                find_imports(child)
        
        find_imports(root)
        return imports

class TypeScriptAdapter(LanguageAdapter):
    def get_definitions(self, ast_nodes, source_code):
        definitions = []
        root = self._build_tree(ast_nodes)
        if not root:
            return []

        def find_defs(node, is_exported=False):
            # Check for export statement wrapper
            if node.kind == "export_statement":
                for child in node.children:
                    find_defs(child, is_exported=True)
                return

            if node.kind in ["function_declaration", "class_declaration", "interface_declaration"]:
                name_node = next((c for c in node.children if c.kind == 'identifier'), None)
                if name_node:
                    definitions.append({
                        "name": self._get_node_text(name_node, source_code),
                        "kind": node.kind,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                        "is_public": is_exported,
                    })
            
            for child in node.children:
                find_defs(child, is_exported)
        
        find_defs(root)
        return definitions

    def get_imports(self, ast_nodes, source_code):
        imports = []
        root = self._build_tree(ast_nodes)
        if not root:
            return []

        def find_imports(node):
            if node.kind == "import_statement":
                source_node = next((c for c in node.children if c.kind == 'string'), None)
                if source_node:
                    # Remove quotes from the string
                    imports.append(self._get_node_text(source_node, source_code)[1:-1])
            
            for child in node.children:
                find_imports(child)
        
        find_imports(root)
        return imports

def get_adapter(file_path: Path) -> LanguageAdapter | None:
    if file_path.suffix == '.py':
        return PythonAdapter()
    elif file_path.suffix in ['.ts', '.tsx', '.js', '.jsx']:
        return TypeScriptAdapter()
    return None