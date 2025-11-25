"""
Export tracking module - adds 'exported_from' field to name_registry.

This should be called after build_graphs_and_registry() in pipeline_runner.py.
"""

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any
import json


def find_exports_for_symbols(name_registry: Dict[str, Any], variable_registry: Dict[str, List[Any]], repo_path: Path) -> Dict[str, Any]:
    """
    Add 'exported_from' field to each symbol in name_registry.

    This does a second pass over the registry to find which files export each symbol.

    Args:
        name_registry: The existing name registry dict
        variable_registry: The variable registry (contains export info)
        repo_path: Path to repository root

    Returns:
        Updated name_registry with 'exported_from' fields added
    """
    # Build a reverse index: symbol_name -> [files that define it]
    symbols_by_name = defaultdict(list)
    for fqn, symbol_data in name_registry.items():
        symbol_name = fqn.split(':')[-1]  # Extract name from file:name
        symbols_by_name[symbol_name].append({
            'fqn': fqn,
            'file': symbol_data['file_path'],
            'data': symbol_data
        })

    # For each symbol, find which files export it
    for symbol_name, occurrences in symbols_by_name.items():
        for occurrence in occurrences:
            fqn = occurrence['fqn']
            defining_file = occurrence['file']
            exported_from = []

            # Check if it's directly exported from the defining file
            if defining_file in variable_registry:
                for var in variable_registry[defining_file]:
                    if var.get('name') == symbol_name and var.get('scope') == 'export':
                        exported_from.append(defining_file)
                        break

            # Check if it's re-exported from index files
            # Look for index.ts, __init__.py files that might re-export this symbol
            for file_path, variables in variable_registry.items():
                if file_path == defining_file:
                    continue  # Already checked

                # Check if this file exports the symbol
                for var in variables:
                    if var.get('name') == symbol_name and var.get('scope') == 'export':
                        # This file exports a symbol with the same name
                        # It's likely a re-export from an index file
                        if 'index' in file_path or '__init__' in file_path:
                            exported_from.append(file_path)

            # Update the name_registry entry
            if exported_from:
                name_registry[fqn]['exported_from'] = exported_from

    return name_registry


def find_exports_simple(name_registry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified export tracking that just marks symbols as exported if they're public.

    This is a fallback when variable_registry is not available.

    Args:
        name_registry: The existing name registry

    Returns:
        Updated name_registry with 'exported_from' fields
    """
    for fqn, symbol_data in name_registry.items():
        if symbol_data.get('is_public', False):
            # If it's public, assume it's exported from its defining file
            symbol_data['exported_from'] = [symbol_data['file_path']]

    return name_registry


def track_index_file_exports(name_registry: Dict[str, Any], import_graph: Dict[str, List[str]], repo_path: Path) -> Dict[str, Any]:
    """
    Track which symbols are re-exported from index files.

    This handles cases like:
        // src/array.ts
        export class ArraySchema {}

        // index.ts
        export { ArraySchema } from './src/array'

    Args:
        name_registry: The name registry
        import_graph: The import graph (file -> [imported files])
        repo_path: Repository root path

    Returns:
        Updated name_registry
    """
    # Find all index-like files
    index_files = []
    for file_path in import_graph.keys():
        file_name = Path(file_path).name
        if file_name in ['index.ts', 'index.js', '__init__.py', 'index.tsx', 'index.jsx']:
            index_files.append(file_path)

    # For each index file, check what it imports
    for index_file in index_files:
        imported_files = import_graph.get(index_file, [])

        # For each imported file, mark its public symbols as exported from this index
        for imported_file in imported_files:
            for fqn, symbol_data in name_registry.items():
                # Check if this symbol is from the imported file
                if symbol_data.get('file_path') == imported_file:
                    # Check if it's public (likely to be re-exported)
                    if symbol_data.get('is_public', False):
                        # Add this index file to exported_from
                        if 'exported_from' not in symbol_data:
                            symbol_data['exported_from'] = []
                        if index_file not in symbol_data['exported_from']:
                            symbol_data['exported_from'].append(index_file)

    return name_registry


# Integration example for pipeline_runner.py:
"""
Add this to Pipeline.build_graphs_and_registry(), after line 298:

    # Save graphs to output and cache
    with open(self.output_path / "name_registry.json", "w") as f:
        json.dump(self.name_registry, f, indent=2)

    # NEW: Add export tracking
    if self.variable_registry:
        from .export_tracker import find_exports_for_symbols, track_index_file_exports
        self.name_registry = find_exports_for_symbols(
            self.name_registry,
            dict(self.variable_registry),
            self.repo_path
        )
        self.name_registry = track_index_file_exports(
            self.name_registry,
            self.import_graph,
            self.repo_path
        )

        # Re-save with export info
        with open(self.output_path / "name_registry.json", "w") as f:
            json.dump(self.name_registry, f, indent=2)
"""
