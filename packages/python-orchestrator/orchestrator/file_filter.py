"""
File Filtering Module

Filters out or minimizes files that are typically large and low-value for LLM prompts,
such as API route files, configuration files, and boilerplate code.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Set
from enum import Enum


class FileCategory(Enum):
    """Categories of files for filtering decisions."""
    CORE_LOGIC = "core_logic"           # Business logic, models, services
    API_ROUTES = "api_routes"           # API endpoint/route definitions
    CONFIG = "config"                    # Configuration files
    TEST = "test"                        # Test files
    BUILD = "build"                      # Build artifacts
    DOCUMENTATION = "documentation"      # Docs, README, etc.
    UNKNOWN = "unknown"                  # Uncategorized


class FileFilterStrategy(Enum):
    """How to handle filtered file categories."""
    INCLUDE_FULL = "include_full"        # Include complete file
    INCLUDE_MINIMAL = "include_minimal"  # Include only signatures/structure
    EXCLUDE = "exclude"                  # Skip entirely


# File pattern rules for categorization
FILE_PATTERNS = {
    FileCategory.API_ROUTES: [
        # Python API files
        r".*/api\.py$",
        r".*/routes\.py$",
        r".*/endpoints\.py$",
        r".*/views\.py$",
        r".*/api/.*\.py$",
        r".*/routes/.*\.py$",
        r".*/endpoints/.*\.py$",

        # JavaScript/TypeScript API files
        r".*/api\.ts$",
        r".*/api\.js$",
        r".*/routes\.ts$",
        r".*/routes\.js$",
        r".*/router\.ts$",
        r".*/router\.js$",
        r".*/api/.*\.(ts|js)$",
        r".*/routes/.*\.(ts|js)$",
        r".*/endpoints/.*\.(ts|js)$",
        r".*/controllers/.*\.(ts|js|py)$",
    ],

    FileCategory.CONFIG: [
        r".*/(config|settings|configuration)\.(py|ts|js|json|yaml|yml)$",
        r".*/\.env",
        r".*/webpack\.config\.(js|ts)$",
        r".*/vite\.config\.(js|ts)$",
        r".*/tsconfig\.json$",
        r".*/package\.json$",
        r".*/setup\.py$",
        r".*/pyproject\.toml$",
    ],

    FileCategory.TEST: [
        r".*/test_.*\.py$",
        r".*_test\.py$",
        r".*\.test\.(ts|js)$",
        r".*\.spec\.(ts|js)$",
        r".*/tests/.*",
        r".*/__tests__/.*",
    ],

    FileCategory.BUILD: [
        r"dist/.*",
        r"build/.*",
        r"node_modules/.*",
        r"__pycache__/.*",
        r"\.next/.*",
        r"\.cache/.*",
        r".*/dist/.*",
        r".*/build/.*",
        r".*/node_modules/.*",
        r".*/__pycache__/.*",
        r".*/\.next/.*",
        r".*/\.cache/.*",
    ],

    FileCategory.DOCUMENTATION: [
        r".*README\.md$",
        r".*\.md$",
        r".*/docs/.*",
    ],
}


def categorize_file(file_path: str) -> FileCategory:
    """
    Categorize a file based on its path and patterns.

    Args:
        file_path: Relative file path

    Returns:
        FileCategory enum
    """
    # Normalize path separators
    normalized_path = file_path.replace('\\', '/')

    # Check against each category's patterns
    for category, patterns in FILE_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, normalized_path, re.IGNORECASE):
                return category

    # Default to core logic if not matched
    return FileCategory.CORE_LOGIC


def should_include_file(
    file_path: str,
    strategy_map: Optional[Dict[FileCategory, FileFilterStrategy]] = None
) -> FileFilterStrategy:
    """
    Determine if and how a file should be included based on its category.

    Args:
        file_path: Relative file path
        strategy_map: Optional mapping of category to strategy. If None, uses defaults.

    Returns:
        FileFilterStrategy indicating how to handle the file
    """
    # Default strategies (conservative - include most things)
    default_strategies = {
        FileCategory.CORE_LOGIC: FileFilterStrategy.INCLUDE_FULL,
        FileCategory.API_ROUTES: FileFilterStrategy.INCLUDE_MINIMAL,  # Minimize API files
        FileCategory.CONFIG: FileFilterStrategy.EXCLUDE,              # Skip config
        FileCategory.TEST: FileFilterStrategy.EXCLUDE,                # Skip tests
        FileCategory.BUILD: FileFilterStrategy.EXCLUDE,               # Skip build artifacts
        FileCategory.DOCUMENTATION: FileFilterStrategy.EXCLUDE,       # Skip docs
        FileCategory.UNKNOWN: FileFilterStrategy.INCLUDE_FULL,
    }

    strategies = strategy_map or default_strategies
    category = categorize_file(file_path)
    return strategies.get(category, FileFilterStrategy.INCLUDE_FULL)


def extract_minimal_api_content(source_code: str, language: str) -> str:
    """
    Extract minimal representation of API file - just signatures and route definitions.

    For API files, we extract:
    - Route decorators (@app.route, @router.get, etc.)
    - Function signatures
    - Skip implementation details

    Args:
        source_code: Full source code
        language: Programming language (python, typescript, javascript)

    Returns:
        Minimal representation with just signatures
    """
    if language in ('python', 'py'):
        return _extract_minimal_python_api(source_code)
    elif language in ('typescript', 'ts', 'javascript', 'js'):
        return _extract_minimal_js_api(source_code)
    else:
        # For unsupported languages, just return first 20 lines
        lines = source_code.split('\n')
        return '\n'.join(lines[:20]) + '\n# ... (implementation omitted)'


def _extract_minimal_python_api(source_code: str) -> str:
    """Extract minimal Python API representation."""
    lines = source_code.split('\n')
    result = []
    in_function = False
    function_indent = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Keep imports
        if stripped.startswith(('import ', 'from ')):
            result.append(line)
            continue

        # Keep decorators (routes, etc.)
        if stripped.startswith('@'):
            result.append(line)
            in_function = True
            continue

        # Keep function/class definitions
        if stripped.startswith(('def ', 'class ', 'async def ')):
            result.append(line)
            in_function = True
            function_indent = indent
            # Add pass statement to make it valid Python
            result.append(' ' * (indent + 4) + 'pass')
            continue

        # Reset when we're back at module level
        if in_function and indent <= function_indent and stripped and not stripped.startswith(('def ', 'class ', '@')):
            in_function = False

    if not result:
        # Fallback: return first 10 lines
        result = lines[:10] + ['# ... (implementation omitted)']

    return '\n'.join(result)


def _extract_minimal_js_api(source_code: str) -> str:
    """Extract minimal JavaScript/TypeScript API representation."""
    lines = source_code.split('\n')
    result = []

    for line in lines:
        stripped = line.strip()

        # Keep imports
        if stripped.startswith(('import ', 'export ', 'from ')):
            result.append(line)
            continue

        # Keep route definitions (simplified)
        if any(keyword in stripped for keyword in ['router.', 'app.', '.get(', '.post(', '.put(', '.delete(', '.patch(']):
            # Extract just the route signature, not implementation
            if '{' in line:
                # Truncate at opening brace
                result.append(line.split('{')[0] + '{ /* ... */ }')
            else:
                result.append(line)
            continue

        # Keep function signatures
        if re.match(r'(async\s+)?function\s+\w+', stripped) or re.match(r'(export\s+)?(async\s+)?\w+\s*\(', stripped):
            if '{' in line:
                result.append(line.split('{')[0] + '{ /* ... */ }')
            else:
                result.append(line)
            continue

    if not result:
        # Fallback: return first 10 lines
        result = lines[:10] + ['// ... (implementation omitted)']

    return '\n'.join(result)


def filter_file_list(
    files: List[str],
    strategy_map: Optional[Dict[FileCategory, FileFilterStrategy]] = None,
    exclude_categories: Optional[List[FileCategory]] = None
) -> List[str]:
    """
    Filter a list of files based on category strategies.

    Args:
        files: List of file paths
        strategy_map: Optional mapping of category to strategy
        exclude_categories: Optional list of categories to completely exclude

    Returns:
        Filtered list of file paths
    """
    if exclude_categories is None:
        exclude_categories = [FileCategory.BUILD, FileCategory.TEST]

    filtered = []
    for file_path in files:
        category = categorize_file(file_path)

        # Skip excluded categories
        if category in exclude_categories:
            continue

        # Check strategy
        strategy = should_include_file(file_path, strategy_map)
        if strategy != FileFilterStrategy.EXCLUDE:
            filtered.append(file_path)

    return filtered


def get_aggressive_filter_strategy() -> Dict[FileCategory, FileFilterStrategy]:
    """
    Get an aggressive filtering strategy that maximizes token savings.

    Returns:
        Strategy map for aggressive filtering
    """
    return {
        FileCategory.CORE_LOGIC: FileFilterStrategy.INCLUDE_FULL,
        FileCategory.API_ROUTES: FileFilterStrategy.EXCLUDE,         # Skip API files entirely
        FileCategory.CONFIG: FileFilterStrategy.EXCLUDE,
        FileCategory.TEST: FileFilterStrategy.EXCLUDE,
        FileCategory.BUILD: FileFilterStrategy.EXCLUDE,
        FileCategory.DOCUMENTATION: FileFilterStrategy.EXCLUDE,
        FileCategory.UNKNOWN: FileFilterStrategy.INCLUDE_FULL,
    }


def get_minimal_filter_strategy() -> Dict[FileCategory, FileFilterStrategy]:
    """
    Get a minimal filtering strategy that includes API files with minimal content.

    Returns:
        Strategy map for minimal filtering
    """
    return {
        FileCategory.CORE_LOGIC: FileFilterStrategy.INCLUDE_FULL,
        FileCategory.API_ROUTES: FileFilterStrategy.INCLUDE_MINIMAL,  # Include minimal API
        FileCategory.CONFIG: FileFilterStrategy.EXCLUDE,
        FileCategory.TEST: FileFilterStrategy.EXCLUDE,
        FileCategory.BUILD: FileFilterStrategy.EXCLUDE,
        FileCategory.DOCUMENTATION: FileFilterStrategy.EXCLUDE,
        FileCategory.UNKNOWN: FileFilterStrategy.INCLUDE_FULL,
    }


def get_permissive_filter_strategy() -> Dict[FileCategory, FileFilterStrategy]:
    """
    Get a permissive filtering strategy that includes most files.

    Returns:
        Strategy map for permissive filtering
    """
    return {
        FileCategory.CORE_LOGIC: FileFilterStrategy.INCLUDE_FULL,
        FileCategory.API_ROUTES: FileFilterStrategy.INCLUDE_FULL,
        FileCategory.CONFIG: FileFilterStrategy.EXCLUDE,
        FileCategory.TEST: FileFilterStrategy.EXCLUDE,
        FileCategory.BUILD: FileFilterStrategy.EXCLUDE,
        FileCategory.DOCUMENTATION: FileFilterStrategy.EXCLUDE,
        FileCategory.UNKNOWN: FileFilterStrategy.INCLUDE_FULL,
    }


if __name__ == '__main__':
    # Test the file categorization
    test_files = [
        "src/models/user.py",
        "src/api/routes.py",
        "src/api.py",
        "tests/test_user.py",
        "config/settings.py",
        "dist/bundle.js",
        "src/services/auth.ts",
        "src/routes/users.ts",
        "README.md",
    ]

    print("=== File Categorization ===\n")
    for file_path in test_files:
        category = categorize_file(file_path)
        strategy = should_include_file(file_path)
        print(f"{file_path:40} -> {category.value:20} -> {strategy.value}")

    print("\n=== Filtered List (Default) ===")
    filtered = filter_file_list(test_files)
    for f in filtered:
        print(f"  ✓ {f}")

    print("\n=== Filtered List (Aggressive - No APIs) ===")
    aggressive_filtered = filter_file_list(test_files, get_aggressive_filter_strategy())
    for f in aggressive_filtered:
        print(f"  ✓ {f}")
