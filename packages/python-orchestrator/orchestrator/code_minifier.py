"""
Code Minification Module

Strips comments, docstrings, and excessive whitespace from source code
to reduce token count when generating prompts.
"""

import re
from typing import Optional


def minify_python(source_code: str, keep_docstrings: bool = False) -> str:
    """
    Minify Python code by removing comments and excessive whitespace.

    Args:
        source_code: Python source code
        keep_docstrings: If True, keep docstrings (useful for APIs)

    Returns:
        Minified Python code
    """
    lines = source_code.split('\n')
    result = []
    in_multiline_string = None

    for line in lines:
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            continue

        # Skip comment-only lines
        if stripped.startswith('#'):
            continue

        # Remove inline comments (but preserve strings with #)
        # Simple approach: remove # and everything after if not in string
        if '#' in line:
            # Check if # is in a string
            in_string = False
            quote_char = None
            new_line = []
            i = 0
            while i < len(line):
                char = line[i]
                if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None

                if char == '#' and not in_string:
                    # Found comment - stop here
                    break

                new_line.append(char)
                i += 1

            line = ''.join(new_line).rstrip()

        # Handle docstrings
        if not keep_docstrings:
            # Skip triple-quoted strings (docstrings)
            if '"""' in line or "'''" in line:
                if in_multiline_string is None:
                    # Starting a multiline string
                    quote = '"""' if '"""' in line else "'''"
                    # Check if it ends on same line
                    if line.count(quote) >= 2:
                        # Single line docstring - skip it
                        continue
                    else:
                        in_multiline_string = quote
                        continue
                elif in_multiline_string in line:
                    # Ending the multiline string
                    in_multiline_string = None
                    continue

            # Skip lines inside multiline string
            if in_multiline_string:
                continue

        # Keep non-empty, non-comment lines
        if line.strip():
            result.append(line.rstrip())

    return '\n'.join(result)


def minify_javascript(source_code: str, keep_jsdoc: bool = False) -> str:
    """
    Minify JavaScript/TypeScript code by removing comments and excessive whitespace.

    Args:
        source_code: JS/TS source code
        keep_jsdoc: If True, keep JSDoc comments (/** */)

    Returns:
        Minified code
    """
    # Remove single-line comments
    source_code = re.sub(r'//.*?$', '', source_code, flags=re.MULTILINE)

    # Remove multi-line comments (but preserve JSDoc if requested)
    if keep_jsdoc:
        # Only remove /* */ comments, not /** */ (JSDoc)
        source_code = re.sub(r'/\*(?!\*).*?\*/', '', source_code, flags=re.DOTALL)
    else:
        # Remove all /* */ and /** */ comments
        source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)

    # Remove blank lines
    lines = source_code.split('\n')
    lines = [line.rstrip() for line in lines if line.strip()]

    return '\n'.join(lines)


def minify_typescript(source_code: str, keep_jsdoc: bool = False) -> str:
    """
    Minify TypeScript code (same as JavaScript).

    Args:
        source_code: TypeScript source code
        keep_jsdoc: If True, keep JSDoc comments

    Returns:
        Minified code
    """
    return minify_javascript(source_code, keep_jsdoc)


def minify_code(source_code: str, language: str, keep_docs: bool = False) -> str:
    """
    Minify code based on language.

    Args:
        source_code: Source code to minify
        language: Programming language (python, javascript, typescript, etc.)
        keep_docs: Whether to keep documentation (docstrings, JSDoc)

    Returns:
        Minified source code
    """
    language = language.lower()

    if language in ('python', 'py'):
        return minify_python(source_code, keep_docstrings=keep_docs)
    elif language in ('javascript', 'js', 'jsx'):
        return minify_javascript(source_code, keep_jsdoc=keep_docs)
    elif language in ('typescript', 'ts', 'tsx'):
        return minify_typescript(source_code, keep_jsdoc=keep_docs)
    else:
        # For unsupported languages, just remove blank lines and inline comments
        lines = source_code.split('\n')
        # Remove blank lines
        lines = [line.rstrip() for line in lines if line.strip()]
        return '\n'.join(lines)


def estimate_token_savings(original: str, minified: str) -> dict:
    """
    Estimate token savings from minification.

    Args:
        original: Original source code
        minified: Minified source code

    Returns:
        Dict with statistics
    """
    def estimate_tokens(text):
        return len(text) // 4  # Rough estimate: 1 token ≈ 4 chars

    orig_tokens = estimate_tokens(original)
    min_tokens = estimate_tokens(minified)
    savings = orig_tokens - min_tokens
    savings_pct = (savings / orig_tokens * 100) if orig_tokens > 0 else 0

    return {
        'original_tokens': orig_tokens,
        'minified_tokens': min_tokens,
        'tokens_saved': savings,
        'savings_percent': round(savings_pct, 1),
        'original_lines': original.count('\n') + 1,
        'minified_lines': minified.count('\n') + 1,
        'lines_removed': (original.count('\n') + 1) - (minified.count('\n') + 1)
    }


if __name__ == '__main__':
    # Test the minifier
    sample_python = '''
# This is a comment
def hello_world():
    """This is a docstring"""
    # Another comment
    print("Hello, World!")  # Inline comment


class MyClass:
    """
    Multi-line
    docstring
    """
    def method(self):
        return 42
'''

    print("=== Original Python ===")
    print(sample_python)
    print(f"Tokens: {len(sample_python) // 4}")

    minified = minify_python(sample_python, keep_docstrings=False)
    print("\n=== Minified Python ===")
    print(minified)

    stats = estimate_token_savings(sample_python, minified)
    print(f"\n=== Savings ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Test JavaScript
    sample_js = '''
// Single line comment
function hello() {
    /* Multi-line
       comment */
    console.log("Hello");  // Inline comment
}

/**
 * JSDoc comment
 * @param {string} name
 */
function greet(name) {
    return `Hello, ${name}`;
}
'''

    print("\n\n=== Original JavaScript ===")
    print(sample_js)

    minified_js = minify_javascript(sample_js, keep_jsdoc=True)
    print("\n=== Minified JavaScript (keeping JSDoc) ===")
    print(minified_js)

    stats_js = estimate_token_savings(sample_js, minified_js)
    print(f"\n=== Savings ===")
    for k, v in stats_js.items():
        print(f"{k}: {v}")
