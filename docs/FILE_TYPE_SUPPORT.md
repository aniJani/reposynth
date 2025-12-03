# File Type Support

This document describes the file types supported by the RepoSynth parsing pipeline and how to test them.

## Supported File Types

### Fully Supported (AST Parsing + Analysis)

| Language | Extensions | Adapter | What's Extracted |
|----------|------------|---------|------------------|
| Python | `.py` | `PythonAdapter` | Functions, classes, imports, variables, `__all__` exports |
| TypeScript | `.ts`, `.tsx` | `TypeScriptAdapter` | Functions, classes, interfaces, types, enums, imports, exports |
| JavaScript | `.js`, `.jsx` | `JavaScriptAdapter` | Functions, classes, variables, imports (ES6 + CommonJS), exports |
| CSS | `.css` | `CSSAdapter` | Selectors, @-rules (@media, @keyframes), @import, CSS custom properties (--var) |
| SCSS | `.scss` | `SCSSAdapter` | Mixins, functions, $variables, selectors, @import/@use/@forward |
| HTML | `.html`, `.htm` | `HTMLAdapter` | Elements with id, link/script/img imports, data-* attributes |

### Partial Support (Token Counting Only)

These file types are counted for token estimation but don't have full AST analysis:

`.rs`, `.go`, `.java`, `.c`, `.cpp`, `.h`, `.hpp`, `.cs`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, `.vue`, `.svelte`, `.md`, `.json`, `.yaml`, `.yml`

## Architecture

The parsing pipeline consists of:

1. **Rust Parser Daemon** (`packages/rust-parser-daemon/`)
   - Uses tree-sitter grammars to parse files into AST
   - Communicates via JSON over stdin/stdout
   - Grammars located in `grammars/` subdirectory

2. **Python Orchestrator** (`packages/python-orchestrator/`)
   - `parser_client.py` - Discovers files and communicates with daemon
   - `language_adapter.py` - Language-specific AST analysis
   - `estimator.py` - Token estimation
   - `token_utils.py` - Token counting utilities

## Testing

### Prerequisites

1. Build the Rust parser daemon:
```bash
cd packages/rust-parser-daemon
cargo build --release
```

2. Install Python dependencies:
```bash
cd packages/python-orchestrator
pip install -r requirements.txt
```

### Test CSS Parsing

Create a test CSS file:
```css
/* test.css */
@import url('variables.css');

:root {
  --primary-color: #007bff;
  --font-size: 16px;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

#main-header {
  background: var(--primary-color);
}

@media (max-width: 768px) {
  .container {
    padding: 0 15px;
  }
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
```

Expected extractions:
- **Definitions**: `.container`, `#main-header`, `@media (max-width: 768px)`, `@keyframes fadeIn`
- **Imports**: `variables.css`
- **Variables**: `--primary-color`, `--font-size`

### Test SCSS Parsing

Create a test SCSS file:
```scss
/* test.scss */
@use 'variables';
@import 'mixins';

$primary-color: #007bff;
$spacing: 16px;

@mixin flex-center {
  display: flex;
  justify-content: center;
  align-items: center;
}

@function calculate-rem($pixels) {
  @return $pixels / 16 * 1rem;
}

.container {
  @include flex-center;
  padding: $spacing;
}

%button-base {
  border: none;
  cursor: pointer;
}
```

Expected extractions:
- **Definitions**: `@mixin flex-center`, `@function calculate-rem`, `.container`, `%button-base`
- **Imports**: `variables`, `mixins`
- **Variables**: `$primary-color`, `$spacing`

### Test HTML Parsing

Create a test HTML file:
```html
<!-- test.html -->
<!DOCTYPE html>
<html>
<head>
  <link href="styles.css" rel="stylesheet">
  <script src="app.js"></script>
</head>
<body>
  <header id="main-header">
    <nav id="navigation" data-theme="dark">
      <a href="index.html">Home</a>
    </nav>
  </header>
  <main id="content" data-page="home">
    <img src="images/logo.png" alt="Logo">
  </main>
</body>
</html>
```

Expected extractions:
- **Definitions**: `#main-header`, `#navigation`, `#content`
- **Imports**: `styles.css`, `app.js`, `images/logo.png`
- **Variables**: `data-theme`, `data-page`

### Running the Parser

#### Option 1: Direct Python Test

```python
from pathlib import Path
from orchestrator.parser_client import ParserClient
from orchestrator.language_adapter import get_adapter
import json

# Start the daemon
daemon_path = Path("packages/rust-parser-daemon/target/release/rust-parser-daemon.exe")
client = ParserClient(str(daemon_path))

# Parse a single file
test_file = Path("test.css")
# ... (the daemon will output AST JSON)

# Get the appropriate adapter
adapter = get_adapter(test_file)

# Extract definitions, imports, variables
# ... (use adapter methods on the AST)

client.shutdown()
```

#### Option 2: Parse a Repository

```bash
cd packages/python-orchestrator
python -m orchestrator.parser_client
```

This will parse the entire repository and output AST files to `orchestrator/ast_raw/`.

### Verifying the Build

Check that grammars are compiled:
```bash
ls packages/rust-parser-daemon/grammars/
# Should show: tree-sitter-css, tree-sitter-html, tree-sitter-python, tree-sitter-scss, tree-sitter-typescript
```

Check that the daemon handles new file types:
```bash
echo '{"id":"1","path":"test.css"}' | packages/rust-parser-daemon/target/release/rust-parser-daemon.exe
# Should return JSON with AST nodes
```

## Adding New File Types

To add support for a new file type, follow this checklist:

1. [ ] Clone the tree-sitter grammar to `packages/rust-parser-daemon/grammars/`
2. [ ] Add grammar compilation to `build.rs`
3. [ ] Add extern declaration to `main.rs`
4. [ ] Add extension detection to `main.rs`
5. [ ] Create adapter class in `language_adapter.py`
6. [ ] Update `get_adapter()` function
7. [ ] Add glob patterns to `parser_client.py`
8. [ ] Add to `estimator.py` extension mapping
9. [ ] Add to `token_utils.py` source_extensions
10. [ ] Rebuild: `cargo build --release`
11. [ ] Test with sample files

## Planned Additions

### Phase 2: Config Files
- JSON (`.json`)
- YAML (`.yaml`, `.yml`)
- TOML (`.toml`)

### Phase 3: Backend Languages
- Go (`.go`)
- Rust (`.rs`)
- Java (`.java`)

### Phase 4: DevOps & Documentation
- Dockerfile
- Markdown (`.md`)
