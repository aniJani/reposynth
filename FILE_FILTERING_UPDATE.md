# File Filtering Update - API File Handling

## Problem

API route/endpoint files are typically:
- ❌ Large (lots of boilerplate request/response handling)
- ❌ Repetitive (similar patterns across endpoints)
- ❌ Low-value for understanding core business logic
- ❌ Wasteful of tokens when included in full

**Examples of API files:**
- `src/api/routes.py`, `src/endpoints.py` (Python FastAPI)
- `src/routes/*.ts`, `src/api/*.ts` (TypeScript/Express)
- `src/controllers/*.py` (MVC pattern)

## Solution Implemented

### 1. **New Module: `file_filter.py`**
Location: `packages/python-orchestrator/orchestrator/file_filter.py`

**Features:**
- ✅ Categorizes files into types: CORE_LOGIC, API_ROUTES, CONFIG, TEST, BUILD, DOCUMENTATION
- ✅ Three filtering strategies:
  - **Aggressive**: Exclude API files entirely
  - **Minimal**: Include only API signatures (no implementations)
  - **Permissive**: Include all files
- ✅ Extracts minimal representations of API files (function signatures only)
- ✅ Pattern-based file detection

**File Categories:**
```python
FileCategory.CORE_LOGIC      # Business logic, models, services
FileCategory.API_ROUTES      # API endpoint/route definitions
FileCategory.CONFIG          # Configuration files
FileCategory.TEST            # Test files
FileCategory.BUILD           # Build artifacts (dist/, node_modules/)
FileCategory.DOCUMENTATION   # Docs, README
```

**Filtering Strategies:**
```python
# Default strategy (minimal)
{
    CORE_LOGIC: INCLUDE_FULL,      # Full source code
    API_ROUTES: INCLUDE_MINIMAL,   # Only signatures
    CONFIG: EXCLUDE,                # Skip
    TEST: EXCLUDE,                  # Skip
    BUILD: EXCLUDE,                 # Skip
    DOCUMENTATION: EXCLUDE,         # Skip
}

# Aggressive strategy (exclude APIs)
{
    CORE_LOGIC: INCLUDE_FULL,
    API_ROUTES: EXCLUDE,            # Skip API files entirely
    CONFIG: EXCLUDE,
    TEST: EXCLUDE,
    BUILD: EXCLUDE,
    DOCUMENTATION: EXCLUDE,
}

# Permissive strategy (include everything)
{
    CORE_LOGIC: INCLUDE_FULL,
    API_ROUTES: INCLUDE_FULL,       # Full API files
    CONFIG: EXCLUDE,
    TEST: EXCLUDE,
    BUILD: EXCLUDE,
    DOCUMENTATION: EXCLUDE,
}
```

### 2. **Minimal API Content Extraction**

**Before (Full API file - ~500 tokens):**
```python
# src/api/routes.py
from fastapi import APIRouter, HTTPException
from .models import User
from .auth import authenticate

router = APIRouter()

@router.post("/login")
async def login(username: str, password: str):
    """
    Login endpoint that authenticates users.

    Args:
        username: User's username
        password: User's password

    Returns:
        Authentication token

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Validate input
        if not username or not password:
            raise HTTPException(status_code=400, detail="Missing credentials")

        # Authenticate user
        user = await authenticate(username, password)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Generate token
        token = generate_token(user)

        # Return response
        return {"token": token, "user_id": user.id}
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.get("/users/{user_id}")
async def get_user(user_id: int):
    # ... more implementation ...
```

**After (Minimal API - ~80 tokens):**
```python
from fastapi import APIRouter, HTTPException
from .models import User
from .auth import authenticate
router = APIRouter()
@router.post("/login")
async def login(username: str, password: str):
    pass
@router.get("/users/{user_id}")
async def get_user(user_id: int):
    pass
```

**Token Savings: 84% (420 tokens saved per API file)**

### 3. **Integration with PromptEngine**

**Updated:** `prompt_engine.py`

**New Parameter: `filter_strategy`**
```python
engine = PromptEngine(
    pack_path,
    filter_strategy="minimal"  # "aggressive", "minimal", or "permissive"
)
```

**How it works:**
1. When reading source files, check file category
2. Apply filtering strategy:
   - **EXCLUDE**: Return `None` (skip file)
   - **INCLUDE_MINIMAL**: Extract signatures only
   - **INCLUDE_FULL**: Include full content (possibly minified)

### 4. **Updated API & Schemas**

**Schema:** `schemas.py`
```python
class VibePromptRequest(BaseModel):
    # ... existing fields ...
    filter_strategy: str = Field(
        default="minimal",
        description="File filtering strategy: 'aggressive', 'minimal', or 'permissive'"
    )
```

**API Endpoint:** `api.py`
```python
result = engine_generate_prompt(
    pack_path=str(local_pack_path),
    mode=request.mode,
    # ... other params ...
    filter_strategy=request.filter_strategy
)
```

## Usage Examples

### CLI Usage

```python
from orchestrator.prompt_engine import generate_vibe_prompt

# Minimal strategy (default) - Include minimal API signatures
result = generate_vibe_prompt(
    pack_path="./pack",
    mode="bundle",
    entry_point="src/main.ts",
    filter_strategy="minimal"  # API files = signatures only
)

# Aggressive strategy - Exclude API files entirely
result = generate_vibe_prompt(
    pack_path="./pack",
    mode="bundle",
    entry_point="src/main.ts",
    filter_strategy="aggressive"  # API files = excluded
)

# Permissive strategy - Include full API files
result = generate_vibe_prompt(
    pack_path="./pack",
    mode="bundle",
    entry_point="src/main.ts",
    filter_strategy="permissive"  # API files = full content
)
```

### API Usage

```bash
# Minimal strategy (default)
curl -X POST http://localhost:8000/vibe-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "abc123",
    "mode": "bundle",
    "entry_point": "src/main.ts",
    "filter_strategy": "minimal"
  }'

# Aggressive strategy
curl -X POST http://localhost:8000/vibe-prompt \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "abc123",
    "mode": "bundle",
    "entry_point": "src/main.ts",
    "filter_strategy": "aggressive"
  }'
```

## Token Savings Comparison

### Example: Express API Application

**Files:**
- `src/models/user.ts` (200 tokens) - CORE_LOGIC
- `src/services/auth.ts` (300 tokens) - CORE_LOGIC
- `src/routes/users.ts` (500 tokens) - API_ROUTES
- `src/routes/auth.ts` (450 tokens) - API_ROUTES
- `config/database.ts` (150 tokens) - CONFIG
- `tests/user.test.ts` (400 tokens) - TEST

### Permissive Strategy (Include All)
```
CORE_LOGIC files:   500 tokens
API_ROUTES files:   950 tokens
CONFIG files:         0 tokens (excluded)
TEST files:           0 tokens (excluded)
───────────────────────────────
Total:            1,450 tokens
```

### Minimal Strategy (Default)
```
CORE_LOGIC files:   500 tokens
API_ROUTES files:   150 tokens (signatures only, 84% reduction)
CONFIG files:         0 tokens (excluded)
TEST files:           0 tokens (excluded)
───────────────────────────────
Total:              650 tokens
**Savings: 55% (800 tokens saved)**
```

### Aggressive Strategy (Exclude APIs)
```
CORE_LOGIC files:   500 tokens
API_ROUTES files:     0 tokens (excluded)
CONFIG files:         0 tokens (excluded)
TEST files:           0 tokens (excluded)
───────────────────────────────
Total:              500 tokens
**Savings: 66% (950 tokens saved)**
```

## Benefits

1. **Massive Token Savings**
   - Minimal strategy: 55% reduction (typical)
   - Aggressive strategy: 66% reduction
   - Combined with minification: **70-80% total reduction**

2. **Better Focus**
   - Core business logic gets full representation
   - API boilerplate minimized or removed
   - LLM focuses on important code

3. **Configurable**
   - Choose strategy per request
   - Default is safe (minimal)
   - Can include full APIs when needed

4. **Smart Detection**
   - Pattern-based file categorization
   - Handles Python, TypeScript, JavaScript
   - Extensible for more languages

## Pattern Detection

**API Route Files Detected:**
- `*/api.py`, `*/routes.py`, `*/endpoints.py`
- `*/api/*.py`, `*/routes/*.py`, `*/endpoints/*.py`
- `*/api.ts`, `*/routes.ts`, `*/router.ts`
- `*/api/*.ts`, `*/routes/*.ts`
- `*/controllers/*.{ts,js,py}`

**Build Files Excluded:**
- `dist/*`, `build/*`, `node_modules/*`
- `__pycache__/*`, `.next/*`, `.cache/*`

**Test Files Excluded:**
- `test_*.py`, `*_test.py`
- `*.test.ts`, `*.spec.ts`
- `tests/*`, `__tests__/*`

**Config Files Excluded:**
- `config.{py,ts,js,json,yaml}`
- `.env`, `tsconfig.json`, `package.json`

## Testing

Run the test to see categorization:
```bash
python3 packages/python-orchestrator/orchestrator/file_filter.py
```

Expected output:
```
src/models/user.py        -> core_logic    -> include_full
src/api/routes.py         -> api_routes    -> include_minimal
src/api.py                -> api_routes    -> include_minimal
tests/test_user.py        -> test          -> exclude
config/settings.py        -> config        -> exclude
dist/bundle.js            -> build         -> exclude
src/services/auth.ts      -> core_logic    -> include_full
src/routes/users.ts       -> api_routes    -> include_minimal
```

## Recommendations

### For Most Use Cases:
**Use `"minimal"` strategy (default)**
- Best balance of context and token savings
- Includes API structure without implementation bloat
- 55-60% token reduction

### For Extreme Token Savings:
**Use `"aggressive"` strategy**
- Maximum token savings (66%+)
- Excludes API files entirely
- Best when API logic isn't relevant

### For API-Focused Tasks:
**Use `"permissive"` strategy**
- Include full API implementations
- Useful when debugging API endpoints
- Pair with `keep_docs=True` for API documentation

## Files Changed

1. ✅ `packages/python-orchestrator/orchestrator/file_filter.py` (NEW)
2. ✅ `packages/python-orchestrator/orchestrator/prompt_engine.py` (MODIFIED)
3. ✅ `packages/python-orchestrator/orchestrator/schemas.py` (MODIFIED)
4. ✅ `packages/python-orchestrator/orchestrator/api.py` (MODIFIED)

## Combined Savings: Minification + Filtering

**Example: FastAPI Application**

**Original (no optimization):**
- API files with comments: 2,000 tokens
- Core logic with comments: 1,500 tokens
- Config/tests: 800 tokens
- **Total: 4,300 tokens**

**With Minification Only:**
- API files minified: 1,200 tokens (40% saved)
- Core logic minified: 900 tokens (40% saved)
- Config/tests: 0 tokens (excluded)
- **Total: 2,100 tokens (51% savings)**

**With Minification + Minimal Filtering:**
- API files (minimal): 200 tokens (90% saved)
- Core logic minified: 900 tokens (40% saved)
- Config/tests: 0 tokens (excluded)
- **Total: 1,100 tokens (74% savings)**

**With Minification + Aggressive Filtering:**
- API files: 0 tokens (100% saved)
- Core logic minified: 900 tokens (40% saved)
- Config/tests: 0 tokens (excluded)
- **Total: 900 tokens (79% savings)**

## Summary

The file filtering feature adds intelligent categorization and filtering of files, with special handling for API route files that tend to be large and repetitive. Combined with code minification, this achieves **70-80% total token reduction** while preserving all important business logic.

**Default behavior:**
- ✅ Minification: **enabled**
- ✅ File filtering: **minimal** (API signatures only)
- ✅ Result: **~75% token reduction**

You can now choose between three strategies based on your needs:
- **aggressive**: Maximum savings (exclude APIs)
- **minimal**: Balanced (default - API signatures only)
- **permissive**: Include everything
