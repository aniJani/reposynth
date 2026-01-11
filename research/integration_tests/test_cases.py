"""
Test Cases for Week 7 Integration Tests.

Defines 10 comprehensive test scenarios covering:
- Missing context scenarios (5 tests)
- Language choice only (3 tests)
- Mixed uncertainty (2 tests)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TestCase:
    """Complete specification of an integration test case."""

    # --- Test Identity ---
    test_id: str = ""
    name: str = ""
    category: str = ""

    # --- Input ---
    query: str = ""
    initial_context: str = ""
    system_prompt: Optional[str] = None

    # --- Expected Behavior ---
    expected_retrieval: bool = False
    expected_n_retrievals: int = 0
    expected_total_tokens: int = 0
    expected_max_entropy: float = 0.0

    # --- Validation ---
    validate_entropy_trace: bool = False
    entropy_check_pattern: str = ""

    validate_retrieval_query: bool = False
    expected_query_keywords: List[str] = field(default_factory=list)

    validate_context_budget: bool = False
    expected_context_utilization: float = 0.0

    # --- Assertions ---
    critical_assertions: List[str] = field(default_factory=list)
    optional_assertions: List[str] = field(default_factory=list)

    # --- Notes ---
    description: str = ""
    notes: str = ""


# ==============================================================================
# TEST CASE DEFINITIONS
# ==============================================================================

TEST_CASES: List[TestCase] = [
    TestCase(
        test_id="TC001",
        name="Pandas API Without Context",
        category="missing_context",
        query="How do I use pandas.read_csv() to read a file with custom delimiter?",
        initial_context="",
        system_prompt=None,
        expected_retrieval=True,
        expected_n_retrievals=1,
        expected_total_tokens=80,
        expected_max_entropy=4.0,
        validate_entropy_trace=True,
        entropy_check_pattern="high_spike",
        validate_retrieval_query=True,
        expected_query_keywords=["pandas", "read_csv", "delimiter", "sep"],
        validate_context_budget=True,
        expected_context_utilization=0.6,
        critical_assertions=[
            "Entropy spike detected at API call position",
            "Retrieval triggered once",
            "Query contains 'pandas read_csv'",
            "Response mentions read_csv parameters",
            "Context stays within budget",
        ],
        description="Tests that system detects missing API knowledge and retrieves documentation.",
    ),
    TestCase(
        test_id="TC002",
        name="React State Management",
        category="missing_context",
        query="How do I update state in React after an API call?",
        initial_context="",
        system_prompt=None,
        expected_retrieval=True,
        expected_n_retrievals=1,
        expected_total_tokens=100,
        expected_max_entropy=3.8,
        validate_entropy_trace=True,
        entropy_check_pattern="high_spike",
        validate_retrieval_query=True,
        expected_query_keywords=["React", "useState", "setState", "API"],
        validate_context_budget=True,
        expected_context_utilization=0.55,
        critical_assertions=[
            "Entropy spikes at 'useState' or 'setState'",
            "Retrieval triggered",
            "Query includes React hooks",
            "Response shows setState usage",
        ],
        description="Tests React-specific knowledge retrieval.",
    ),
    TestCase(
        test_id="TC003",
        name="Firebase Authentication Setup",
        category="missing_context",
        query="How do I implement Firebase authentication with email/password?",
        initial_context="",
        system_prompt=None,
        expected_retrieval=True,
        expected_n_retrievals=1,
        expected_total_tokens=120,
        expected_max_entropy=4.2,
        validate_entropy_trace=True,
        entropy_check_pattern="high_spike",
        validate_retrieval_query=True,
        expected_query_keywords=["Firebase", "auth", "signIn", "email", "password"],
        critical_assertions=[
            "Retrieval triggered for Firebase auth",
            "Response includes signInWithEmailAndPassword",
        ],
        description="Tests Firebase API knowledge retrieval.",
    ),
    TestCase(
        test_id="TC004",
        name="FastAPI Route Dependencies",
        category="missing_context",
        query="How do I use Depends() for authentication in FastAPI routes?",
        initial_context="",
        system_prompt=None,
        expected_retrieval=True,
        expected_n_retrievals=1,
        expected_total_tokens=90,
        expected_max_entropy=3.9,
        validate_retrieval_query=True,
        expected_query_keywords=["FastAPI", "Depends", "authentication", "route"],
        critical_assertions=[
            "Retrieval triggered for Depends()",
            "Response shows dependency injection pattern",
        ],
        description="Tests FastAPI dependency injection retrieval.",
    ),
    TestCase(
        test_id="TC005",
        name="NumPy Array Operations",
        category="missing_context",
        query="How do I reshape a NumPy array to 2D?",
        initial_context="",
        system_prompt=None,
        expected_retrieval=True,
        expected_n_retrievals=1,
        expected_total_tokens=70,
        expected_max_entropy=3.5,
        validate_retrieval_query=True,
        expected_query_keywords=["NumPy", "array", "reshape", "2D"],
        critical_assertions=[
            "Retrieval triggered for reshape method",
            "Response shows np.reshape() or .reshape()",
        ],
        description="Tests NumPy array manipulation retrieval.",
    ),
    TestCase(
        test_id="TC006",
        name="Synonym Definition Question",
        category="language_choice",
        query="What is authentication?",
        initial_context="",
        system_prompt=None,
        expected_retrieval=False,
        expected_n_retrievals=0,
        expected_total_tokens=60,
        expected_max_entropy=1.5,
        validate_entropy_trace=True,
        entropy_check_pattern="low_flat",
        validate_context_budget=False,
        expected_context_utilization=0.0,
        critical_assertions=[
            "No retrieval events",
            "Entropy remains low throughout",
            "Response is natural language explanation",
        ],
        description="Tests that system doesn't retrieve for pure language questions.",
    ),
    TestCase(
        test_id="TC007",
        name="How-To Phrasing Question",
        category="language_choice",
        query="How does a REST API work?",
        initial_context="",
        system_prompt=None,
        expected_retrieval=False,
        expected_n_retrievals=0,
        expected_total_tokens=80,
        expected_max_entropy=1.8,
        validate_entropy_trace=True,
        entropy_check_pattern="low_flat",
        validate_context_budget=False,
        expected_context_utilization=0.0,
        critical_assertions=[
            "No retrieval events",
            "Low entropy throughout",
            "Response is conceptual explanation",
        ],
        description="Tests language-only conceptual explanation.",
    ),
    TestCase(
        test_id="TC008",
        name="Architecture Overview Question",
        category="language_choice",
        query="What are the main components of a web application?",
        initial_context="",
        system_prompt=None,
        expected_retrieval=False,
        expected_n_retrievals=0,
        expected_total_tokens=100,
        expected_max_entropy=2.0,
        validate_entropy_trace=True,
        entropy_check_pattern="low_flat",
        validate_context_budget=False,
        expected_context_utilization=0.0,
        critical_assertions=[
            "No retrieval",
            "Low entropy",
            "Response lists components (frontend, backend, database)",
        ],
        description="Tests that general architecture questions don't trigger retrieval.",
    ),
    TestCase(
        test_id="TC009",
        name="Code Question With Partial Context",
        category="mixed",
        query="How do I implement a custom error handler in FastAPI?",
        initial_context="FastAPI provides HTTPException for raising errors.\nYou can create exception handlers for specific exception types.",
        system_prompt=None,
        expected_retrieval=True,
        expected_n_retrievals=1,
        expected_total_tokens=90,
        expected_max_entropy=2.8,
        validate_entropy_trace=True,
        entropy_check_pattern="mixed",
        validate_retrieval_query=True,
        expected_query_keywords=["FastAPI", "exception", "handler", "custom"],
        critical_assertions=[
            "Initial part has low entropy (context present)",
            "Entropy spikes at custom handler part",
            "Retrieval triggered at spike",
        ],
        description="Tests mixed scenario with partial context.",
    ),
    TestCase(
        test_id="TC010",
        name="Complex Multi-Part Question",
        category="mixed",
        query="Compare authentication in FastAPI vs Firebase Auth",
        initial_context="",
        system_prompt=None,
        expected_retrieval=True,
        expected_n_retrievals=2,
        expected_total_tokens=150,
        expected_max_entropy=3.5,
        validate_entropy_trace=True,
        entropy_check_pattern="mixed",
        critical_assertions=[
            "Two retrieval events (one for each)",
            "Both queries relevant",
            "Context budget respected",
        ],
        description="Tests multiple retrieval scenario.",
    ),
]


def get_test_case(test_id: str) -> Optional[TestCase]:
    """Get test case by ID."""
    for case in TEST_CASES:
        if case.test_id == test_id:
            return case
    return None


def get_test_cases_by_category(category: str) -> List[TestCase]:
    """Get all test cases in a category."""
    return [c for c in TEST_CASES if c.category == category]
