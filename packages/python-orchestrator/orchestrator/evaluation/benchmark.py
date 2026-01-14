"""
Benchmark Dataset for CCE Evaluation.

Provides structured examples for evaluating context retrieval methods.

Phase 4, Week 8: Benchmark Dataset

Dataset structure:
- 100 code Q&A examples
- Ground truth files and answers
- Difficulty levels and categories
- Repository metadata
"""

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union


class Difficulty(Enum):
    """Difficulty levels for benchmark examples."""
    EASY = "easy"       # Single file, direct answer
    MEDIUM = "medium"   # 2-3 files, requires understanding
    HARD = "hard"       # Cross-file, architectural understanding


class Category(Enum):
    """Categories of code understanding questions."""
    ARCHITECTURE = "architecture"       # System design, patterns
    API_USAGE = "api_usage"            # How to use APIs
    IMPLEMENTATION = "implementation"   # How code works
    DEBUGGING = "debugging"            # Finding/fixing bugs
    CONFIGURATION = "configuration"    # Setup, config files
    DATA_FLOW = "data_flow"           # How data moves through system


@dataclass
class BenchmarkExample:
    """A single benchmark example for evaluation."""
    id: str
    query: str
    difficulty: Difficulty
    category: Category
    ground_truth_answer: str
    ground_truth_files: List[str]
    ground_truth_keywords: List[str] = field(default_factory=list)
    ground_truth_missing_positions: List[int] = field(default_factory=list)
    source_repository: str = ""

    # Optional context for generation
    initial_context: str = ""
    system_prompt: Optional[str] = None
    language: str = "python"

    # For analysis
    expected_retrieval_count: int = 1
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "query": self.query,
            "difficulty": self.difficulty.value,
            "category": self.category.value,
            "ground_truth_answer": self.ground_truth_answer,
            "ground_truth_files": self.ground_truth_files,
            "ground_truth_keywords": self.ground_truth_keywords,
            "ground_truth_missing_positions": self.ground_truth_missing_positions,
            "source_repository": self.source_repository,
            "initial_context": self.initial_context,
            "system_prompt": self.system_prompt,
            "language": self.language,
            "expected_retrieval_count": self.expected_retrieval_count,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkExample":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            query=data["query"],
            difficulty=Difficulty(data.get("difficulty", "medium")),
            category=Category(data.get("category", "implementation")),
            ground_truth_answer=data["ground_truth_answer"],
            ground_truth_files=data["ground_truth_files"],
            ground_truth_keywords=data.get("ground_truth_keywords", []),
            ground_truth_missing_positions=data.get("ground_truth_missing_positions", []),
            source_repository=data.get("source_repository", data.get("repository", "")),
            initial_context=data.get("initial_context", ""),
            system_prompt=data.get("system_prompt"),
            language=data.get("language", "python"),
            expected_retrieval_count=data.get("expected_retrieval_count", 1),
            notes=data.get("notes", ""),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"BenchmarkExample(id='{self.id}', "
            f"difficulty={self.difficulty.value}, "
            f"files={len(self.ground_truth_files)})"
        )


class BenchmarkDataset:
    """Collection of benchmark examples with filtering and splitting."""

    def __init__(
        self,
        examples: List[BenchmarkExample],
        name: str = "benchmark",
        version: str = "1.0.0",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.examples = examples
        self.name = name
        self.version = version
        self.description = description
        self.metadata = metadata or {}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> BenchmarkExample:
        return self.examples[idx]

    def __iter__(self) -> Iterator[BenchmarkExample]:
        return iter(self.examples)

    def filter(
        self,
        difficulty: Optional[Difficulty] = None,
        category: Optional[Category] = None,
        source_repository: Optional[str] = None,
        language: Optional[str] = None,
    ) -> "BenchmarkDataset":
        """
        Filter examples by criteria.

        Args:
            difficulty: Filter by difficulty level
            category: Filter by category
            source_repository: Filter by repository name
            language: Filter by programming language

        Returns:
            New BenchmarkDataset with filtered examples
        """
        filtered = self.examples

        if difficulty is not None:
            filtered = [ex for ex in filtered if ex.difficulty == difficulty]

        if category is not None:
            filtered = [ex for ex in filtered if ex.category == category]

        if source_repository is not None:
            filtered = [ex for ex in filtered if ex.source_repository == source_repository]

        if language is not None:
            filtered = [ex for ex in filtered if ex.language == language]

        return BenchmarkDataset(
            examples=filtered,
            name=f"{self.name}_filtered",
            version=self.version,
            metadata=self.metadata,
        )

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple["BenchmarkDataset", "BenchmarkDataset"]:
        """
        Split into train and test sets.

        Args:
            train_ratio: Fraction for training
            seed: Random seed for reproducibility

        Returns:
            (train_dataset, test_dataset) tuple
        """
        random.seed(seed)
        shuffled = self.examples.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * train_ratio)
        train = shuffled[:split_idx]
        test = shuffled[split_idx:]

        return (
            BenchmarkDataset(train, f"{self.name}_train", self.version, metadata=self.metadata),
            BenchmarkDataset(test, f"{self.name}_test", self.version, metadata=self.metadata),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        by_difficulty = {}
        by_category = {}
        by_language = {}
        by_repository = {}
        total_files = 0
        total_answer_len = 0

        for ex in self.examples:
            # Count by difficulty
            diff_key = ex.difficulty.value
            by_difficulty[diff_key] = by_difficulty.get(diff_key, 0) + 1

            # Count by category
            cat_key = ex.category.value
            by_category[cat_key] = by_category.get(cat_key, 0) + 1

            # Count by language
            by_language[ex.language] = by_language.get(ex.language, 0) + 1

            # Count by repository
            if ex.source_repository:
                by_repository[ex.source_repository] = by_repository.get(ex.source_repository, 0) + 1

            # Count files and answer length
            total_files += len(ex.ground_truth_files)
            total_answer_len += len(ex.ground_truth_answer)

        return {
            "total_examples": len(self.examples),
            "by_difficulty": by_difficulty,
            "by_category": by_category,
            "by_language": by_language,
            "by_repository": by_repository,
            "avg_ground_truth_files": total_files / len(self.examples) if self.examples else 0,
            "avg_answer_length": total_answer_len / len(self.examples) if self.examples else 0,
        }

    def validate(self) -> List[str]:
        """Validate dataset integrity. Returns list of errors."""
        errors = []
        seen_ids = set()

        for ex in self.examples:
            # Check for duplicate IDs
            if ex.id in seen_ids:
                errors.append(f"Duplicate ID: {ex.id}")
            seen_ids.add(ex.id)

            # Check for empty fields
            if not ex.query.strip():
                errors.append(f"{ex.id}: Empty query")
            if not ex.ground_truth_answer.strip():
                errors.append(f"{ex.id}: Empty ground truth answer")
            if not ex.ground_truth_files:
                errors.append(f"{ex.id}: No ground truth files")

        return errors

    def add_example(self, example: BenchmarkExample):
        """Add an example to the dataset."""
        self.examples.append(example)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metadata": self.metadata,
            "examples": [ex.to_dict() for ex in self.examples],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkDataset":
        """Create from dictionary."""
        examples = [BenchmarkExample.from_dict(ex) for ex in data.get("examples", [])]
        return cls(
            examples=examples,
            name=data.get("name", "benchmark"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        path = Path(path) if isinstance(path, str) else path
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BenchmarkDataset":
        """Load dataset from JSON file."""
        path = Path(path) if isinstance(path, str) else path
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_benchmark(
    examples: Optional[List[Dict[str, Any]]] = None,
    name: str = "reposynth_benchmark",
    version: str = "1.0.0",
    description: str = "",
) -> BenchmarkDataset:
    """
    Create a benchmark dataset.

    Args:
        examples: Optional list of example dictionaries to parse
        name: Dataset name
        version: Version string
        description: Dataset description

    Returns:
        BenchmarkDataset instance
    """
    if examples:
        parsed = [BenchmarkExample.from_dict(ex) for ex in examples]
    else:
        parsed = []

    return BenchmarkDataset(
        examples=parsed,
        name=name,
        version=version,
        description=description,
        metadata={
            "created_by": "RepoSynth",
            "target_size": 100,
        },
    )


def load_benchmark(path: Union[str, Path]) -> BenchmarkDataset:
    """Load a benchmark dataset from file."""
    return BenchmarkDataset.load(path)


def create_sample_benchmark() -> BenchmarkDataset:
    """Create a sample benchmark with a few examples for testing."""
    examples = [
        BenchmarkExample(
            id="sample_001",
            query="How does user authentication work in this Flask app?",
            difficulty=Difficulty.MEDIUM,
            category=Category.ARCHITECTURE,
            ground_truth_answer=(
                "User authentication uses JWT tokens. The user logs in via /login endpoint. "
                "The server creates a JWT token with create_token() function. "
                "The token is verified using verify_token() on protected routes."
            ),
            ground_truth_files=["src/auth/jwt.py", "src/api/routes.py"],
            ground_truth_keywords=["JWT", "token", "login", "create_token", "verify_token", "Authorization"],
            ground_truth_missing_positions=[15, 32, 48],
            source_repository="sample_flask_app",
            language="python",
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="sample_002",
            query="What database models are defined in the application?",
            difficulty=Difficulty.EASY,
            category=Category.IMPLEMENTATION,
            ground_truth_answer=(
                "The application defines a User model with id, username, and password_hash columns. "
                "It uses SQLAlchemy for ORM."
            ),
            ground_truth_files=["src/db/models.py"],
            ground_truth_keywords=["User", "model", "SQLAlchemy", "Column", "Integer", "String"],
            ground_truth_missing_positions=[8, 22],
            source_repository="sample_flask_app",
            language="python",
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="sample_003",
            query="How is the application configured and what environment variables are used?",
            difficulty=Difficulty.EASY,
            category=Category.CONFIGURATION,
            ground_truth_answer=(
                "Configuration is managed through a Config class that reads from environment variables. "
                "It uses SECRET_KEY and DATABASE_URL environment variables."
            ),
            ground_truth_files=["src/config.py"],
            ground_truth_keywords=["Config", "SECRET_KEY", "DATABASE_URL", "environ", "DEBUG"],
            ground_truth_missing_positions=[5, 15],
            source_repository="sample_flask_app",
            language="python",
            expected_retrieval_count=1,
        ),
    ]

    return BenchmarkDataset(
        examples=examples,
        name="sample_benchmark",
        version="0.1.0",
        description="Sample benchmark for testing evaluation framework",
        metadata={
            "is_sample": True,
            "target_size": 3,
        },
    )
