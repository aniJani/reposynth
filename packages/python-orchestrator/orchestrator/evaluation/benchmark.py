"""
Benchmark Dataset Module for Evaluation.

This module provides structured benchmark datasets for evaluating
adaptive code generation with uncertainty-triggered retrieval.

Phase 4, Week 8: Benchmark Dataset

Dataset structure:
- 100 code Q&A examples
- Ground truth files and answers
- Difficulty levels and categories
- Repository metadata
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import os


class Difficulty(Enum):
    """Difficulty levels for benchmark examples."""
    EASY = "easy"           # Single file, obvious answer
    MEDIUM = "medium"       # 2-3 files, some inference
    HARD = "hard"           # Multiple files, complex reasoning


class Category(Enum):
    """Categories of code questions."""
    ARCHITECTURE = "architecture"       # System design, structure
    API_USAGE = "api_usage"            # How to use APIs/functions
    IMPLEMENTATION = "implementation"   # How something is implemented
    DEBUGGING = "debugging"            # Finding/fixing issues
    CONFIGURATION = "configuration"    # Config, setup, environment
    DATA_FLOW = "data_flow"           # How data moves through system


@dataclass
class BenchmarkExample:
    """A single benchmark example for evaluation."""
    id: str                              # Unique identifier
    query: str                           # The question being asked
    repository: str                      # Repository name/identifier

    # Ground truth
    ground_truth_files: List[str]        # Files needed to answer
    ground_truth_answer: str             # Expected answer content
    ground_truth_keywords: List[str] = field(default_factory=list)  # Key terms

    # Metadata
    difficulty: Difficulty = Difficulty.MEDIUM
    category: Category = Category.IMPLEMENTATION
    language: str = "python"

    # Optional context
    initial_context: str = ""            # Any provided context
    system_prompt: Optional[str] = None

    # For analysis
    expected_retrieval_count: int = 1    # Expected number of retrievals
    notes: str = ""                      # Additional notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'query': self.query,
            'repository': self.repository,
            'ground_truth_files': self.ground_truth_files,
            'ground_truth_answer': self.ground_truth_answer,
            'ground_truth_keywords': self.ground_truth_keywords,
            'difficulty': self.difficulty.value,
            'category': self.category.value,
            'language': self.language,
            'initial_context': self.initial_context,
            'system_prompt': self.system_prompt,
            'expected_retrieval_count': self.expected_retrieval_count,
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkExample':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            query=data['query'],
            repository=data['repository'],
            ground_truth_files=data['ground_truth_files'],
            ground_truth_answer=data['ground_truth_answer'],
            ground_truth_keywords=data.get('ground_truth_keywords', []),
            difficulty=Difficulty(data.get('difficulty', 'medium')),
            category=Category(data.get('category', 'implementation')),
            language=data.get('language', 'python'),
            initial_context=data.get('initial_context', ''),
            system_prompt=data.get('system_prompt'),
            expected_retrieval_count=data.get('expected_retrieval_count', 1),
            notes=data.get('notes', ''),
        )

    def __repr__(self) -> str:
        return (
            f"BenchmarkExample(id='{self.id}', "
            f"difficulty={self.difficulty.value}, "
            f"files={len(self.ground_truth_files)})"
        )


@dataclass
class BenchmarkDataset:
    """
    Collection of benchmark examples for evaluation.

    Provides utilities for:
    - Loading/saving datasets
    - Filtering by difficulty, category, repository
    - Splitting into train/test sets
    - Statistics and summaries

    Example:
        >>> dataset = BenchmarkDataset.load('benchmark.json')
        >>> print(f"Total examples: {len(dataset)}")
        >>> hard_examples = dataset.filter(difficulty=Difficulty.HARD)
        >>> for example in dataset:
        ...     result = evaluate(example)
    """
    name: str
    version: str
    examples: List[BenchmarkExample]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def filter(
        self,
        difficulty: Optional[Difficulty] = None,
        category: Optional[Category] = None,
        repository: Optional[str] = None,
        language: Optional[str] = None,
    ) -> 'BenchmarkDataset':
        """
        Filter examples by criteria.

        Args:
            difficulty: Filter by difficulty level
            category: Filter by category
            repository: Filter by repository name
            language: Filter by programming language

        Returns:
            New BenchmarkDataset with filtered examples
        """
        filtered = self.examples

        if difficulty:
            filtered = [e for e in filtered if e.difficulty == difficulty]
        if category:
            filtered = [e for e in filtered if e.category == category]
        if repository:
            filtered = [e for e in filtered if e.repository == repository]
        if language:
            filtered = [e for e in filtered if e.language == language]

        return BenchmarkDataset(
            name=f"{self.name}_filtered",
            version=self.version,
            examples=filtered,
            description=f"Filtered subset of {self.name}",
            metadata=self.metadata,
        )

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> tuple:
        """
        Split into train and test sets.

        Args:
            train_ratio: Fraction for training
            seed: Random seed for reproducibility

        Returns:
            (train_dataset, test_dataset) tuple
        """
        import random
        random.seed(seed)

        indices = list(range(len(self.examples)))
        random.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_examples = [self.examples[i] for i in train_indices]
        test_examples = [self.examples[i] for i in test_indices]

        train_ds = BenchmarkDataset(
            name=f"{self.name}_train",
            version=self.version,
            examples=train_examples,
            metadata=self.metadata,
        )
        test_ds = BenchmarkDataset(
            name=f"{self.name}_test",
            version=self.version,
            examples=test_examples,
            metadata=self.metadata,
        )

        return train_ds, test_ds

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'total_examples': len(self.examples),
            'by_difficulty': {},
            'by_category': {},
            'by_repository': {},
            'by_language': {},
            'avg_ground_truth_files': 0,
            'avg_answer_length': 0,
        }

        for diff in Difficulty:
            count = sum(1 for e in self.examples if e.difficulty == diff)
            stats['by_difficulty'][diff.value] = count

        for cat in Category:
            count = sum(1 for e in self.examples if e.category == cat)
            stats['by_category'][cat.value] = count

        repos = {}
        languages = {}
        total_files = 0
        total_answer_len = 0

        for e in self.examples:
            repos[e.repository] = repos.get(e.repository, 0) + 1
            languages[e.language] = languages.get(e.language, 0) + 1
            total_files += len(e.ground_truth_files)
            total_answer_len += len(e.ground_truth_answer)

        stats['by_repository'] = repos
        stats['by_language'] = languages

        if self.examples:
            stats['avg_ground_truth_files'] = total_files / len(self.examples)
            stats['avg_answer_length'] = total_answer_len / len(self.examples)

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'metadata': self.metadata,
            'examples': [e.to_dict() for e in self.examples],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkDataset':
        """Create from dictionary."""
        examples = [BenchmarkExample.from_dict(e) for e in data.get('examples', [])]
        return cls(
            name=data['name'],
            version=data['version'],
            examples=examples,
            description=data.get('description', ''),
            metadata=data.get('metadata', {}),
        )

    def save(self, path: str):
        """Save dataset to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'BenchmarkDataset':
        """Load dataset from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def add_example(self, example: BenchmarkExample):
        """Add an example to the dataset."""
        self.examples.append(example)

    def validate(self) -> List[str]:
        """
        Validate dataset integrity.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        ids = set()
        for i, example in enumerate(self.examples):
            # Check for duplicate IDs
            if example.id in ids:
                errors.append(f"Duplicate ID: {example.id}")
            ids.add(example.id)

            # Check required fields
            if not example.query:
                errors.append(f"Example {example.id}: empty query")
            if not example.ground_truth_files:
                errors.append(f"Example {example.id}: no ground truth files")
            if not example.ground_truth_answer:
                errors.append(f"Example {example.id}: no ground truth answer")

        return errors


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_benchmark(
    name: str = "reposynth_benchmark",
    version: str = "1.0.0",
    description: str = "",
) -> BenchmarkDataset:
    """
    Create an empty benchmark dataset.

    Args:
        name: Dataset name
        version: Version string
        description: Dataset description

    Returns:
        Empty BenchmarkDataset ready for examples
    """
    return BenchmarkDataset(
        name=name,
        version=version,
        examples=[],
        description=description,
        metadata={
            'created_by': 'RepoSynth',
            'target_size': 100,
        },
    )


def load_benchmark(path: str) -> BenchmarkDataset:
    """Load a benchmark dataset from file."""
    return BenchmarkDataset.load(path)


def create_sample_benchmark() -> BenchmarkDataset:
    """
    Create a sample benchmark with a few examples for testing.

    Returns:
        BenchmarkDataset with sample examples
    """
    examples = [
        BenchmarkExample(
            id="sample_001",
            query="How does user authentication work in this Flask app?",
            repository="sample-flask-app",
            ground_truth_files=["src/auth/jwt.py", "src/api/routes.py"],
            ground_truth_answer="""
User authentication uses JWT tokens:
1. User logs in via /login endpoint
2. Server creates JWT token with create_token()
3. Token is returned to client
4. Client includes token in Authorization header
5. Server verifies token with verify_token()
""",
            ground_truth_keywords=["JWT", "token", "login", "verify", "Authorization"],
            difficulty=Difficulty.MEDIUM,
            category=Category.ARCHITECTURE,
            language="python",
            expected_retrieval_count=2,
        ),
        BenchmarkExample(
            id="sample_002",
            query="What database models are defined in the application?",
            repository="sample-flask-app",
            ground_truth_files=["src/db/models.py"],
            ground_truth_answer="""
The User model is defined with SQLAlchemy:
- id: Integer primary key
- username: Unique string
- password_hash: String for hashed password
""",
            ground_truth_keywords=["User", "SQLAlchemy", "model", "Column", "Integer", "String"],
            difficulty=Difficulty.EASY,
            category=Category.IMPLEMENTATION,
            language="python",
            expected_retrieval_count=1,
        ),
        BenchmarkExample(
            id="sample_003",
            query="How is the application configured and what environment variables are used?",
            repository="sample-flask-app",
            ground_truth_files=["src/config.py"],
            ground_truth_answer="""
Configuration is handled by the Config class:
- SECRET_KEY: From SECRET_KEY env var, defaults to 'dev-key'
- DATABASE_URL: From DATABASE_URL env var, defaults to 'sqlite:///app.db'
- DEBUG: From DEBUG env var, defaults to False
""",
            ground_truth_keywords=["Config", "SECRET_KEY", "DATABASE_URL", "environ", "DEBUG"],
            difficulty=Difficulty.EASY,
            category=Category.CONFIGURATION,
            language="python",
            expected_retrieval_count=1,
        ),
    ]

    return BenchmarkDataset(
        name="sample_benchmark",
        version="0.1.0",
        examples=examples,
        description="Sample benchmark for testing evaluation framework",
        metadata={
            'is_sample': True,
            'target_size': 3,
        },
    )
