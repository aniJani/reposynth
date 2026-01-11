"""
Mock Retriever for Week 7 Integration Tests.

Full simulation of RepoSynth retrieval system with realistic ranking.
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import random


class MockRetriever:
    """
    Full simulation of RepoSynth retrieval system.

    Simulates:
    - Semantic search with relevance scoring
    - Multiple data sources (codebase, docs, external)
    - Ranking and top-k selection
    - Latency simulation (based on document size)
    - Empty results (no matches found)
    - Context window limits
    """

    # Built-in mock data for common libraries
    DEFAULT_DOCUMENTS: Dict[str, str] = {
        # Python Data Science
        "pandas_read_csv": """
pandas.read_csv() reads CSV files into a DataFrame.

Parameters:
    filepath_or_buffer: Path to file or file-like object
    sep: Delimiter to use (default: ',')
    header: Row number for column names (default: 0)
    names: Column names if no header (default: None)
    usecols: Columns to read (default: None = all)
    dtype: Data type for columns (default: None)
    engine: Parser engine ('c', 'python', 'pyarrow')
    skiprows: Rows to skip at start (default: None)
    skipfooter: Rows to skip at end (default: None)
    nrows: Number of rows to read (default: None)

Returns: pandas.DataFrame

Example:
    df = pd.read_csv('data.csv', sep=',', header=0)
    """,
        "numpy_array": """
numpy.array() creates an N-dimensional array.

Parameters:
    object: Array-like input (list, tuple, ndarray, etc.)
    dtype: Desired data type (default: None - inferred)
    copy: Whether to copy input (default: False)
    order: Memory layout ('C', 'F', 'A', 'K')
    subok: Allow subclasses (default: False)
    ndmin: Minimum dimensions (default: 0)

Returns: numpy.ndarray

Methods:
    reshape(shape): Change array shape
    flatten(): Return flattened copy
    transpose(): Transpose axes
    dot(other): Dot product
    mean(axis): Compute mean
    sum(axis): Compute sum

Example:
    arr = np.array([[1, 2], [3, 4]])
    arr = arr.reshape(4, 2)
    """,
        # React
        "react_hooks": """
React Hooks for functional components.

useState():
    Manage state in function components
    const [state, setState] = useState(initialValue)
    
useEffect():
    Handle side effects (API calls, subscriptions)
    useEffect(() => { cleanup }, [dependencies])
    
useCallback():
    Memoize callback function
    const memoizedCallback = useCallback(callback, deps)
    
useMemo():
    Memoize expensive calculation
    const memoizedValue = useMemo(calculation, deps)
    
useState Guidelines:
    - Initialize with default value
    - Use setter for updates (don't mutate directly)
    - Updates trigger re-render
    """,
        "react_components": """
React Component Best Practices.

Functional Components:
    const Component = ({ prop1, prop2 }) => {
        return <div>{prop1} {prop2}</div>
    }

Props:
    - Pass data down to child components
    - Immutable (don't modify)
    - Destructure for clarity

Lifecycle:
    - Mount: Component rendered to DOM
    - Update: Props or state changed
    - Unmount: Component removed from DOM

Example:
    const UserProfile = ({ name, email }) => {
        return (
            <div className="profile">
                <h2>{name}</h2>
                <p>{email}</p>
            </div>
        )
    }
    """,
        # Firebase
        "firebase_auth": """
Firebase Authentication Methods.

Email/Password Auth:
    signInWithEmailAndPassword(email, password)
        Creates user or signs in existing user
        Returns: UserCredential
    
    createUserWithEmailAndPassword(email, password)
        Creates new user account
        Returns: UserCredential
    
Sign Out:
    signOut()
        Signs out current user
        Returns: Promise<void>

Error Handling:
    try {
        await auth.signInWithEmailAndPassword(email, password)
    } catch (error) {
        // error.code: 'auth/invalid-email', 'auth/wrong-password', etc.
        // error.message: Human-readable error
    }

Common Error Codes:
    - auth/invalid-email: Email format invalid
    - auth/user-disabled: User account disabled
    - auth/user-not-found: Email not registered
    - auth/wrong-password: Incorrect password
    - auth/weak-password: Password too weak
    """,
        "firebase_firestore": """
Cloud Firestore NoSQL Database.

Structure:
    Database → Collection → Documents → Fields

Collection Operations:
    collection('users'): Get collection reference
    addDoc(collectionRef, data): Add document (auto ID)
    setDoc(docRef, data): Create or replace document
    getDoc(docRef): Read document
    updateDoc(docRef, data): Update document fields
    deleteDoc(docRef): Delete document

Queries:
    query(collection, where('age', '>=', 18)): Filter
    orderBy(collection, 'name'): Sort
    limit(query, 10): Restrict results

Example:
    const db = getFirestore()
    const usersRef = collection(db, 'users')
    await addDoc(usersRef, { name: 'John', age: 30 })
    """,
        # FastAPI
        "fastapi_routes": """
FastAPI Route Decorators and Patterns.

Route Decorators:
    @app.get('/items/{item_id}')
    @app.post('/items')
    @app.put('/items/{item_id}')
    @app.delete('/items/{item_id}')
    @app.patch('/items/{item_id}')

Path Parameters:
    @app.get('/items/{item_id}')
    async def read_item(item_id: int):
        return {'item_id': item_id}

Query Parameters:
    @app.get('/items/')
    async def read_items(skip: int = 0, limit: int = 10):
        return {'skip': skip, 'limit': limit}

Request Body:
    @app.post('/items/')
    async def create_item(item: Item):
        return item

Return Types:
    - JSON: dict or list (auto-serialized)
    - Response: HTTPResponse, JSONResponse, RedirectResponse
    - FileResponse: For file downloads
    - StreamingResponse: For streaming data
    """,
        "fastapi_dependency": """
FastAPI Dependency Injection System.

Depends() Usage:
    from fastapi import Depends

    # Define dependency
    async def get_db():
        return DatabaseConnection()
    
    # Use in route
    @app.get('/users/')
    async def get_users(db: DatabaseConnection = Depends(get_db)):
        return db.query('SELECT * FROM users')

Common Dependencies:
    - Authentication: get_current_user()
    - Database: get_db_session()
    - Configuration: get_settings()
    - Rate limiting: check_rate_limit()

Class-Based Dependencies:
    class CommonQueryParams:
        def __init__(self, skip: int = 0, limit: int = 100):
            self.skip = skip
            self.limit = limit
    
    @app.get('/items/')
    async def read_items(commons: CommonQueryParams = Depends()):
        return commons

Dependency Override:
    # For testing
    app.dependency_overrides[get_db] = override_get_db
    """,
    }

    def __init__(
        self,
        documents: Optional[Dict[str, str]] = None,
        embedding_dim: int = 384,
        latency_base_ms: float = 50.0,
        latency_per_doc_ms: float = 5.0,
        seed: int = 42,
    ):
        """
        Initialize mock retriever.

        Args:
            documents: Dict mapping {source: content}
            embedding_dim: Dimension of simulated embeddings
            latency_base_ms: Base retrieval latency (ms)
            latency_per_doc_ms: Additional latency per document searched
            seed: Random seed for reproducibility
        """
        self.documents = documents or self.DEFAULT_DOCUMENTS.copy()
        self.embedding_dim = embedding_dim
        self.latency_base = latency_base_ms
        self.latency_per_doc = latency_per_doc_ms
        self.seed = seed
        random.seed(seed)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        source_filter: Optional[List[str]] = None,
        min_relevance: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query.

        Returns:
            List of dicts with:
            - source: str (document identifier)
            - content: str (document content)
            - score: float (relevance score [0, 1])
            - retrieval_time_ms: float (simulated latency)
            - metadata: dict (optional fields)

        Args:
            query: Search query string
            top_k: Number of results to return
            source_filter: Optional list of allowed sources
            min_relevance: Minimum relevance score to include
        """
        # Simulate retrieval latency
        start_time = time.time()

        # Score all documents
        scored_docs = []
        query_lower = query.lower()

        for source, content in self.documents.items():
            # Check source filter
            if source_filter:
                if not any(allowed in source for allowed in source_filter):
                    continue

            # Compute relevance
            score = self._compute_relevance(query_lower, source, content)

            if score >= min_relevance:
                scored_docs.append((source, content, score))

        # Sort by relevance
        scored_docs.sort(key=lambda x: -x[2])  # Descending

        # Select top-k
        top_results = scored_docs[:top_k]

        # Build result list
        results = []
        for i, (source, content, score) in enumerate(top_results):
            retrieval_time = self._simulate_latency(len(self.documents))

            result = {
                "source": source,
                "content": content.strip(),
                "score": score,
                "retrieval_time_ms": retrieval_time,
                "metadata": {
                    "rank": i + 1,
                    "match_type": self._detect_match_type(query_lower, content),
                },
            }
            results.append(result)

        return results

    def _compute_relevance(self, query: str, source: str, document: str) -> float:
        """
        Compute relevance score [0, 1].

        Factors:
        1. Keyword overlap (exact word matches)
        2. Semantic similarity (simulated via word embeddings)
        3. Query specificity (longer queries = higher precision)
        4. Document length (shorter = more precise)

        Formula:
            score = 0.4 * keyword_overlap +
                     0.4 * semantic_similarity +
                     0.1 * specificity_bonus +
                     0.1 * length_penalty
        """
        query_words = set(query.lower().split())
        source_lower = source.lower()
        doc_lower = document.lower()

        # 1. Keyword overlap
        keyword_matches = 0
        for word in query_words:
            if len(word) < 3:  # Skip very short words
                continue
            if word in doc_lower:
                keyword_matches += 1
            elif word in source_lower:
                keyword_matches += 0.5  # Partial credit for source match

        keyword_score = min(1.0, keyword_matches / max(1, len(query_words)))

        # 2. Semantic similarity (simulated via simple word embedding)
        semantic_score = self._simulate_semantic_similarity(query_lower, doc_lower)

        # 3. Specificity bonus (longer queries = more specific)
        specificity = min(1.0, len(query) / 50.0)

        # 4. Length penalty (prefer shorter, more precise docs)
        length_penalty = max(0.5, 1.0 - min(0.5, len(document) / 2000.0))

        # Combine
        score = (
            0.4 * keyword_score
            + 0.4 * semantic_score
            + 0.1 * specificity
            + 0.1 * length_penalty
        )

        return min(1.0, max(0.0, score))

    def _simulate_semantic_similarity(self, query: str, document: str) -> float:
        """
        Simulate semantic similarity without real embeddings.

        Uses simple heuristics:
        - Common words
        - Jaccard similarity of word sets
        """
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())

        if not query_words or not doc_words:
            return 0.0

        # Jaccard similarity
        intersection = len(query_words & doc_words)
        union = len(query_words | doc_words)
        jaccard = intersection / union if union > 0 else 0.0

        return jaccard

    def _detect_match_type(self, query: str, document: str) -> str:
        """Detect how query matched document."""
        query_lower = query.lower()
        doc_lower = document.lower()

        # Exact phrase match
        if query_lower in doc_lower:
            return "exact_phrase"

        # Partial phrase match
        for word in query_lower.split():
            if word in doc_lower:
                return "partial_word"

        return "semantic"

    def _simulate_latency(self, n_docs_searched: int) -> float:
        """
        Simulate realistic retrieval latency.

        Model:
            latency = base + per_doc * n_docs_searched + noise
        """
        import random

        base = self.latency_base
        per_doc = self.latency_per_doc * n_docs_searched
        noise = random.gauss(0, 10)  # 10ms std dev

        latency = base + per_doc + noise
        return max(0, latency)

    def add_document(self, source: str, content: str):
        """Add a document to the store."""
        self.documents[source] = content

    def remove_document(self, source: str) -> bool:
        """Remove a document from the store."""
        if source in self.documents:
            del self.documents[source]
            return True
        return False

    def clear_documents(self):
        """Clear all documents."""
        self.documents.clear()

    def get_document_count(self) -> int:
        """Get number of documents."""
        return len(self.documents)
