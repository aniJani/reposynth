"""
Unit tests for token_classifier module

Tests keyword-based token classification for CCE computation.
Week 2, Day 3: Keyword Sets & Fast-Path Classification
"""

import unittest
from orchestrator.entropy.token_classifier import (
    KeywordClassifier,
    classify_token,
    build_code_keyword_set,
    build_language_keyword_set,
    get_keyword_stats,
    CODE_KEYWORDS,
    LANGUAGE_WORDS,
    CODE_OPERATORS,
)


class TestKeywordSets(unittest.TestCase):
    """Test keyword set definitions."""

    def test_code_keywords_not_empty(self):
        """Code keywords set should not be empty."""
        self.assertGreater(len(CODE_KEYWORDS), 0)

    def test_language_words_not_empty(self):
        """Language words set should not be empty."""
        self.assertGreater(len(LANGUAGE_WORDS), 0)

    def test_code_operators_not_empty(self):
        """Code operators set should not be empty."""
        self.assertGreater(len(CODE_OPERATORS), 0)

    def test_no_overlap_code_language(self):
        """Code and language sets should not overlap."""
        overlap = CODE_KEYWORDS & LANGUAGE_WORDS
        self.assertEqual(len(overlap), 0,
                        f"Found overlap between code and language: {overlap}")

    def test_domain_specific_keywords_included(self):
        """Critical domain-specific keywords should be included."""
        # These were problematic in Week 1 POC
        critical_keywords = {
            'pandas', 'numpy', 'requests', 'firebase', 'useState',
            'FastAPI', 'React', 'DataFrame', 'axios',
        }

        missing = critical_keywords - CODE_KEYWORDS
        self.assertEqual(len(missing), 0,
                        f"Missing critical domain keywords: {missing}")

    def test_common_language_words_included(self):
        """Common documentation words should be included."""
        common_words = {
            'explain', 'show', 'tell', 'how', 'what', 'the', 'are',
        }

        missing = common_words - LANGUAGE_WORDS
        self.assertEqual(len(missing), 0,
                        f"Missing common language words: {missing}")

    def test_keyword_set_sizes(self):
        """Keyword sets should meet minimum size requirements."""
        # We want comprehensive coverage
        self.assertGreater(len(CODE_KEYWORDS), 100,
                          f"Code keywords too small: {len(CODE_KEYWORDS)}")
        self.assertGreater(len(LANGUAGE_WORDS), 50,
                          f"Language words too small: {len(LANGUAGE_WORDS)}")


class TestKeywordClassifier(unittest.TestCase):
    """Test KeywordClassifier class."""

    def setUp(self):
        """Create classifier instance for tests."""
        self.classifier = KeywordClassifier()

    def test_initialization(self):
        """Classifier should initialize properly."""
        self.assertIsNotNone(self.classifier.code_keywords)
        self.assertIsNotNone(self.classifier.language_words)
        self.assertIsNotNone(self.classifier.code_operators)

    def test_classify_python_keywords(self):
        """Should classify Python keywords as code."""
        python_keywords = ['def', 'class', 'if', 'for', 'import', 'return']
        for kw in python_keywords:
            result = self.classifier.classify(kw)
            self.assertEqual(result, 'code',
                           f"'{kw}' should be classified as 'code'")

    def test_classify_javascript_keywords(self):
        """Should classify JavaScript keywords as code."""
        js_keywords = ['const', 'let', 'function', 'async', 'await']
        for kw in js_keywords:
            result = self.classifier.classify(kw)
            self.assertEqual(result, 'code',
                           f"'{kw}' should be classified as 'code'")

    def test_classify_domain_specific_libraries(self):
        """Should classify domain-specific library names as code."""
        libraries = ['pandas', 'numpy', 'requests', 'firebase', 'FastAPI',
                    'React', 'useState', 'DataFrame', 'axios']
        for lib in libraries:
            result = self.classifier.classify(lib)
            self.assertEqual(result, 'code',
                           f"'{lib}' should be classified as 'code'")

    def test_classify_operators(self):
        """Should classify operators as code."""
        operators = ['+', '-', '==', '!=', '&&', '||', '{', '}', ';']
        for op in operators:
            result = self.classifier.classify(op)
            self.assertEqual(result, 'code',
                           f"'{op}' should be classified as 'code'")

    def test_classify_language_words(self):
        """Should classify language words correctly."""
        language_words = ['explain', 'show', 'how', 'what', 'the', 'are']
        for word in language_words:
            result = self.classifier.classify(word)
            self.assertEqual(result, 'language',
                           f"'{word}' should be classified as 'language'")

    def test_classify_case_insensitive(self):
        """Classification should be case-insensitive for keywords."""
        # Python keyword in different cases
        self.assertEqual(self.classifier.classify('def'), 'code')
        self.assertEqual(self.classifier.classify('Def'), 'code')
        self.assertEqual(self.classifier.classify('DEF'), 'code')

        # Language word in different cases
        self.assertEqual(self.classifier.classify('explain'), 'language')
        self.assertEqual(self.classifier.classify('Explain'), 'language')
        self.assertEqual(self.classifier.classify('EXPLAIN'), 'language')

    def test_classify_unknown_tokens(self):
        """Should return None for unknown tokens."""
        unknown_tokens = [
            'foobar',  # Random word
            'xyzabc',  # Random identifier
            '12345',   # Number
            'my_custom_variable',  # Custom identifier
        ]
        for token in unknown_tokens:
            result = self.classifier.classify(token)
            self.assertIsNone(result,
                            f"'{token}' should return None (unknown)")

    def test_classify_whitespace_handling(self):
        """Should handle tokens with whitespace."""
        # Should strip whitespace
        self.assertEqual(self.classifier.classify('  def  '), 'code')
        self.assertEqual(self.classifier.classify(' explain '), 'language')

    def test_get_coverage_empty_list(self):
        """Coverage of empty list should be all zeros."""
        coverage = self.classifier.get_coverage([])
        self.assertEqual(coverage['code'], 0.0)
        self.assertEqual(coverage['language'], 0.0)
        self.assertEqual(coverage['unknown'], 0.0)

    def test_get_coverage_all_code(self):
        """Coverage of all code tokens."""
        tokens = ['def', 'class', 'if', 'pandas', 'React']
        coverage = self.classifier.get_coverage(tokens)
        self.assertEqual(coverage['code'], 1.0)
        self.assertEqual(coverage['language'], 0.0)
        self.assertEqual(coverage['unknown'], 0.0)

    def test_get_coverage_all_language(self):
        """Coverage of all language tokens."""
        tokens = ['explain', 'show', 'how', 'what', 'the']
        coverage = self.classifier.get_coverage(tokens)
        self.assertEqual(coverage['code'], 0.0)
        self.assertEqual(coverage['language'], 1.0)
        self.assertEqual(coverage['unknown'], 0.0)

    def test_get_coverage_mixed(self):
        """Coverage of mixed tokens."""
        tokens = ['def', 'explain', 'unknown1', 'if', 'show', 'unknown2']
        coverage = self.classifier.get_coverage(tokens)

        # 2 code (def, if), 2 language (explain, show), 2 unknown out of 6 total
        self.assertAlmostEqual(coverage['code'], 2/6)
        self.assertAlmostEqual(coverage['language'], 2/6)
        self.assertAlmostEqual(coverage['unknown'], 2/6)

    def test_get_coverage_sums_to_one(self):
        """Coverage proportions should sum to 1.0."""
        tokens = ['def', 'explain', 'unknown', 'pandas', 'how']
        coverage = self.classifier.get_coverage(tokens)

        total = coverage['code'] + coverage['language'] + coverage['unknown']
        self.assertAlmostEqual(total, 1.0)


class TestRealWorldExamples(unittest.TestCase):
    """Test classifier on real-world code examples."""

    def setUp(self):
        """Create classifier instance for tests."""
        self.classifier = KeywordClassifier()

    def test_python_import_statement(self):
        """Test tokens from Python import statement."""
        # "import pandas as pd"
        tokens = ['import', 'pandas', 'as', 'pd']
        results = [self.classifier.classify(t) for t in tokens]

        # All should be 'code'
        self.assertEqual(results, ['code', 'code', 'code', 'code'])

    def test_react_hook_usage(self):
        """Test tokens from React hook usage."""
        # "const [state, setState] = useState(0)"
        tokens = ['const', 'useState']
        results = [self.classifier.classify(t) for t in tokens]

        # Both should be 'code'
        self.assertEqual(results, ['code', 'code'])

    def test_firebase_authentication(self):
        """Test tokens from Firebase auth code."""
        # "firebase.auth().signInWithEmailAndPassword"
        tokens = ['firebase', 'auth', 'signInWithEmailAndPassword']
        results = [self.classifier.classify(t) for t in tokens]

        # All should be 'code'
        self.assertEqual(results, ['code', 'code', 'code'])

    def test_documentation_prompt(self):
        """Test tokens from documentation prompt."""
        # "Explain how the function works"
        tokens = ['Explain', 'how', 'the', 'function', 'works']
        results = [self.classifier.classify(t) for t in tokens]

        # 'Explain', 'how', 'the' are language; 'function' is code; 'works' is unknown
        self.assertEqual(results[0], 'language')  # Explain
        self.assertEqual(results[1], 'language')  # how
        self.assertEqual(results[2], 'language')  # the
        self.assertEqual(results[3], 'code')      # function
        self.assertIsNone(results[4])             # works

    def test_mixed_code_documentation(self):
        """Test tokens from mixed code and documentation."""
        # "Show me a function that imports pandas"
        tokens = ['Show', 'me', 'a', 'function', 'that', 'import', 'pandas']
        results = [self.classifier.classify(t) for t in tokens]

        # Language words: Show, me, a, that
        # Code words: function, import, pandas
        self.assertEqual(results[0], 'language')  # Show
        self.assertEqual(results[1], 'language')  # me
        self.assertEqual(results[2], 'language')  # a
        self.assertEqual(results[3], 'code')      # function
        self.assertEqual(results[4], 'language')  # that
        self.assertEqual(results[5], 'code')      # import
        self.assertEqual(results[6], 'code')      # pandas


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_build_code_keyword_set(self):
        """build_code_keyword_set should return a set."""
        keyword_set = build_code_keyword_set()
        self.assertIsInstance(keyword_set, set)
        self.assertGreater(len(keyword_set), 0)

    def test_build_language_keyword_set(self):
        """build_language_keyword_set should return a set."""
        word_set = build_language_keyword_set()
        self.assertIsInstance(word_set, set)
        self.assertGreater(len(word_set), 0)

    def test_get_keyword_stats(self):
        """get_keyword_stats should return statistics."""
        stats = get_keyword_stats()

        # Check required keys
        required_keys = [
            'code_keywords', 'code_operators', 'language_words',
            'total_keywords', 'domain_specific'
        ]
        for key in required_keys:
            self.assertIn(key, stats)
            self.assertGreater(stats[key], 0)

    def test_keyword_stats_accuracy(self):
        """Keyword stats should be accurate."""
        stats = get_keyword_stats()

        # Verify counts match actual sets
        self.assertEqual(stats['code_keywords'], len(CODE_KEYWORDS))
        self.assertEqual(stats['code_operators'], len(CODE_OPERATORS))
        self.assertEqual(stats['language_words'], len(LANGUAGE_WORDS))

    def test_functional_interface(self):
        """Test module-level classify_token function."""
        # Should work same as class method
        self.assertEqual(classify_token('def'), 'code')
        self.assertEqual(classify_token('explain'), 'language')
        self.assertIsNone(classify_token('unknown_token'))


class TestWeek1POCIssues(unittest.TestCase):
    """
    Test that Week 1 POC issues are resolved.

    Week 1 found that library names like 'requests', 'pandas', 'Firebase'
    were classified as 'other' instead of 'code'. These tests verify the fix.
    """

    def setUp(self):
        """Create classifier instance for tests."""
        self.classifier = KeywordClassifier()

    def test_requests_library(self):
        """'requests' library should be classified as code."""
        self.assertEqual(self.classifier.classify('requests'), 'code')
        self.assertEqual(self.classifier.classify('get'), 'code')
        self.assertEqual(self.classifier.classify('post'), 'code')

    def test_pandas_library(self):
        """'pandas' library should be classified as code."""
        self.assertEqual(self.classifier.classify('pandas'), 'code')
        self.assertEqual(self.classifier.classify('pd'), 'code')
        self.assertEqual(self.classifier.classify('DataFrame'), 'code')
        self.assertEqual(self.classifier.classify('read_csv'), 'code')

    def test_firebase_library(self):
        """'Firebase' library should be classified as code."""
        self.assertEqual(self.classifier.classify('firebase'), 'code')
        self.assertEqual(self.classifier.classify('Firebase'), 'code')
        self.assertEqual(self.classifier.classify('auth'), 'code')
        self.assertEqual(self.classifier.classify('firestore'), 'code')

    def test_react_hooks(self):
        """React hooks should be classified as code."""
        self.assertEqual(self.classifier.classify('useState'), 'code')
        self.assertEqual(self.classifier.classify('useEffect'), 'code')
        self.assertEqual(self.classifier.classify('useContext'), 'code')

    def test_fastapi_framework(self):
        """FastAPI should be classified as code."""
        self.assertEqual(self.classifier.classify('FastAPI'), 'code')
        self.assertEqual(self.classifier.classify('Request'), 'code')
        self.assertEqual(self.classifier.classify('Response'), 'code')


if __name__ == '__main__':
    unittest.main()
