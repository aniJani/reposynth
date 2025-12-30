# Cell 10: Full Sentence Generation Testing

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# ============================================================================
# TEST CASES - FULL SENTENCE GENERATION
# ============================================================================
# NEW APPROACH: Generate full sentences and check if code uncertainty appears

# CODE test cases - we EXPECT the system to stop at least once for code uncertainty
# Example: "Authentication using" → "the" (continue) → "JWT" (STOP!)
CODE_TEST_CASES = [
    # Authentication & Security
    {'prompt': 'The authentication is done using', 'category': 'auth_method'},
    {'prompt': 'Passwords are hashed with', 'category': 'auth_hash'},
    {'prompt': 'JWT tokens are signed using', 'category': 'auth_signing'},
    {'prompt': 'The OAuth provider we use is', 'category': 'auth_provider'},
    {'prompt': 'Session data is stored in', 'category': 'session_store'},

    # Databases
    {'prompt': 'The database we use is', 'category': 'database_type'},
    {'prompt': 'We query the database using', 'category': 'database_query'},
    {'prompt': 'The ORM library is', 'category': 'database_orm'},
    {'prompt': 'Caching is implemented with', 'category': 'database_cache'},

    # Web Frameworks
    {'prompt': 'The API is built with', 'category': 'web_framework'},
    {'prompt': 'The web server runs on', 'category': 'web_server'},
    {'prompt': 'HTTP requests are made using', 'category': 'http_client'},
    {'prompt': 'Our GraphQL server uses', 'category': 'graphql_server'},

    # Frontend
    {'prompt': 'The frontend framework is', 'category': 'frontend_framework'},
    {'prompt': 'State management is handled by', 'category': 'frontend_state'},
    {'prompt': 'Components are built with', 'category': 'frontend_components'},
    {'prompt': 'Routing is done using', 'category': 'frontend_routing'},

    # ML/AI
    {'prompt': 'The model is trained with', 'category': 'ml_framework'},
    {'prompt': 'Deep learning is implemented using', 'category': 'ml_deep_learning'},
    {'prompt': 'The optimizer we use is', 'category': 'ml_optimizer'},

    # DevOps & Cloud
    {'prompt': 'We deploy to', 'category': 'cloud_platform'},
    {'prompt': 'Containers are orchestrated with', 'category': 'cloud_containers'},
    {'prompt': 'The CI/CD pipeline uses', 'category': 'cloud_cicd'},

    # Testing & Build
    {'prompt': 'Unit tests are written with', 'category': 'test_unit'},
    {'prompt': 'The bundler we use is', 'category': 'build_bundler'},
    {'prompt': 'Package management is done with', 'category': 'build_package_manager'},
]

# LANGUAGE test cases - we EXPECT the system to NEVER stop (only language uncertainty)
# Example: "The system is" → "very" (continue) → "robust" (continue) → "and" (continue) → ...
LANGUAGE_TEST_CASES = [
    # Pure descriptions (no code references)
    {'prompt': 'The weather today is', 'category': 'description_weather'},
    {'prompt': 'The meeting was', 'category': 'description_meeting'},
    {'prompt': 'My favorite color is', 'category': 'description_color'},
    {'prompt': 'The book I read was', 'category': 'description_book'},
    {'prompt': 'The movie seemed', 'category': 'description_movie'},

    # General explanations
    {'prompt': 'The main idea is to', 'category': 'explanation_idea'},
    {'prompt': 'The process works by', 'category': 'explanation_process'},
    {'prompt': 'This approach helps to', 'category': 'explanation_approach'},
    {'prompt': 'The benefit of this is', 'category': 'explanation_benefit'},

    # Instructions (generic)
    {'prompt': 'To solve this problem, you should', 'category': 'instruction_solve'},
    {'prompt': 'First, make sure to', 'category': 'instruction_first'},
    {'prompt': 'Before starting, you need to', 'category': 'instruction_before'},

    # Comparisons (generic)
    {'prompt': 'Unlike the previous approach, this', 'category': 'comparison_unlike'},
    {'prompt': 'The main difference is that it', 'category': 'comparison_difference'},

    # Questions (generic)
    {'prompt': 'The question is whether we', 'category': 'question_whether'},
    {'prompt': 'What matters most is', 'category': 'question_what'},
]

ALL_TEST_CASES = [
    {**case, 'expected_stopped': True} for case in CODE_TEST_CASES  # Should stop at least once
] + [
    {**case, 'expected_stopped': False} for case in LANGUAGE_TEST_CASES  # Should never stop
]

print(f"\n{'='*80}")
print(f"FULL SENTENCE GENERATION TESTING")
print(f"{'='*80}")
print(f"\nTotal test cases: {len(ALL_TEST_CASES)}")
print(f"  CODE tests (should stop): {len(CODE_TEST_CASES)}")
print(f"  LANGUAGE tests (should not stop): {len(LANGUAGE_TEST_CASES)}")
print(f"\nApproach: Generate full sentences and check if code uncertainty appears")

# ============================================================================
# RUN FULL SENTENCE GENERATION TESTS
# ============================================================================

print(f"\n{'='*80}")
print(f"RUNNING FULL SENTENCE GENERATION")
print(f"{'='*80}\n")

test_results = []

for test_case in tqdm(ALL_TEST_CASES, desc="Testing"):
    prompt = test_case['prompt']

    # Generate full sentence (up to period or max tokens)
    result = generate_with_contrastive_probe(
        prompt,
        entropy_threshold=3.0,
        top_k_candidates=10,
        max_tokens=20,  # Generate longer sequences
        verbose=False
    )

    # Check if it stopped for code uncertainty at ANY point during generation
    stopped_for_code = result['stop_reason'] == 'code_uncertainty'
    expected_stopped = test_case['expected_stopped']
    correct = stopped_for_code == expected_stopped

    # Get max entropy across all generation steps
    max_entropy = np.max(result['entropy_trace']) if result['entropy_trace'] else 0

    test_results.append({
        'prompt': prompt,
        'category': test_case['category'],
        'expected_stopped': expected_stopped,
        'stopped_for_code': stopped_for_code,
        'correct': correct,
        'stop_reason': result['stop_reason'],
        'max_entropy': max_entropy,
        'generated_text': result.get('generated_text', ''),
        'num_tokens_generated': len(result.get('generated_tokens', [])),
    })

print(f"\n✅ Testing complete on {len(test_results)} cases!")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================

df_results = pd.DataFrame(test_results)

print(f"\n{'='*80}")
print(f"TEST RESULTS - FULL SENTENCE GENERATION")
print(f"{'='*80}")

overall_accuracy = df_results['correct'].mean()
print(f"\n📊 OVERALL ACCURACY: {overall_accuracy:.1%}")
print(f"   (on {len(df_results)} test cases)")

print(f"\n📈 BY CLASS:")
for expected_val in [True, False]:
    class_name = "CODE (should stop)" if expected_val else "LANGUAGE (should not stop)"
    subset = df_results[df_results['expected_stopped'] == expected_val]
    accuracy = subset['correct'].mean() if len(subset) > 0 else 0
    correct = subset['correct'].sum()
    total = len(subset)
    stopped_count = subset['stopped_for_code'].sum()
    print(f"\n   {class_name}")
    print(f"      Correct: {correct}/{total}")
    print(f"      Accuracy: {accuracy:.1%}")
    print(f"      Stopped for code: {stopped_count}/{total}")
    if total > 0:
        print(f"      Avg entropy: {subset['max_entropy'].mean():.2f} bits")
        print(f"      Avg tokens generated: {subset['num_tokens_generated'].mean():.1f}")

# Confusion matrix
tp = len(df_results[(df_results['expected_stopped']==True) & (df_results['stopped_for_code']==True)])
fp = len(df_results[(df_results['expected_stopped']==False) & (df_results['stopped_for_code']==True)])
fn = len(df_results[(df_results['expected_stopped']==True) & (df_results['stopped_for_code']==False)])
tn = len(df_results[(df_results['expected_stopped']==False) & (df_results['stopped_for_code']==False)])

print(f"\n🎯 CONFUSION MATRIX")
print(f"                       Predicted NO STOP    Predicted STOPPED")
print(f"Expected NO STOP             {tn:<12}          {fp:<12}")
print(f"Expected STOP                {fn:<12}          {tp:<12}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📐 METRICS (CODE class - should stop)")
print(f"   Precision: {precision:.1%}")
print(f"   Recall: {recall:.1%}")
print(f"   F1-Score: {f1:.1%}")

# Category breakdown
print(f"\n📋 PERFORMANCE BY CATEGORY")
print(f"{'Category':<30} {'Correct':<10} {'Stopped':<10} {'Accuracy':<12} {'Avg Tokens'}")
print(f"{'-'*85}")
for category in sorted(df_results['category'].unique()):
    cat_data = df_results[df_results['category'] == category]
    correct = cat_data['correct'].sum()
    stopped = cat_data['stopped_for_code'].sum()
    total = len(cat_data)
    accuracy = correct / total if total > 0 else 0
    avg_tokens = cat_data['num_tokens_generated'].mean()
    print(f"{category:<30} {correct}/{total:<8} {stopped}/{total:<8} {accuracy:>6.1%}        {avg_tokens:>6.1f}")

# Errors with generated text
errors = df_results[~df_results['correct']]
if len(errors) > 0:
    print(f"\n❌ ERRORS ({len(errors)}/{len(df_results)}):")
    for idx, error in errors.iterrows():
        exp = "SHOULD STOP" if error['expected_stopped'] else "SHOULD NOT STOP"
        act = "STOPPED" if error['stopped_for_code'] else "DID NOT STOP"
        print(f"\n   Prompt: '{error['prompt']}'")
        print(f"      Expected: {exp}, Actual: {act}")
        print(f"      Generated: '{error['generated_text']}'")
        print(f"      Stop reason: {error['stop_reason']}, Max entropy: {error['max_entropy']:.2f}")
else:
    print(f"\n🎉 PERFECT! No errors!")

print(f"\n{'='*80}")
