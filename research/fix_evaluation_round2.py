"""
Fix Week8_Comprehensive_Evaluation.ipynb - Round 2

Issues to fix:
1. Token counting wrong - count actual prompt tokens + gen_len
2. Remove [:2500] character truncation - use full retrieved content
3. Export config wrong - fix to match actual values
4. Remove misleading "FIXED" comments
5. Add bootstrap CI for statistics
"""

import json

def fix_notebook():
    print("Fixing Week8_Comprehensive_Evaluation.ipynb - Round 2")
    print("=" * 60)

    with open('Week8_Comprehensive_Evaluation.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changes_made = []

    # Process all cells
    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source_lines = cell['source']
        modified = False

        for i, line in enumerate(source_lines):
            # Fix 1: Remove [:2500] truncation
            if 'ret_result.retrieved_content[:2500]' in line:
                source_lines[i] = line.replace('ret_result.retrieved_content[:2500]', 'ret_result.retrieved_content')
                modified = True
                changes_made.append(f"Cell {cell_idx}: Removed [:2500] truncation")

            # Fix 2: Fix token counting - the broken one
            if 'tokens_used = file_list_tokens + ret_result.total_tokens + gen_len' in line:
                source_lines[i] = line.replace(
                    'tokens_used = file_list_tokens + ret_result.total_tokens + gen_len',
                    'tokens_used = len(tokenizer.encode(prompt)) + gen_len  # Actual prompt tokens'
                )
                modified = True
                changes_made.append(f"Cell {cell_idx}: Fixed token counting (with context)")

            # Fix 3: Fix fallback token counting
            if 'tokens_used = file_list_tokens + gen_len' in line:
                source_lines[i] = line.replace(
                    'tokens_used = file_list_tokens + gen_len',
                    'tokens_used = len(tokenizer.encode(fallback_prompt)) + gen_len  # Actual prompt tokens'
                )
                modified = True
                changes_made.append(f"Cell {cell_idx}: Fixed fallback token counting")

            # Fix 4: Remove misleading FIXED comments
            if '# FIXED:' in line:
                source_lines[i] = line.replace('# FIXED:', '#')
                modified = True
                changes_made.append(f"Cell {cell_idx}: Removed 'FIXED' comment")

            # Fix 5: Export config - uncertainty_threshold
            if "'uncertainty_threshold': 3.0" in line:
                source_lines[i] = line.replace("'uncertainty_threshold': 3.0", "'uncertainty_threshold': 0.5")
                modified = True
                changes_made.append(f"Cell {cell_idx}: Fixed uncertainty_threshold to 0.5")

            # Fix 6: Export config - max_gen_tokens
            if "'real_cce_max_gen_tokens': 50" in line:
                source_lines[i] = line.replace("'real_cce_max_gen_tokens': 50", "'real_cce_max_gen_tokens': 200")
                modified = True
                changes_made.append(f"Cell {cell_idx}: Fixed real_cce_max_gen_tokens to 200")

        if modified:
            cell['source'] = source_lines

    # Remove the unused prompt_tokens and count_tokens_unified
    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell['source'])

        # Remove unused prompt_tokens calculation
        if 'prompt_tokens = len(tokenizer.encode(f"Question: {example.query}' in source:
            new_lines = []
            skip_next = False
            for line in cell['source']:
                if 'prompt_tokens = len(tokenizer.encode' in line:
                    changes_made.append(f"Cell {cell_idx}: Removed unused prompt_tokens line")
                    continue
                # Also skip the comment line before it
                if '# Unified token counting' in line.lower():
                    continue
                new_lines.append(line)
            cell['source'] = new_lines

        # Remove unused count_tokens_unified function
        if 'def count_tokens_unified' in source:
            new_lines = []
            skip_count = 0
            for line in cell['source']:
                if 'Helper function: unified token counting' in line:
                    skip_count = 4  # Skip this and the next 3 lines
                if skip_count > 0:
                    skip_count -= 1
                    continue
                if 'def count_tokens_unified' in line:
                    skip_count = 3  # Skip function definition
                    continue
                new_lines.append(line)
            cell['source'] = new_lines
            changes_made.append(f"Cell {cell_idx}: Removed unused count_tokens_unified function")

    # Add bootstrap CI cell after the comparison table
    bootstrap_code = '''# Bootstrap confidence intervals for small-N statistics
import numpy as np

def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """Compute bootstrapped confidence interval."""
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN
    if len(data) < 2:
        mean = np.mean(data) if len(data) > 0 else 0
        return mean, mean, mean
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(data), lower, upper

# Report with bootstrap CIs instead of t-tests
print("\\nResults with 95% Bootstrap Confidence Intervals:")
print("=" * 60)
methods = df_results['method'].unique()
for metric in ['answer_similarity', 'keyword_recall', 'file_recall']:
    print(f"\\n{metric}:")
    for method in methods:
        scores = df_results[df_results['method'] == method][metric].values
        mean, ci_low, ci_high = bootstrap_ci(scores)
        print(f"  {method}: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
'''

    # Check if bootstrap_ci already exists
    bootstrap_exists = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def bootstrap_ci' in source:
                bootstrap_exists = True
                break

    if not bootstrap_exists:
        # Find comparison table cell
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'df_comparison' in source and 'groupby' in source:
                    new_cell = {
                        'cell_type': 'code',
                        'execution_count': None,
                        'metadata': {},
                        'outputs': [],
                        'source': [line + '\n' for line in bootstrap_code.split('\n')]
                    }
                    nb['cells'].insert(i + 1, new_cell)
                    changes_made.append(f"Added bootstrap CI cell after index {i}")
                    break

    # Save
    with open('Week8_Comprehensive_Evaluation.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("\nChanges made:")
    for change in changes_made:
        print(f"  - {change}")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("  1. Removed [:2500] character truncation")
    print("  2. Fixed token counting to use actual prompt tokens")
    print("  3. Fixed export config (0.5 threshold, 200 gen tokens)")
    print("  4. Removed misleading 'FIXED' comments")
    print("  5. Added bootstrap CI function")

if __name__ == '__main__':
    fix_notebook()
