"""Add bootstrap CI cell to notebook"""
import json

def add_bootstrap_ci():
    with open('Week8_Comprehensive_Evaluation.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Check if already exists
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def bootstrap_ci' in source:
                print("Bootstrap CI already exists")
                return

    bootstrap_code = '''# Bootstrap confidence intervals for small-N statistics
import numpy as np

def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """Compute bootstrapped confidence interval."""
    data = np.array(data)
    data = data[~np.isnan(data)]
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

# Report with 95% Bootstrap CIs
print("\\n" + "=" * 70)
print("RESULTS WITH 95% BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 70)

for metric_name, metric_attr in [
    ('Answer Similarity', 'answer_similarity'),
    ('Keyword Recall', 'keyword_recall'),
    ('File Recall', 'file_recall'),
    ('Composite Score', 'get_composite_score')
]:
    print(f"\\n{metric_name}:")
    for method, results in all_results.items():
        if not results:
            continue
        if metric_attr == 'get_composite_score':
            scores = [r.get_composite_score() for r in results]
        else:
            scores = [getattr(r, metric_attr) for r in results]
        mean, ci_low, ci_high = bootstrap_ci(scores)
        print(f"  {method:25s}: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
'''

    # Find where to insert - after the main experiment loop (after all_results is populated)
    insert_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            # Look for the comparison table creation
            if 'df_comparison = pd.DataFrame' in source:
                insert_idx = i
                break

    if insert_idx is None:
        print("Could not find insertion point")
        return

    new_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in bootstrap_code.split('\n')]
    }

    nb['cells'].insert(insert_idx, new_cell)
    print(f"Added bootstrap CI cell at index {insert_idx}")

    with open('Week8_Comprehensive_Evaluation.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print("Saved notebook")

if __name__ == '__main__':
    add_bootstrap_ci()
