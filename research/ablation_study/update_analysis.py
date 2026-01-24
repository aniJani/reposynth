import json

with open('C:/Users/rajka/reposynth/research/ablation_study/Multi_Hop_Retrieval_Ablation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = '''# Cell 26.5: Real CCE Ablation Analysis (FULL CCE: H_code - H_lang)
print("\\n" + "="*70)
print("REAL CCE ABLATION ANALYSIS (Contrastive Code Entropy)")
print("="*70)

print("""
Key Insight: This analysis uses FULL Contrastive Code Entropy:
  CCE = H_code - H_lang

Where:
  H_code = entropy over CODE tokens only (numpy, pandas, def, class...)
  H_lang = entropy over LANGUAGE tokens only (the, what, how, is, are...)

CCE > 0: Model uncertain about CODE tokens -> RETRIEVE
CCE < 0: Model uncertain about LANGUAGE tokens -> DON'T retrieve

This ensures retrieval only triggers on CODE uncertainty, not formatting.

Ablation Variants:
  - cce_topk_only: Uses ONLY CODE tokens for retrieval
  - cce_query_plus_topk: Uses original query + CODE tokens

Research Question: Does including the original query improve retrieval,
or does the pure CODE uncertainty signal work better?
""")

# Statistical comparison: CCE variants
if 'cce_topk_only' in df_results['strategy'].values and 'cce_query_plus_topk' in df_results['strategy'].values:
    print("\\nStatistical Comparison: CCE Query Construction")
    print("-"*60)

    topk_only = df_results[df_results['strategy'] == 'cce_topk_only']
    query_plus = df_results[df_results['strategy'] == 'cce_query_plus_topk']

    # Check for skipped retrievals (no spike)
    topk_retrieved = topk_only[topk_only['num_hops'] > 0]
    query_retrieved = query_plus[query_plus['num_hops'] > 0]

    print(f"\\nSpike Detection Rate:")
    print(f"  cce_topk_only: {len(topk_retrieved)}/{len(topk_only)} examples triggered retrieval")
    print(f"  cce_query_plus_topk: {len(query_retrieved)}/{len(query_plus)} examples triggered retrieval")

    if len(topk_retrieved) > 0 and len(query_retrieved) > 0:
        print(f"\\nRetrieval Quality (when spike detected):")
        print(f"  cce_topk_only:      F1={topk_retrieved['file_f1'].mean():.3f}")
        print(f"  cce_query_plus_topk: F1={query_retrieved['file_f1'].mean():.3f}")

        # Statistical test
        if len(topk_retrieved) >= 3 and len(query_retrieved) >= 3:
            from scipy import stats
            # Use only examples where both had spikes
            common_ids = set(topk_retrieved['example_id']) & set(query_retrieved['example_id'])
            if len(common_ids) >= 3:
                topk_vals = topk_retrieved[topk_retrieved['example_id'].isin(common_ids)]['file_f1'].values
                query_vals = query_retrieved[query_retrieved['example_id'].isin(common_ids)]['file_f1'].values
                t_stat, p_val = stats.ttest_rel(topk_vals[:len(query_vals)], query_vals[:len(topk_vals)])
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"\\n  Statistical test (paired t-test): p={p_val:.4f}{sig}")

    # Analyze CODE tokens extracted
    print("\\n" + "-"*60)
    print("CODE Token Analysis (from H_code - H_lang)")
    print("-"*60)

    cce_traces = [r for r in all_results if r.strategy in ['cce_topk_only', 'cce_query_plus_topk']]
    all_code_tokens = []
    cce_values = []
    h_code_values = []
    h_lang_values = []

    for result in cce_traces:
        for hop_trace in result.trace:
            tokens = hop_trace.get('confused_tokens', [])
            cce = hop_trace.get('cce', 0)
            h_code = hop_trace.get('h_code', 0)
            h_lang = hop_trace.get('h_lang', 0)
            all_code_tokens.extend(tokens)
            if cce != 0:
                cce_values.append(cce)
                h_code_values.append(h_code)
                h_lang_values.append(h_lang)

    if all_code_tokens:
        from collections import Counter
        token_counts = Counter(all_code_tokens)
        print(f"\\nMost common CODE tokens extracted:")
        for token, count in token_counts.most_common(10):
            print(f"    '{token}': {count}")

    if cce_values:
        import numpy as np
        print(f"\\nCCE Statistics (H_code - H_lang):")
        print(f"  Mean CCE:    {np.mean(cce_values):.3f}")
        print(f"  Mean H_code: {np.mean(h_code_values):.2f}")
        print(f"  Mean H_lang: {np.mean(h_lang_values):.2f}")
        print(f"  Min CCE:     {np.min(cce_values):.3f}")
        print(f"  Max CCE:     {np.max(cce_values):.3f}")

else:
    print("\\nNote: Run the retrieval ablation with real CCE strategies first")

# Comparison with proactive strategies
print("\\n" + "="*70)
print("REACTIVE (CCE) vs PROACTIVE (Multi-Hop) COMPARISON")
print("="*70)

print("""
| Strategy | Type | When to Retrieve | Query Construction |
|----------|------|------------------|-------------------|
| embedding_only | Proactive | Always (upfront) | Full query |
| iterative_refinement | Proactive | Always (upfront) | Full query + refinement |
| cce_topk_only | Reactive | Only on CODE spike (CCE>0) | CODE tokens only |
| cce_query_plus_topk | Reactive | Only on CODE spike (CCE>0) | Query + CODE tokens |

Note: CCE = H_code - H_lang. Only triggers when model is uncertain about CODE tokens.
""")

# Compare reactive vs proactive
if 'cce_topk_only' in df_results['strategy'].values:
    reactive = df_results[df_results['strategy'].isin(['cce_topk_only', 'cce_query_plus_topk'])]
    proactive = df_results[df_results['strategy'].isin(['embedding_only', 'iterative_refinement'])]

    # Only count examples where retrieval happened
    reactive_retrieved = reactive[reactive['num_hops'] > 0]

    if len(reactive_retrieved) > 0 and len(proactive) > 0:
        print(f"\\nOverall Comparison:")
        print(f"  CCE (H_code-H_lang) avg F1: {reactive_retrieved['file_f1'].mean():.3f} (n={len(reactive_retrieved)})")
        print(f"  Proactive avg F1:           {proactive['file_f1'].mean():.3f} (n={len(proactive)})")

        reactive_tokens = reactive_retrieved['total_tokens'].mean()
        proactive_tokens = proactive['total_tokens'].mean()
        print(f"\\n  CCE avg tokens:        {reactive_tokens:.0f}")
        print(f"  Proactive avg tokens:  {proactive_tokens:.0f}")

        if reactive_tokens < proactive_tokens:
            savings = (1 - reactive_tokens/proactive_tokens) * 100
            print(f"\\n  Token savings (CCE vs proactive): {savings:.1f}%")'''

# Update cell 33
nb['cells'][33] = {
    'cell_type': 'code',
    'source': [line + '\n' for line in new_source.split('\n')],
    'metadata': {},
    'execution_count': None,
    'outputs': []
}
nb['cells'][33]['source'][-1] = nb['cells'][33]['source'][-1].rstrip('\n')

with open('C:/Users/rajka/reposynth/research/ablation_study/Multi_Hop_Retrieval_Ablation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Cell 33 updated with Real CCE ablation analysis')
