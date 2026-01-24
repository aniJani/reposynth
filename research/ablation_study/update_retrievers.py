import json

with open('C:/Users/rajka/reposynth/research/ablation_study/Multi_Hop_Retrieval_Ablation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = '''# Cell 15: Initialize all retrievers (including Real CCE variants)

multi_hop_retrievers = {
    'bm25': BM25Retriever(documents, tokenizer, top_k=3),  # Keyword baseline
    'embedding_only': EmbeddingOnlyRetriever(retriever, tokenizer, top_k=3),
    'multi_hop_coverage': MultiHopCoverageRetriever(retriever, tokenizer, top_k=2, max_hops=3),
    'cot_retrieval': ChainOfThoughtRetriever(retriever, tokenizer, top_k_per_hop=2),
    'iterative_refinement': IterativeRefinementRetriever(retriever, tokenizer, top_k=2, max_iterations=2),

    # Real CCE Multi-Hop variants (NO SIMULATION - actual entropy detection)
    'cce_topk_only': CCETopKOnlyRetriever(
        retriever, tokenizer, model,
        top_k=2, max_hops=2, uncertainty_threshold=3.0, top_k_tokens=10
    ),
    'cce_query_plus_topk': CCEQueryPlusTopKRetriever(
        retriever, tokenizer, model,
        top_k=2, max_hops=2, uncertainty_threshold=3.0, top_k_tokens=10
    ),
}

print("Initialized multi-hop retrievers:")
for name, r in multi_hop_retrievers.items():
    print(f"  - {name}: {r.__class__.__name__}")

print()
print("Real CCE variants for ablation study:")
print("  - cce_topk_only: retrieves using ONLY confused tokens from logits")
print("  - cce_query_plus_topk: retrieves using query + confused tokens")'''

# Update cell 18
nb['cells'][18] = {
    'cell_type': 'code',
    'source': [line + '\n' for line in new_source.split('\n')],
    'metadata': {},
    'execution_count': None,
    'outputs': []
}
nb['cells'][18]['source'][-1] = nb['cells'][18]['source'][-1].rstrip('\n')

with open('C:/Users/rajka/reposynth/research/ablation_study/Multi_Hop_Retrieval_Ablation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Cell 18 updated with Real CCE retrievers')
