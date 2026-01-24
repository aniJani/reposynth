import json

with open('C:/Users/rajka/reposynth/research/ablation_study/Multi_Hop_Retrieval_Ablation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = '''# Cell 23: Run generation comparison (FULL CCE: H_code - H_lang)
print("="*70)
print("GENERATION COMPARISON: Multi-Hop Retrieval + Full CCE (H_code - H_lang)")
print("="*70)

MAX_TOKENS = 150
generation_results = []

# Select a small subset for generation (expensive)
gen_examples = list(benchmark)[:6]  # Limit for memory

for i, example in enumerate(gen_examples):
    print(f"\\n[{i+1}/{len(gen_examples)}] {example.id}")
    print(f"  Query: {example.query[:50]}...")

    # Test embedding-only with generation
    try:
        retriever.reset()
        ret_result = multi_hop_retrievers['embedding_only'].retrieve(example.query)

        prompt = f"""Context:
{ret_result.retrieved_content[:2500]}

Question: {example.query}
Answer:"""

        start = time.time()
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_time = time.time() - start

        correctness = metrics.answer_correctness(answer, example.ground_truth_answer, example.ground_truth_keywords)
        print(f"  embedding_only: correctness={correctness:.2f}, time={gen_time:.1f}s")

        generation_results.append({
            'example_id': example.id,
            'strategy': 'embedding_only',
            'correctness': correctness,
            'time': gen_time,
            'hops': ret_result.num_hops,
        })

    except Exception as e:
        print(f"  embedding_only: ERROR - {str(e)[:40]}")

    # Test Real CCE: Top-K Only variant
    try:
        retriever.reset()
        ret_result = multi_hop_retrievers['cce_topk_only'].retrieve(example.query)

        if ret_result.num_hops == 0:
            print(f"  cce_topk_only: NO SPIKE - skipped retrieval")
            generation_results.append({
                'example_id': example.id,
                'strategy': 'cce_topk_only',
                'correctness': 0,
                'time': 0,
                'hops': 0,
                'skipped': True,
            })
        else:
            prompt = f"""Context:
{ret_result.retrieved_content[:2500]}

Question: {example.query}
Answer:"""

            start = time.time()
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            gen_time = time.time() - start

            correctness = metrics.answer_correctness(answer, example.ground_truth_answer, example.ground_truth_keywords)

            # Show CODE tokens and CCE value
            if ret_result.trace:
                tokens = ret_result.trace[0].get('confused_tokens', [])[:5]
                cce = ret_result.trace[0].get('cce', 0)
                print(f"  cce_topk_only: correctness={correctness:.2f}, CCE={cce:.2f}, code_tokens={tokens}")

            generation_results.append({
                'example_id': example.id,
                'strategy': 'cce_topk_only',
                'correctness': correctness,
                'time': gen_time,
                'hops': ret_result.num_hops,
            })

    except Exception as e:
        print(f"  cce_topk_only: ERROR - {str(e)[:40]}")

    # Test Real CCE: Query + Top-K variant
    try:
        retriever.reset()
        ret_result = multi_hop_retrievers['cce_query_plus_topk'].retrieve(example.query)

        if ret_result.num_hops == 0:
            print(f"  cce_query_plus_topk: NO SPIKE - skipped retrieval")
            generation_results.append({
                'example_id': example.id,
                'strategy': 'cce_query_plus_topk',
                'correctness': 0,
                'time': 0,
                'hops': 0,
                'skipped': True,
            })
        else:
            prompt = f"""Context:
{ret_result.retrieved_content[:2500]}

Question: {example.query}
Answer:"""

            start = time.time()
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            gen_time = time.time() - start

            correctness = metrics.answer_correctness(answer, example.ground_truth_answer, example.ground_truth_keywords)

            if ret_result.trace:
                tokens = ret_result.trace[0].get('confused_tokens', [])[:5]
                cce = ret_result.trace[0].get('cce', 0)
                print(f"  cce_query_plus_topk: correctness={correctness:.2f}, CCE={cce:.2f}, code_tokens={tokens}")

            generation_results.append({
                'example_id': example.id,
                'strategy': 'cce_query_plus_topk',
                'correctness': correctness,
                'time': gen_time,
                'hops': ret_result.num_hops,
            })

    except Exception as e:
        print(f"  cce_query_plus_topk: ERROR - {str(e)[:40]}")

    # Test CCE adaptive (original Week 8 approach)
    try:
        retriever.reset()
        start = time.time()
        gen_result = adaptive_gen.generate(query=example.query, initial_context=file_list_context)
        gen_time = time.time() - start

        correctness = metrics.answer_correctness(gen_result.response, example.ground_truth_answer, example.ground_truth_keywords)
        print(f"  cce_adaptive: correctness={correctness:.2f}, time={gen_time:.1f}s, retrievals={gen_result.total_retrievals}")

        generation_results.append({
            'example_id': example.id,
            'strategy': 'cce_adaptive',
            'correctness': correctness,
            'time': gen_time,
            'hops': gen_result.total_retrievals,
        })

    except Exception as e:
        print(f"  cce_adaptive: ERROR - {str(e)[:40]}")

    # Clear CUDA cache
    torch.cuda.empty_cache()

print("\\nGeneration comparison complete!")'''

# Update cell 28
nb['cells'][28] = {
    'cell_type': 'code',
    'source': [line + '\n' for line in new_source.split('\n')],
    'metadata': {},
    'execution_count': None,
    'outputs': []
}
nb['cells'][28]['source'][-1] = nb['cells'][28]['source'][-1].rstrip('\n')

with open('C:/Users/rajka/reposynth/research/ablation_study/Multi_Hop_Retrieval_Ablation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Cell 28 updated with Real CCE generation comparison')
