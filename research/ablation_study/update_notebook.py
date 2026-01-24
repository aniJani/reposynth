import json

# Read the notebook
with open('C:/Users/rajka/reposynth/research/ablation_study/Multi_Hop_Retrieval_Ablation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# New cell source for RealCCEMultiHopRetriever
new_source = '''# Cell 14.5: Real CCE Multi-Hop (No Simulation - Actual Entropy Detection)

class RealCCEMultiHopRetriever(MultiHopRetriever):
    """
    REAL CCE Multi-Hop: Actual entropy-based confused token extraction.

    NO SIMULATION - uses real model generation and logits.

    Key differences from simulated version:
    - Confused tokens come from torch.argsort(logits)[-k:] NOT regex
    - Actually runs generation to detect entropy spikes
    - Skips retrieval if no spike detected (model not confused)
    - Uses embedding similarity for refinement (no regex)
    """

    def __init__(self, base_retriever, tokenizer, model,
                 top_k: int = 2, max_hops: int = 2,
                 uncertainty_threshold: float = 3.0,
                 top_k_tokens: int = 10,
                 max_gen_tokens: int = 50):
        super().__init__(base_retriever, tokenizer, "real_cce_multi_hop")
        self.model = model
        self.top_k = top_k
        self.max_hops = max_hops
        self.uncertainty_threshold = uncertainty_threshold
        self.top_k_tokens = top_k_tokens
        self.max_gen_tokens = max_gen_tokens

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute Shannon entropy from logits."""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log2(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy.item()

    def _extract_top_tokens_from_logits(self, logits: torch.Tensor) -> List[str]:
        """
        REAL confused token extraction - from model's logits.
        These are the tokens the model is uncertain between.
        """
        top_indices = torch.argsort(logits, descending=True)[:self.top_k_tokens]

        tokens = []
        for idx in top_indices:
            token = self.tokenizer.decode([idx.item()])
            token = token.strip()
            if len(token) > 1 and not token.startswith('<') and not token.startswith('\\n'):
                tokens.append(token)

        return tokens[:self.top_k_tokens]

    def _generate_until_spike(self, query: str) -> Tuple[List[str], float, int]:
        """
        Generate tokens until entropy spike detected.
        Returns: (confused_tokens, spike_entropy, spike_position)
        """
        prompt = f"Question: {query}\\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

        generated_ids = inputs['input_ids']

        for i in range(self.max_gen_tokens):
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs.logits[0, -1, :]

            entropy = self._compute_entropy(logits)

            if entropy > self.uncertainty_threshold:
                confused_tokens = self._extract_top_tokens_from_logits(logits)
                return confused_tokens, entropy, i

            next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        return [], 0.0, -1

    def _extract_refinement_terms_by_embedding(
        self, content: str, query: str, confused_tokens: List[str]
    ) -> List[str]:
        """
        Extract refinement terms using EMBEDDING SIMILARITY (no regex patterns).
        """
        lines = [l.strip() for l in content.split('\\n') if l.strip() and len(l.strip()) > 10]

        if not lines or len(lines) < 2:
            return []

        line_embeddings = self.retriever.model.encode(lines)
        query_with_tokens = f"{query} {' '.join(confused_tokens)}"
        query_embedding = self.retriever.model.encode([query_with_tokens])[0]

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], line_embeddings)[0]

        top_indices = similarities.argsort()[-3:][::-1]
        relevant_lines = [lines[i] for i in top_indices]

        terms = []
        for line in relevant_lines:
            words = re.split(r'[^a-zA-Z0-9_]+', line)
            for word in words:
                if len(word) > 2 and word.lower() not in query.lower():
                    if word not in confused_tokens and word not in terms:
                        terms.append(word)

        return terms[:5]

    def retrieve(self, query: str) -> MultiHopRetrievalResult:
        """
        Real CCE multi-hop retrieval:
        1. Generate until entropy spike
        2. Extract confused tokens from logits
        3. Multi-hop retrieval using confused tokens + query
        """
        all_files = []
        all_content = []
        all_scores = []
        trace = []
        seen_files = set()

        # PHASE 1: Generate until spike, extract REAL confused tokens
        confused_tokens, spike_entropy, spike_pos = self._generate_until_spike(query)

        if not confused_tokens:
            # No spike = model NOT confused = NO retrieval needed
            return MultiHopRetrievalResult(
                retrieved_files=[],
                retrieved_content="",
                scores=[],
                num_hops=0,
                total_tokens=0,
                trace=[{
                    'hop': 0,
                    'method': 'no_spike_detected',
                    'action': 'skip_retrieval',
                    'reason': 'Model not confused - no retrieval needed',
                    'max_tokens_checked': self.max_gen_tokens,
                }]
            )

        # HOP 1: Embedding retrieval using query + confused tokens
        combined_query = f"{query} {' '.join(confused_tokens)}"
        results = self.retriever.retrieve(combined_query, top_k=self.top_k, deduplicate=False)

        for r in results:
            if r['source'] not in seen_files:
                seen_files.add(r['source'])
                all_files.append(r['source'])
                all_content.append(r['content'])
                all_scores.append(r['score'])

        trace.append({
            'hop': 1,
            'method': 'cce_embedding',
            'spike_entropy': spike_entropy,
            'spike_position': spike_pos,
            'confused_tokens': confused_tokens,
            'combined_query': combined_query,
            'files': [r['source'] for r in results],
        })

        # PHASE 2: Multi-hop refinement using embedding similarity
        for hop in range(1, self.max_hops):
            if not all_content:
                break

            combined_content = "\\n".join(all_content)
            refinement_terms = self._extract_refinement_terms_by_embedding(
                combined_content, query, confused_tokens
            )

            if not refinement_terms:
                break

            refined_query = f"{query} {' '.join(confused_tokens)} {' '.join(refinement_terms)}"
            results = self.retriever.retrieve(refined_query, top_k=self.top_k, deduplicate=False)
            new_results = [r for r in results if r['source'] not in seen_files]

            if not new_results:
                break

            for r in new_results:
                seen_files.add(r['source'])
                all_files.append(r['source'])
                all_content.append(r['content'])
                all_scores.append(r['score'])

            trace.append({
                'hop': hop + 1,
                'method': 'embedding_refinement',
                'refinement_terms': refinement_terms,
                'refined_query': refined_query,
                'new_files': [r['source'] for r in new_results],
            })

        content = "\\n\\n".join(all_content)
        return MultiHopRetrievalResult(
            retrieved_files=all_files,
            retrieved_content=content,
            scores=all_scores,
            num_hops=len(trace),
            total_tokens=self._count_tokens(content),
            trace=trace
        )


# Ablation variant: Top-K Only (no original query)
class CCETopKOnlyRetriever(RealCCEMultiHopRetriever):
    """Use ONLY confused tokens for retrieval query (no original query)."""

    def retrieve(self, query: str) -> MultiHopRetrievalResult:
        all_files = []
        all_content = []
        all_scores = []
        trace = []
        seen_files = set()

        confused_tokens, spike_entropy, spike_pos = self._generate_until_spike(query)

        if not confused_tokens:
            return MultiHopRetrievalResult(
                retrieved_files=[], retrieved_content="", scores=[],
                num_hops=0, total_tokens=0,
                trace=[{'hop': 0, 'method': 'no_spike_detected', 'action': 'skip_retrieval'}]
            )

        # TOP-K ONLY: Use only confused tokens, not original query
        topk_query = ' '.join(confused_tokens)
        results = self.retriever.retrieve(topk_query, top_k=self.top_k, deduplicate=False)

        for r in results:
            if r['source'] not in seen_files:
                seen_files.add(r['source'])
                all_files.append(r['source'])
                all_content.append(r['content'])
                all_scores.append(r['score'])

        trace.append({
            'hop': 1,
            'method': 'cce_topk_only',
            'spike_entropy': spike_entropy,
            'spike_position': spike_pos,
            'confused_tokens': confused_tokens,
            'query_used': topk_query,
            'files': [r['source'] for r in results],
        })

        # Multi-hop refinement
        for hop in range(1, self.max_hops):
            if not all_content:
                break
            combined_content = "\\n".join(all_content)
            refinement_terms = self._extract_refinement_terms_by_embedding(
                combined_content, query, confused_tokens
            )
            if not refinement_terms:
                break
            # Still use only top-k + refinement, no original query
            refined_query = f"{' '.join(confused_tokens)} {' '.join(refinement_terms)}"
            results = self.retriever.retrieve(refined_query, top_k=self.top_k, deduplicate=False)
            new_results = [r for r in results if r['source'] not in seen_files]
            if not new_results:
                break
            for r in new_results:
                seen_files.add(r['source'])
                all_files.append(r['source'])
                all_content.append(r['content'])
                all_scores.append(r['score'])
            trace.append({
                'hop': hop + 1, 'method': 'embedding_refinement',
                'refinement_terms': refinement_terms, 'new_files': [r['source'] for r in new_results],
            })

        content = "\\n\\n".join(all_content)
        return MultiHopRetrievalResult(
            retrieved_files=all_files, retrieved_content=content, scores=all_scores,
            num_hops=len(trace), total_tokens=self._count_tokens(content), trace=trace
        )


# Ablation variant: Query + Top-K (same as base, but explicit class)
class CCEQueryPlusTopKRetriever(RealCCEMultiHopRetriever):
    """Combine original query + confused tokens (explicit variant)."""
    pass  # Base class already does query + top-k


print("Real CCE Multi-Hop Retrieval defined (NO SIMULATION)")
print("  - Confused tokens: extracted from model logits via torch.argsort()")
print("  - Entropy detection: actual computation during generation")
print("  - No spike: skip retrieval entirely")
print("  - Refinement: embedding similarity (no regex)")
print()
print("Ablation variants:")
print("  - CCETopKOnlyRetriever: uses only confused tokens")
print("  - CCEQueryPlusTopKRetriever: uses query + confused tokens")'''

# Replace cell 17
nb['cells'][17] = {
    'cell_type': 'code',
    'source': [line + '\n' for line in new_source.split('\n')],
    'metadata': {},
    'execution_count': None,
    'outputs': []
}

# Fix last line (no trailing newline)
if nb['cells'][17]['source']:
    nb['cells'][17]['source'][-1] = nb['cells'][17]['source'][-1].rstrip('\n')

# Save the notebook
with open('C:/Users/rajka/reposynth/research/ablation_study/Multi_Hop_Retrieval_Ablation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Cell 17 replaced with RealCCEMultiHopRetriever + ablation variants')
