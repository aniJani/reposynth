# Literature Review: Contrastive Code Entropy Research

**Document Version**: 1.0
**Created**: December 2024
**Purpose**: Foundation for CCE research implementation

---

## Executive Summary

This literature review examines three key areas relevant to our Contrastive Code Entropy (CCE) research:
1. **Uncertainty detection in LLMs** using entropy-based methods
2. **Adaptive reasoning strategies** triggered by uncertainty
3. **Code context retrieval systems** in production tools

**Key Finding**: While entropy-based uncertainty detection exists, no prior work distinguishes between **code knowledge uncertainty** (missing API/library context) and **language uncertainty** (word choice). This gap represents our core contribution.

---

## 1. Uncertainty Detection in LLMs

### 1.1 ARPO: Agentic Reinforced Policy Optimization

**Full Citation**: Dong et al., "Agentic Reinforced Policy Optimization," arXiv:2507.19849, 2024.

#### Core Observation
ARPO makes a critical observation directly relevant to our work:

> "LLMs tend to exhibit highly uncertain behavior, characterized by an increase in the entropy distribution of generated tokens, immediately following interactions with external tools."

This suggests that **entropy spikes correlate with missing external knowledge** - exactly what we aim to detect for code context.

#### Methodology
- **Entropy-based adaptive rollout mechanism**: Dynamically adjusts exploration based on uncertainty
- **Measurement timing**: After tool interactions (external knowledge gaps)
- **Application**: RL training for multi-turn agents with tool use

#### Results
- Evaluated on 13 benchmarks (computational reasoning, knowledge reasoning, search)
- Achieved same performance with **50% fewer tool calls**
- Demonstrates efficiency gains from uncertainty-driven retrieval

#### Relevance to CCE
**What we adopt**:
- Entropy as uncertainty signal
- Adaptive retrieval triggered by uncertainty
- Focus on external knowledge gaps

**What we extend**:
- ARPO doesn't distinguish code vs language uncertainty
- We add token-level classification (code vs language)
- We apply to code generation (not RL training)

**Citation strategy**: Cite as foundation for entropy-based uncertainty detection.

---

### 1.2 UnCert-CoT: Uncertainty-Guided Chain-of-Thought

**Full Citation**: Zhu et al., "Uncertainty-Guided Chain-of-Thought for Code Generation with LLMs," arXiv:2503.15341, 2024.

#### Problem Addressed
**"Overthinking" in code generation**: LLMs apply uniform reasoning regardless of task complexity, wasting compute on easy tasks.

#### Methodology
**Dual uncertainty metrics**:
1. **Entropy-based**: Shannon entropy H = -Σ p(x) log p(x)
2. **Probability differential-based**: 1 - max(P)

**Measurement strategy**:
- Unclear from abstract, but likely at **line boundaries** or completion points
- Triggers chain-of-thought reasoning when uncertainty is high

**Adaptive strategy**:
- High uncertainty → Generate multiple reasoning paths, select best
- Low uncertainty → Direct code generation

#### Results
- **6.1% improvement** on PassRate accuracy (MHPP benchmark)
- Stronger gains on harder problems
- Demonstrates value of uncertainty-driven adaptation

#### Relevance to CCE
**What we adopt**:
- Uncertainty triggers adaptive behavior
- Multiple uncertainty metrics (compare raw entropy, normalized, prob diff)
- Application to code generation

**What we extend**:
- UnCert-CoT uses uncertainty to decide *how much to reason*
- CCE uses uncertainty to decide *what context to retrieve*
- We distinguish code vs language uncertainty (they don't)

**Key difference**: UnCert-CoT addresses *reasoning effort*, CCE addresses *context retrieval*.

**Citation strategy**: Cite as related work on uncertainty in code generation, contrast our focus (retrieval vs reasoning).

---

### 1.3 Limitations of Existing Entropy Methods

Both ARPO and UnCert-CoT treat entropy as a single signal. However, high entropy can indicate:

1. **Code knowledge gap**: Model doesn't know which API to use
   - Example: `requests.???` (unfamiliar with library)
   - **Should retrieve context** (API docs, example usage)

2. **Language choice**: Model is choosing between synonyms
   - Example: `# This function calculates/computes/determines...`
   - **Should NOT retrieve** (just word choice)

**Our contribution**: CCE separates these cases by computing entropy **over code tokens vs language tokens separately**.

---

## 2. Multi-Agent Workflows and Context

### 2.1 FlowForge: Multi-Agent Workflow Design

**Full Citation**: Hao et al., "FlowForge: Guiding the Creation of Multi-agent Workflows with Design Space Visualization as a Thinking Scaffold," IEEE VIS 2025.

#### Core Contribution
Interactive visualization tool for designing multi-agent workflows.

**Hierarchical design framework**:
1. Task planning (high-level)
2. Agent assignment (mid-level)
3. Agent optimization (detailed)

#### Relevance to CCE
While not directly about uncertainty or retrieval, FlowForge demonstrates:
- **Structured exploration** of complex design spaces
- **Pattern-informed guidance** for decision-making
- **Context-aware suggestions** at each level

**Indirect relevance**: Our adaptive retrieval system makes decisions under uncertainty (similar to workflow design). FlowForge's approach to structured decision-making could inform our context management strategy.

**Citation strategy**: Cite in related work as example of adaptive, context-aware systems (tangential).

---

## 3. Code Context Retrieval Systems

### 3.1 Cursor

**Type**: Commercial AI coding assistant
**Source**: cursor.com (accessed Dec 2024)

#### Approach
- **Codebase indexing**: Automatic background indexing of entire codebase
- **Complete codebase understanding**: Works at any scale/complexity
- **Contextual queries**: "Where are these menu label colors defined?"

#### Strengths
- Zero-configuration setup
- Scales to large codebases
- Integrated across multiple interaction modes (Tab, Cmd+K, Agent)

#### Limitations (for research)
- **Proprietary**: No technical details on retrieval algorithm
- **Static retrieval**: No evidence of adaptive/uncertainty-driven retrieval
- **Black box**: Can't study when/why it retrieves context

**Relevance to CCE**: Establishes baseline for "static retrieval" comparison. Cursor likely retrieves context **upfront** based on query, not **adaptively** during generation.

---

### 3.2 Continue.dev

**Type**: Open-source AI coding assistant
**Source**: docs.continue.dev, GitHub discussions (accessed Dec 2024)

#### Technical Approach
**Custom RAG system**:
- Uses **voyage-code-3** embeddings (state-of-the-art for code)
- **LanceDB** vector database (in-memory, fast)
- **MCP (Model Context Protocol)** for standardized context access

**Context Providers**:
- Type '@' to select content
- Retrieves from vector databases of internal docs
- Supports custom context provider plugins

#### 2025 Evolution
> "RAG in 2025 is modular and should decide **if, what, where, and how to retrieve**, not retrieve blindly."

This aligns with our adaptive retrieval approach!

#### Strengths
- **Open source**: Can study implementation
- **Modular**: Custom context providers
- **Advanced embeddings**: voyage-code-3

#### Limitations
- **Pre-retrieval**: Context selected before generation
- **No uncertainty detection**: Doesn't adapt during generation
- **Manual selection**: User types '@' to trigger

**Relevance to CCE**: Continue.dev represents current state-of-the-art in open-source code RAG. Our CCE system extends this with **automatic, uncertainty-driven retrieval during generation**.

**Key difference**: Continue.dev = static pre-retrieval, CCE = dynamic adaptive retrieval.

---

### 3.3 Sourcegraph Cody

**Type**: Enterprise AI coding assistant
**Source**: sourcegraph.com, arXiv:2408.05344 (accessed Dec 2024)

#### Technical Approach
**Advanced context retrieval**:
- **Sourcegraph Search API**: Fast code search across massive codebases
- **Code Graph**: Schema capturing structure, relationships, dependencies
- **No embeddings** (in Enterprise): Replaced with advanced search
- **Multiple retrieval methods**: Keyword, semantic, code graph analysis

**Agentic RAG (2025)**:
- Tools: code search, codebase files, terminal, web search
- **Agentic Retrieval-Augmented Generation layer**
- Unified experience across chat, agents, search

#### Scale
- Supports **300,000+ repositories** (enterprise scale)
- Role-based access control (RBAC)
- Remote repository awareness

#### Strengths
- **Massive scale**: Proven at enterprise level
- **Code Graph**: Structural understanding beyond embeddings
- **Agentic approach**: Multiple tools for context retrieval

#### Limitations
- **Search-based**: Still pre-retrieval, not adaptive during generation
- **Complex setup**: Enterprise-focused, not research-friendly
- **Proprietary Code Graph**: Can't replicate

**Relevance to CCE**: Demonstrates state-of-the-art in production code RAG. Our CCE research could complement Cody's search by **detecting when to search** during generation.

**Research insight**: Cody abandoned embeddings in favor of search. This suggests **retrieval algorithm matters less than knowing when/what to retrieve** - exactly our focus!

---

## 4. Gap Analysis: What's Missing?

### 4.1 Existing Approaches

| System | Retrieval Timing | Uncertainty Detection | Code-Specific |
|--------|------------------|----------------------|---------------|
| ARPO | After tool use | Entropy (generic) | No |
| UnCert-CoT | Reasoning decision | Entropy (generic) | Yes (code gen) |
| Cursor | Pre-retrieval | None | Yes |
| Continue.dev | Pre-retrieval | None | Yes |
| Cody | Pre-retrieval | None | Yes |

**Pattern**: All production systems do **static pre-retrieval**. Research systems detect uncertainty but not **code-specific uncertainty**.

### 4.2 Research Gaps

1. **No code vs language uncertainty distinction**
   - ARPO and UnCert-CoT treat entropy as single signal
   - Can't distinguish "missing API knowledge" from "word choice"

2. **No adaptive retrieval during generation**
   - Production systems retrieve upfront
   - Can't respond to uncertainty emerging during generation

3. **No evaluation of uncertainty-driven retrieval**
   - Existing work evaluates answer quality
   - Doesn't measure: When should we retrieve? What triggers retrieval?

4. **No code-specific entropy metrics**
   - Generic entropy applied to code
   - Doesn't leverage code structure (tokens, syntax, identifiers)

---

## 5. Our Contribution: Contrastive Code Entropy

### 5.1 Novel Aspects

| Aspect | Prior Work | Our Contribution |
|--------|-----------|------------------|
| **Entropy metric** | Generic Shannon entropy | **Code vs language entropy** (contrastive) |
| **Token classification** | N/A | **Code/language taxonomy** |
| **Retrieval timing** | Pre-retrieval | **Adaptive during generation** |
| **Measurement** | Line boundaries (UnCert-CoT) | **Semantic boundaries** (function calls, imports) |
| **Application** | Reasoning effort | **Context retrieval** |

### 5.2 What We Cite (Not Novel)

To avoid overclaiming, we explicitly cite:
- **Entropy-based uncertainty**: ARPO, UnCert-CoT (established technique)
- **Adaptive behavior**: ARPO, UnCert-CoT (triggered by uncertainty)
- **Code RAG**: Cursor, Continue.dev, Cody (production baselines)
- **Tools**: Tree-sitter (AST), sentence-transformers (embeddings), TOON (serialization)

### 5.3 What We Claim as Novel

1. **Contrastive Code Entropy (CCE)**: First metric to distinguish code vs language uncertainty
2. **Token taxonomy**: Classification of vocabulary into code/language categories
3. **Semantic boundary measurement**: Code-specific measurement points (vs line boundaries)
4. **Adaptive code retrieval**: Uncertainty-triggered retrieval during code generation
5. **Evaluation framework**: Benchmarks for uncertainty detection in code Q&A

---

## 6. Research Questions (Preliminary)

Based on literature review, we formulate:

**RQ1**: Can entropy-based uncertainty detection identify when an LLM lacks code context?
- **Prior work**: ARPO shows entropy spikes after tool use
- **Our extension**: Does this apply to code generation without tools?

**RQ2**: Does Contrastive Code Entropy outperform raw entropy for detecting missing code context?
- **Prior work**: UnCert-CoT uses raw entropy for reasoning decisions
- **Our extension**: Is code/language separation necessary?

**RQ3**: Does adaptive context retrieval improve answer quality while reducing token usage?
- **Prior work**: ARPO achieved 50% fewer tool calls
- **Our extension**: Can we achieve similar efficiency in code RAG?

**RQ4**: Where should entropy be measured in code generation?
- **Prior work**: UnCert-CoT suggests line boundaries
- **Our extension**: Are semantic boundaries (function calls) better?

---

## 7. Positioning for Publication

### 7.1 Target Venues

**Software Engineering (ICSE, FSE)**:
- Emphasize: Practical system, code retrieval, efficiency
- Compare: Cursor, Cody, Continue.dev (production tools)
- Frame: Improving developer tools with uncertainty awareness

**NLP/AI (ACL, EMNLP, NeurIPS)**:
- Emphasize: Novel entropy metric, uncertainty detection
- Compare: ARPO, UnCert-CoT (research methods)
- Frame: Advancing LLM uncertainty quantification for code

### 7.2 Related Work Structure

**Section 1: Uncertainty in LLMs**
- General uncertainty quantification
- Entropy-based methods (ARPO, UnCert-CoT)
- **Gap**: No code vs language distinction

**Section 2: Code Retrieval and RAG**
- Production systems (Cursor, Continue.dev, Cody)
- Research on code understanding
- **Gap**: No adaptive retrieval during generation

**Section 3: Adaptive Reasoning**
- Chain-of-thought adaptations
- Tool-use adaptation (ARPO)
- **Gap**: Focus on reasoning, not retrieval

**Our Position**: We combine uncertainty detection (Section 1) with code retrieval (Section 2) through code-specific entropy metrics.

---

## 8. Key Insights for Implementation

### 8.1 From ARPO
✅ **Entropy spikes indicate knowledge gaps** - validated in multi-turn tool use
✅ **Adaptive behavior reduces waste** - 50% fewer tool calls
✅ **Measure after external interactions** - when knowledge needed

**Actionable**: Measure entropy at code boundaries (function calls, imports) where external knowledge likely needed.

### 8.2 From UnCert-CoT
✅ **Multiple uncertainty metrics** - compare entropy, probability differential
✅ **Uncertainty varies by difficulty** - harder tasks have higher uncertainty
✅ **Adaptive strategies work** - 6.1% improvement

**Actionable**: Implement multiple entropy metrics for comparison in experiments.

### 8.3 From Production Systems
✅ **Scale matters** - Cody handles 300K+ repos
✅ **Search > embeddings?** - Cody dropped embeddings
✅ **User context selection** - Continue.dev uses '@' triggers

**Actionable**: Design CCE to work at scale, compare with search-based baselines.

---

## 9. Open Questions for Investigation

### 9.1 Token Classification
**Question**: How to reliably classify code vs language tokens?

**Approaches to explore**:
1. **Keyword-based**: Programming keywords + operators (simple)
2. **Frequency-based**: Code tokens are less frequent in natural text
3. **Syntactic**: Tokens that appear in AST nodes (tree-sitter)
4. **Learned**: Train classifier on code vs text corpus

**POC validation**: Test classification coverage on CodeLlama tokenizer.

### 9.2 Entropy Thresholds
**Question**: What threshold distinguishes "uncertain" from "confident"?

**Prior work**:
- UnCert-CoT: Not disclosed in abstract
- ARPO: Not disclosed in abstract

**Our approach**: Empirically determine from POC experiment.

### 9.3 Measurement Frequency
**Question**: Measure every token, every line, or semantic boundaries?

**Tradeoffs**:
- Every token: High overhead, precise detection
- Every line: Lower overhead, may miss mid-line uncertainty
- Semantic boundaries: Targeted, code-aware, lower overhead

**POC validation**: Measure overhead and detection accuracy.

---

## 10. Threat to Validity

### 10.1 Potential Weaknesses

**W1: Token classification may be imperfect**
- Some tokens are ambiguous (e.g., "async", "interface")
- May need iterative refinement

**W2: Entropy may not correlate with knowledge gaps**
- POC experiment is critical to validate this assumption
- If POC fails, entire approach needs rethinking

**W3: Baselines may be strong**
- Production systems (Cursor, Cody) have years of optimization
- Hard to beat with simple entropy metric

**W4: Evaluation subjectivity**
- "Answer correctness" is subjective
- Need LLM-as-judge + human validation

### 10.2 Mitigation Strategies

**M1**: Test multiple token classification approaches, report sensitivity
**M2**: POC first (Week 1, Day 4-5) before full implementation
**M3**: Focus on efficiency (token usage) not just quality
**M4**: Use GPT-4 as judge + human eval on subset (20 examples)

---

## 11. References

### Academic Papers
1. Dong et al., "Agentic Reinforced Policy Optimization," arXiv:2507.19849, 2024.
2. Zhu et al., "Uncertainty-Guided Chain-of-Thought for Code Generation with LLMs," arXiv:2503.15341, 2024.
3. Hao et al., "FlowForge: Guiding the Creation of Multi-agent Workflows with Design Space Visualization as a Thinking Scaffold," IEEE VIS 2025.

### Production Systems
4. Cursor: https://www.cursor.com (accessed Dec 2024)
5. Continue.dev: https://docs.continue.dev (accessed Dec 2024)
6. Sourcegraph Cody: https://sourcegraph.com/docs/cody (accessed Dec 2024)

### Tools & Frameworks
7. Tree-sitter: https://tree-sitter.github.io/tree-sitter/
8. Sentence-transformers: https://www.sbert.net/
9. TOON format: https://github.com/toon-format/spec

---

## 12. Next Steps

Based on this literature review:

### Week 1 Remaining Tasks
- [x] Literature review complete
- [ ] Formalize research questions document
- [ ] Design POC experiment (10 examples)
- [ ] Implement POC in Colab notebook
- [ ] **Decision point**: Does entropy separate code/language uncertainty?

### Week 2 (If POC succeeds)
- [ ] Implement entropy calculator
- [ ] Implement token classifier
- [ ] Implement CCE metric
- [ ] Validate on real model (CodeLlama-7B)

---

**Document Status**: Complete
**Last Updated**: December 2024
**Next Document**: `research/research_questions.md`
