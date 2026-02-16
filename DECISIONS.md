# DECISIONS.md — Key Trade-offs & Design Decisions

## Entity Resolution

**Decision: Fuzzy-first, LLM-second (not LLM-only)**
- Why: Fuzzy matching is fast (1000 posts/sec), free, deterministic, and auditable. Same input → same output. LLM is slow (~100ms/post), costs money, and non-deterministic.
- Trade-off: Fuzzy misses abbreviations ($PFE), product-parent links (Tremfya → J&J), and contextual references. LLM catches these but only runs on the 27% of posts fuzzy couldn't resolve.
- If I had more time: Build a feedback-driven alias expansion loop — every time the LLM or a human corrects a fuzzy miss, add the new alias to the catalog automatically.

**Decision: Use `text` field, not `text_altered`**
- Why: `text_altered` replaces entity names with descriptions ("the British-Swedish pharmaceutical company"). Using it loses ~60% of entity matches. This is a data trap in the challenge.

**Decision: Numeric confidence (0-1) not just categorical**
- Why: The assignment asks for `confidence` per entity match. Numeric scores enable threshold tuning (auto-accept above 0.8, flag below 0.5) and distribution visualization in the UI.

## Narrative Clustering

**Decision: LLM topic hashing over TF-IDF/BM25 clustering**
- Why: TF-IDF on short social media posts produces one mega-cluster + dust. Bag-of-words treats "vaccine side effects" and "adverse reactions post-jab" as completely different. The LLM understands semantic similarity and assigns consistent topic labels that we GROUP BY directly.
- Trade-off: Requires API key for best results. Keyword fallback produces workable but lower-quality clusters.
- If I had more time: Implement embedding-based clustering (sentence-transformers) as a middle tier between keywords and LLM. Also implement topic label normalization — a second LLM pass to merge near-duplicate labels.

**Decision: Fuzzy merge for keyword mode (token_sort_ratio ≥ 55)**
- Why: Keyword topic labels are noisy ("vaccine clinical trials" vs "pfizer clinical trial data"). Token-based fuzzy matching merges labels that share enough meaningful words.
- Trade-off: threshold of 55 is empirically tuned on this dataset. May need adjustment for different domains.

## Risk Scoring

**Decision: Weighted composite over LLM scoring**
- Why: Assignment says "Score must not be magic." An LLM that outputs "risk: 73" is exactly the black box they don't want. Weighted composite is auditable — each driver's contribution is visible.
- Trade-off: Weights are manually set (not learned from data). But they're explicit, defensible, and easy to adjust based on feedback.

**Decision: Language gets highest weight (0.30)**
- Why: Content IS the risk. "CEO arrested for fraud" is dangerous at any volume. Taxonomy category is the strongest signal of reputational harm.

**Decision: Percentile normalization (not min-max or z-score)**
- Why: Percentile rank is robust to outliers and intuitively meaningful ("this narrative's engagement is in the 85th percentile"). Min-max would be distorted by a single viral post.

**Decision: Confidence band, not point estimate**
- Why: A risk score of 73 from 2 posts is very different from 73 from 50 posts. The confidence band communicates uncertainty explicitly (±8 for high confidence, ±25 for low).
- If I had more time: Bootstrap confidence intervals — resample posts within each narrative and compute score variance.

## UI

**Decision: Streamlit over React/Flask**
- Why: Assignment specifies Streamlit. Also fastest path to a working prototype.

**Decision: Expander-based detail view (not separate pages)**
- Why: Single-page design keeps context visible. The evaluator can see the narrative list while drilling into detail. Simpler than multi-page routing.

## If I Had More Time (2-4 weeks)

1. **Embedding-based clustering**: sentence-transformers for semantic similarity without LLM API cost. Would replace keyword fallback entirely.
2. **Active learning loop**: Route low-confidence entity matches to human review, use corrections to expand alias catalog.
3. **Temporal risk tracking**: Track narrative risk scores over time, detect spikes and trend changes.
4. **Cross-entity narrative detection**: Same narrative mentioning multiple entities (e.g., "Pfizer and Moderna vaccine comparison").
5. **Langfuse integration**: Trace every LLM call with input/output/latency/cost for observability.
6. **AWS deployment**: ECS for Streamlit, Lambda for pipeline, S3 for artifacts, DynamoDB for feedback. See presentation deck.
7. **Evaluation framework**: Hand-label 100 posts, compute precision/recall for entity resolution, cluster purity for narratives, correlation with human risk ratings.
8. **Drift detection**: Monitor entity resolution confidence distribution over time, alert when confidence drops (new entities, language shift).
