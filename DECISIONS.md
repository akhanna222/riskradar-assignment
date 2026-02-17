# DECISIONS.md — Key Trade-offs & Design Decisions

## Entity Resolution

**Decision: Fuzzy-first, LLM-second (not LLM-only)**
- Why: Fuzzy matching is fast (1000 posts/sec), free, deterministic, and auditable. Same input → same output. LLM is slow (~100ms/post), costs money, and non-deterministic.
- Trade-off: Fuzzy misses abbreviations ($PFE), product-parent links (Tremfya → J&J), and contextual references. LLM catches these but only runs on the 27% of posts fuzzy couldn't resolve.
- If I had more time: Build a feedback-driven alias expansion loop — every time the LLM or a human corrects a fuzzy miss, add the new alias to the catalog automatically.

**Decision: Use `text` field, not `text_altered`**
- Why: `text_altered` replaces entity names with descriptions ("the British-Swedish pharmaceutical company"). Using it loses ~60% of entity matches. This is a data trap in the challenge.
- How I spotted it: Ran entity resolution on both fields, compared match rates. `text_altered` dropped from 73% to ~28% coverage.

**Decision: Numeric confidence (0-1) not just categorical**
- Why: The assignment asks for `confidence` per entity match. Numeric scores enable threshold tuning (auto-accept above 0.8, flag below 0.5) and distribution visualization in the UI.

**Decision: Word-boundary matching (regex \b) instead of substring**
- Why: Substring `"merck" in text` would false-positive on "commercial". Word boundary regex prevents this while still matching "Merck's" and "Merck,".
- Trade-off: ~2x slower than simple `in` operator, but still sub-millisecond per post.

**Decision: LLM-as-judge audit for fuzzy precision estimation**
- Why: Without hand-labeled data, we need a proxy for precision. Sampling 100 fuzzy matches and having the LLM judge them gives an agree rate (e.g., 92%) that serves as an upper-bound precision estimate.
- Trade-off: LLM judges aren't ground truth — they have their own error rate. But it's better than no evaluation signal at all.

## Narrative Clustering

**Decision: LLM topic hashing over TF-IDF/BM25 clustering**
- Why: TF-IDF on short social media posts produces one mega-cluster + dust. Bag-of-words treats "vaccine side effects" and "adverse reactions post-jab" as completely different. The LLM understands semantic similarity and assigns consistent topic labels that we GROUP BY directly.
- Trade-off: Requires API key. No offline fallback — we chose to invest the timebox in LLM quality rather than building a parallel keyword pipeline.
- If I had more time: Implement embedding-based clustering (sentence-transformers + HDBSCAN) as a middle tier — $0 cost, better cluster quality than keywords, no API dependency.

**Decision: Fuzzy merge for near-duplicate topic labels (token_sort_ratio ≥ 55)**
- Why: Even with the consistency prompt, LLMs produce near-duplicates like "vaccine safety concerns" vs "pfizer vaccine safety." Fuzzy merging with rapidfuzz catches these without an extra LLM call.
- Trade-off: Threshold of 55 is empirically tuned. Too low merges unrelated topics; too high leaves duplicates. 55 was the sweet spot on this dataset.

**Decision: LLM-only clustering (no keyword fallback)**
- Why: Within the 4-6 hour timebox, building a high-quality keyword fallback that produces usable narratives would consume ~2 hours. The LLM path produces meaningfully better clusters (semantic understanding of short posts), and the cost is negligible ($0.05/1000 posts on Haiku). We chose to invest timebox in scoring quality and UI instead.
- If I had more time: Add embedding-based clustering (sentence-transformers → HDBSCAN) as the offline fallback. This gives semantic quality without LLM cost.
- Production implication: The system requires API access for narrative clustering. In production, this is standard — the LLM gateway handles rate limiting and fallback.

## Risk Scoring

**Decision: Weighted composite over LLM scoring**
- Why: Assignment says "Score must not be magic." An LLM that outputs "risk: 73" is exactly the black box they don't want. Weighted composite is auditable — each driver's contribution is visible.
- Trade-off: Weights are manually set (not learned from data). But they're explicit, defensible, and easy to adjust based on feedback.

**Decision: Language gets highest weight (0.30)**
- Why: Content IS the risk. "CEO arrested for fraud" is dangerous at any volume. Taxonomy category is the strongest signal of reputational harm.
- Evidence: In the dataset, the highest-scoring narratives all had high language scores (taxonomy: Customer Harm, Regulatory). Volume and engagement amplify risk but don't create it.

**Decision: Percentile normalization (not min-max or z-score)**
- Why: Percentile rank is robust to outliers and intuitively meaningful ("this narrative's engagement is in the 85th percentile"). Min-max would be distorted by a single viral post.
- Exception: Author influence uses tiered scoring instead of percentile — a 89-follower account shouldn't score 50th percentile just because half the dataset has fewer followers.

**Decision: Tiered author scoring (not log-scale or percentile)**
- Why: Follower counts span 1 to 20M. Log-scale compresses the range too much (89 followers scores similarly to 5,000). Linear mapping (0→0, 2M→100) under-weights the middle. Fixed tiers give intuitive, defensible scores:
  - 0–1K followers → 10, 1K–10K → 25, 10K–100K → 50, 100K–1M → 75, 1M+ → 100
- Trade-off: Thresholds are manually set. But they align with social media influence research (micro/macro/mega influencer tiers) and are easy to adjust.
- Author scores are now also percentile-ranked across narratives for consistency with volume and engagement.

**Decision: Platform-aware engagement normalization**
- Why: Facebook does not expose view counts (always 0). Including views in engagement scoring penalizes the 511 Facebook posts (51% of data). We exclude views for Facebook posts specifically.
- Trade-off: This means Facebook posts are scored only on shares + comments + likes. If a future platform also suppresses a metric, the same pattern applies.

**Decision: Confidence band, not point estimate**
- Why: A risk score of 73 from 2 posts is very different from 73 from 50 posts. The confidence band communicates uncertainty explicitly (±8 for high confidence, ±25 for low).
- If I had more time: Bootstrap confidence intervals — resample posts within each narrative and compute score variance.

**Decision: Word-boundary matching for risk keywords**
- Why: Substring matching causes "risk" to match inside "asterisk" and "investigation" to match inside "investigations" (which is fine, but "risk" inside "asterisk" is not). Regex word boundaries prevent false positives while still matching inflected forms.

## UI

**Decision: Streamlit over React/Flask**
- Why: Assignment specifies Streamlit. Also fastest path to a working prototype.

**Decision: Expander-based detail view (not separate pages)**
- Why: Single-page design keeps context visible. The evaluator can see the narrative list while drilling into detail. Simpler than multi-page routing.

**Decision: Per-post entity correction in evidence posts**
- Why: The assignment specifies "Incorrect entity match → user selects correct entity / 'none'" per post. We provide both narrative-level bulk correction AND per-post correction on evidence posts for finer granularity.

**Decision: Risk taxonomy reference in sidebar**
- Why: The taxonomy definitions are part of the risk model. Analysts need to understand what "Customer Harm" vs "Regulatory / Compliance" means to evaluate score drivers effectively.

**Decision: Cache invalidation on feedback**
- Why: Without clearing `st.cache_data` after feedback submission, cached entity stats and resolved data become stale. We clear cache on every feedback submission to ensure the UI reflects the latest overrides immediately.

## Evaluation Strategy (Without Labels)

**How we evaluate without ground truth:**
1. **Entity resolution precision** — LLM-as-judge audit on 100 sampled fuzzy matches. Reports agree/disagree rate.
2. **Entity resolution recall** — We check coverage: what % of posts have at least one resolved entity. Currently 73%.
3. **Cluster quality** — Average cluster size, number of singletons, largest cluster ratio. Fuzzy merge reduces fragmentation.
4. **Risk score face validity** — Worked examples in presentation. Evidence posts let analysts validate that high-scoring narratives are actually risky.
5. **Feedback rate as proxy** — Over time, if analyst correction rate decreases, the system is improving.

**If I had a hand-labeled sample (next step):**
- Label 100 posts with: true entity_id(s), true topic, risk severity (1-5)
- Compute: entity resolution P/R, cluster purity (% of cluster with same true topic), risk score correlation with human severity rating

## If I Had More Time (2-4 weeks)

1. **Embedding-based clustering**: sentence-transformers + HDBSCAN for semantic similarity without LLM API cost.
2. **Active learning loop**: Route low-confidence entity matches to human review, use corrections to expand alias catalog.
3. **Temporal risk tracking**: Track narrative risk scores over time, detect spikes and trend changes.
4. **Cross-entity narrative detection**: Same narrative mentioning multiple entities.
5. **Langfuse integration**: Trace every LLM call with input/output/latency/cost for observability.
6. **AWS deployment**: ECS for Streamlit, Lambda for pipeline, S3 for artifacts. See presentation deck.
7. **Evaluation framework**: Hand-label 100 posts, compute precision/recall for entity resolution, cluster purity for narratives.
8. **Drift detection**: Monitor entity resolution confidence distribution over time, alert when confidence drops.
9. **Alias auto-expansion**: Mine LLM corrections and human overrides to automatically grow the alias catalog.
