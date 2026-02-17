# Presentation Deck — Key Points Per Slide

## Slide 1: Title
- RiskRadar: Social-Only Narrative Risk Triage
- Lead Data Scientist — Take-Home Challenge

## Slide 2: Problem Framing + Assumptions
- Goal: "For a chosen entity, what are top narratives, how risky (0-100), and why?"
- Scope: Social-only signals, entity-agnostic (domain driven by entity catalog)
- Assumptions: text field is ground truth (not text_altered), risk is relative,
  LLM optional, engagement not comparable cross-platform, feedback loop

## Slide 3: Data Overview + Key Challenges
- 1000 posts (511 FB, 489 Twitter), 19 entities, 907 authors, 30-day window
- Challenges: text_altered trap, short posts, zero aliases, mixed languages

## Slide 4: Entity Resolution
- 3-tier: Fuzzy (rapidfuzz ≥ 90, 73%) → LLM (Claude Haiku, 27%) → Merge
- Confidence: fuzzy = ratio/100, LLM = self-reported, merge = highest wins
- Error modes + human override via overrides.json

## Slide 5: Narrative Clustering
- LLM topic hashing → GROUP BY → fuzzy merge (token_sort_ratio ≥ 55)
- Fuzzy merge reduces cluster fragmentation from near-duplicate LLM labels
- Next: embedding clustering (offline, $0 cost), cross-entity detection, cluster purity metrics
- Design choice: LLM-only (no keyword fallback) — timebox invested in scoring quality instead

## Slide 6: Risk Score Design (0-100)
- Formula: 0.30×Language + 0.20×Volume + 0.20×Engagement + 0.15×Velocity + 0.15×Author
- Author: tiered (0-1K→10, 10K→25, 100K→50, 1M→75, 1M+→100)
- Concrete example: Merck narrative → 76.2/100 with full breakdown

## Slide 7: Explainability + Evidence + Audit Trail
- 5 drivers explained with sub-components
- Evidence posts: top 5 by engagement × risk
- Confidence bands: ±8/±15/±25

## Slide 8: Guardrails — Built vs. Proposed
- Built: hallucination prevention, transparency, confidence bands, keyword fallback
- Proposed: Langfuse, drift detection, active learning, bias monitoring

## Slide 9: Production Architecture
- 3-layer: Ingest (batch + embeddings) → Process (ER + clustering + scoring) → Serve (dashboard + HITL)
- Feedback loop from HITL back to pipeline

## Slide 10: What I'd Do Next (2-4 weeks)
- Week 1: Embedding clustering, hand-label 100 posts
- Week 2: Active learning, cross-entity detection
- Week 3: Temporal tracking, drift detection
- Week 4: Langfuse, AWS deploy, evaluation dashboard
