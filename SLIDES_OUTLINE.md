# Presentation Deck — Key Points Per Slide

## Slide 1: Problem Framing + Assumptions
- Goal: "For a chosen entity, what are top narratives, how risky (0-100), and why?"
- Assumption: Social-only signals (no internal data, no news feeds)
- Assumption: Pharma/healthcare domain — risk = reputational harm
- Assumption: 1000 posts, 19 entities, 30-day window (Nov 2025)
- Data traps identified: text_altered field, zero aliases, spam posts

## Slide 2: Data Overview + Key Challenges
- 1000 posts: 511 Facebook, 489 Twitter
- 19 entities: 13 brands, 6 products
- 907 authors: followers range 1 to 20M
- Challenges: text_altered trap, spam/hashtag-stuffed posts (~8%), short text (<280 chars), missing engagement data (views=0 on Facebook)
- Date range: Nov 1-30, 2025

## Slide 3: Entity Resolution
- 3-tier hybrid: Fuzzy (primary, 73% coverage) → Embedding (optional) → LLM (hard cases only, 27%)
- Fuzzy: rapidfuzz ratio ≥ 90 against alias catalog
- LLM: Claude Haiku for abbreviations ($PFE), product-parent links, disambiguation
- Output: 997 mentions, 732 posts linked, all 19 entities found
- Error modes: "Merck Handbook" (book not company), "BMS" ambiguity
- Cost: $0.10 for LLM tier on 268 hard posts

## Slide 4: Narrative Clustering
- Approach: LLM topic hashing → GROUP BY (not TF-IDF clustering)
- Why: Short social posts defeat bag-of-words methods (one mega-cluster)
- LLM labels each post with canonical 3-8 word topic → consistent labels → groupby
- Keyword fallback: extract distinctive words + fuzzy merge (token_sort_ratio ≥ 55)
- Result: 162 narratives across 19 entities
- Improvement path: embedding clustering, topic label normalization

## Slide 5: Risk Score Design (0-100)
- Formula: 0.30×Language + 0.20×Volume + 0.20×Engagement + 0.15×Velocity + 0.15×Author
- Language highest (0.30): content IS the risk — taxonomy category has inherent meaning
- Normalization: percentile rank across all narratives (no arbitrary thresholds)
- Language uses direct mapping: Customer Harm base=90, General=20
- Engagement weighted: shares×3, comments×2, likes×1, views×0.1
- Velocity: posts/day + acceleration detection

## Slide 6: Explainability + Evidence + Audit Trail
- Every score decomposed: "Language contributed 28.2pts (taxonomy=Customer Harm, 72% negative)"
- Evidence posts: top 5 by engagement×risk signal
- Confidence band: ±8 (high), ±15 (medium), ±25 (low)
- Confidence factors: sample size, signal consistency, data completeness
- Full audit trail: every entity resolution carries source + method + score

## Slide 7: Guardrails
- LLM outputs validated against entity catalog (hallucinated IDs filtered)
- Temperature=0 for deterministic outputs
- Every LLM step has keyword fallback (pipeline works offline)
- Grounding: titles and summaries cite actual post content
- Risk score is composite of named signals (not LLM-generated number)
- Uncertainty: confidence bands, needs_review flags
- Feedback loop: captures entity corrections + risk ratings

## Slide 8: Production Plan (AWS)
- ECS Fargate: Streamlit app + pipeline workers
- Lambda: triggered pipeline runs (scheduled or event-driven)
- S3: post data, pipeline artifacts, feedback logs
- DynamoDB: entity catalog, feedback store, alias expansion
- SQS: queue for LLM API calls with rate limiting
- CloudWatch: pipeline metrics, latency, error rates
- Langfuse: LLM observability — traces, cost tracking, quality monitoring
- CI/CD: GitHub Actions → ECR → ECS blue-green deploy

## Slide 9: What I'd Do Next (2-4 weeks)
- Week 1: Embedding clustering (sentence-transformers), hand-label 100 posts for evaluation
- Week 2: Active learning loop — low-confidence → human review → alias expansion
- Week 3: Temporal risk tracking, cross-entity narrative detection
- Week 4: Langfuse integration, drift detection, A/B test risk weights
- Evaluation: precision/recall on hand-labeled set, cluster purity, risk score correlation with human ratings
