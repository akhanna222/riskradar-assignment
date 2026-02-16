# RiskRadar — Social-Only Narrative Risk Triage

Entity resolution + narrative clustering + explainable risk scoring on social media posts about pharmaceutical entities.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Set up API key for LLM-enhanced mode
cp .env.example .env
# Edit .env and paste your Anthropic API key

# Run the pipeline
python run_pipeline.py                    # reads key from .env automatically
python run_pipeline.py --api-key sk-...   # or pass key directly

# Launch the UI
streamlit run app.py
```

The pipeline works in two modes:

**Without API key (default):** Fuzzy entity matching + keyword topic extraction + fuzzy merge clustering. Free, fast, decent baseline.

**With API key (.env or --api-key):** LLM-enhanced entity resolution (catches abbreviations, product-parent links), LLM topic labeling (semantic clustering), LLM narrative summarization. ~$0.15 total for 1000 posts on Claude Haiku.

You can also re-run the pipeline from the Streamlit UI sidebar — paste your API key and click "Run Full Pipeline".

## Architecture

```
posts.jsonl + entities_seed.csv + authors.csv
        │
        ▼
┌─────────────────────────┐
│  1. Entity Resolution   │  entity_resolution.py
│  Fuzzy + LLM hybrid     │  → outputs/resolved_entities.jsonl
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  2. Narrative Clustering │  narrative_clustering.py
│  LLM topic hashing +    │  → outputs/narratives/{entity}.json
│  fuzzy merge fallback    │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  3. Risk Scoring         │  risk_scoring.py
│  Weighted composite      │  → outputs/scored/{entity}_scored.json
│  (0-100) + explainability│
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  4. Streamlit UI         │  app.py
│  Browse, drill down,     │  → outputs/feedback.jsonl
│  capture feedback        │
└─────────────────────────┘
```

## Project Structure

```
riskradar/
├── app.py                     # Streamlit prototype
├── run_pipeline.py            # One-shot pipeline runner
├── entity_resolution.py       # Capability 1: entity mention extraction + resolution
├── narrative_clustering.py    # Capability 2: narrative clustering
├── risk_scoring.py            # Capability 3: risk scoring + explainability
├── data/
│   ├── posts.jsonl            # Input: 1000 social media posts
│   ├── entities_seed.csv      # Input: 19 canonical entities
│   └── authors.csv            # Input: 907 author profiles
├── outputs/
│   ├── resolved_entities.jsonl    # Entity resolution results
│   ├── narratives/                # Per-entity narrative clusters
│   │   ├── pfizer_narratives.json
│   │   └── ...
│   ├── scored/                    # Per-entity risk-scored narratives
│   │   ├── pfizer_scored.json
│   │   └── ...
│   └── feedback.jsonl             # Human feedback from UI
├── README.md
├── DECISIONS.md
├── .env.example
└── requirements.txt
```

## Capability 1: Entity Resolution

**Approach:** 3-tier hybrid — fuzzy matching (primary) → embedding similarity (optional) → LLM tagger (hard cases only).

**Why hybrid:** Fuzzy matching handles 73% of posts (732/1000) with zero API cost. The LLM only runs on the 268 posts where fuzzy found nothing or was ambiguous. This gives production-grade cost control: ~$0.10 for the LLM tier on 268 posts via Claude Haiku.

**Output per post:**
- `resolved_entities`: list of `{entity_id, mention_text, confidence, resolution_method}`
- `needs_review`: boolean for low-confidence matches
- `source`: audit trail (fuzzy_embedding / llm / llm_disambiguation)

**Key decision:** We use the `text` field, NOT `text_altered`. The `text_altered` field replaces entity names with descriptions — using it loses ~60% of matches.

## Capability 2: Narrative Clustering

**Approach:** LLM topic hashing (primary) → keyword extraction + fuzzy merge (fallback).

**Why not TF-IDF/BM25:** Bag-of-words methods produce meaningless mega-clusters on short social media posts. The LLM understands "Pfizer vaccine side effects" and "adverse reactions post-jab" are the same topic.

**How it works:** The LLM labels each post with a 3-8 word canonical topic. Same topic = same cluster. In keyword mode, we extract distinctive keywords and fuzzy-merge labels with ≥55% token overlap.

**Output per narrative:** `{narrative_id, title, summary, post_ids, taxonomy_label, sentiment_distribution}`

## Capability 3: Risk Scoring

**Formula:**
```
risk_score = 0.20×volume + 0.15×velocity + 0.20×engagement + 0.15×author + 0.30×language
```

**Why these weights:** Language (0.30) is highest because content IS the risk — "CEO arrested for fraud" is dangerous at any volume. Engagement (0.20) indicates real amplification. Volume (0.20) captures conversation size. Velocity (0.15) and author influence (0.15) are supporting signals.

**Normalization:** Sub-scores use percentile rank across all narratives (no arbitrary thresholds). Language score uses direct taxonomy mapping (Customer Harm base=90, General=20).

**Explainability:** Every score includes top 5 drivers with contribution breakdown, 5 evidence posts ranked by engagement×risk signal, and confidence band (±8 for high confidence, ±25 for low).

## LLM Usage

LLM (Claude Haiku) is used in 3 places, all optional:

1. **Entity resolution** (Tier 2): Tags entities in posts where fuzzy matching failed. Catches abbreviations ($PFE), product-parent links, contextual disambiguation.
2. **Narrative clustering** (Step 2): Labels each post with a canonical topic for consistent clustering.
3. **Narrative summarization** (Step 4): Generates grounded titles and summaries per narrative.

**Guardrails:**
- All LLM outputs are validated against the entity catalog (hallucinated entity_ids are filtered)
- Temperature=0 for deterministic outputs
- JSON parsing with fallback extraction (handles markdown fences)
- Keyword fallback for every LLM step (pipeline works without API key)
- Batch processing (15 posts/batch) to reduce API calls

**Cost:** ~$0.15 total for 1000 posts (entity resolution $0.10 + clustering $0.05 on Haiku).

## Feedback Loop

The Streamlit UI captures two types of feedback:

1. **Entity correction:** "This post was matched to Pfizer but should be Merck / none"
2. **Risk rating:** "This narrative's risk score is too high / too low / correct"

Feedback is persisted to `outputs/feedback.jsonl`. In production, this feeds into:
- Entity resolution retraining (alias expansion from corrections)
- Risk weight calibration (systematic "too high" on a category → adjust weight)
- Active learning (low-confidence posts prioritized for review)
