# RiskRadar — Social-Only Narrative Risk Triage

Entity resolution, narrative clustering, and explainable risk scoring on social media posts.

## Quick Start

```bash
pip install -r requirements.txt

# Place input data in data/ folder:
#   data/posts.jsonl, data/entities_seed.csv, data/authors.csv

streamlit run app.py    # uses pre-computed outputs — works immediately
```

The app ships with **pre-computed outputs** — works immediately without an API key or re-running the pipeline.

## Re-running the Pipeline

```bash
# Fuzzy/keyword-only mode (free, no API key):
python run_pipeline.py

# Full hybrid with LLM (entity resolution + clustering + audit):
python run_pipeline.py --api-key sk-ant-your-key

# Test with fewer posts:
python run_pipeline.py --limit 50

# Or set via .env (auto-loaded by dotenv):
echo "ANTHROPIC_API_KEY=sk-ant-your-key" > .env
python run_pipeline.py
```

You can also re-run from the Streamlit sidebar — paste your key and click "Run Full Pipeline".

## Running Tests

```bash
python tests.py                # unittest runner
python -m pytest tests.py -v   # verbose (optional)
```

Tests verify: input data integrity, module imports, entity resolution schema, narrative clustering output, risk scoring (range, drivers, weight sum), and human override logic.

## Pipeline Architecture

```
posts.jsonl + entities_seed.csv + authors.csv
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
 Entity          Narrative        Risk
 Resolution      Clustering       Scoring
 (fuzzy+LLM)    (LLM topics)    (weighted 0-100)
    │               │               │
    ▼               ▼               ▼
 resolved_       narratives/     scored/
 entities.jsonl  {entity}.json   {entity}_scored.json
                    │
                    ▼
              Streamlit App
              (browse + feedback → overrides.json)
```

### Entity Resolution (Capability 1)

Fuzzy alias matching resolves 73% of posts. LLM handles the remaining 27% (abbreviations like $PFE, product→parent links like Tremfya→J&J, ambiguous mentions).

**Pipeline:** Tier 1 (rapidfuzz ≥ 90) → Tier 2 (Claude Haiku on unresolved) → Merge (fuzzy priority) → **LLM Audit** → Human overrides.

**LLM Audit (LLM-as-judge):** After merge, the LLM reviews a sample of fuzzy matches (default 100, batched 15/call) and judges each as "correct" or "wrong." This gives a precision estimate for the fuzzy tier without changing core pipeline logic.

**`llm_agrees` field on each resolved entity:**

| Value | Meaning |
|-------|---------|
| `true` | LLM reviewed and agrees the match is correct |
| `false` | LLM reviewed and disagrees — potential error |
| `"not_audited"` | LLM was not asked to review this match |
| `"human_override"` | Human corrected this entity in the UI |

### Narrative Clustering (Capability 2)

LLM assigns each post a canonical 3–8 word topic label → GROUP BY label → fuzzy merge similar labels (token_sort_ratio ≥ 55). Keyword fallback for no-API mode: extract distinctive words, build labels, merge.

### Risk Scoring (Capability 3)

Weighted composite (0–100): `0.30×Language + 0.20×Volume + 0.20×Engagement + 0.15×Velocity + 0.15×Author`

Each score includes 5 driver breakdowns with sub-details, top evidence posts, and confidence band (±8 high, ±15 medium, ±25 low). Author influence uses tiered scoring: 0–1K→10, 10K→25, 100K→50, 1M→75, 1M+→100.

### Human-in-the-Loop

Analyst corrects entities or adjusts risk ratings (too high / too low) in the Streamlit UI. Corrections are saved to `overrides.json` and applied live on page reload. On the next pipeline run, entity overrides are also applied before merge. All corrections are additionally logged to `feedback.jsonl` (append-only audit trail).

## Streamlit UI Features

**Sidebar:** Entity selector, matched post count, average confidence, confidence distribution (high/medium/low), LLM audit agree rate (when audit data exists), and a "Re-run Pipeline" button with API key input.

**Main view:** Narratives ranked by risk score with color-coded severity (🔴 ≥70, 🟡 ≥40, 🟢 <40). Each narrative expands to show: 5 scored drivers with contribution breakdown, taxonomy label, sentiment distribution, confidence band, top evidence posts with full text/author/engagement, and feedback controls (correct entity, adjust risk rating).

## Output Schemas

**`resolved_entities.jsonl`** — one JSON object per post:
```json
{
  "post_id": "12345",
  "text": "post text (truncated to 500 chars)...",
  "resolved_entities": [
    {
      "entity_id": "pfizer",
      "canonical_name": "Pfizer",
      "entity_type": "Brand",
      "mention_text": "pfizer",
      "confidence": 0.95,
      "confidence_label": "high",
      "resolution_method": "fuzzy_alias",
      "source": "fuzzy_alias",
      "llm_agrees": true
    }
  ],
  "needs_review": false
}
```

**`narratives/{entity}.json`** — array of narrative clusters:
```json
{
  "narrative_id": "pfizer_narrative_0",
  "entity_id": "pfizer",
  "title": "Vaccine side effects and safety concerns",
  "summary": "Posts discuss reported side effects...",
  "taxonomy_label": "Customer Harm",
  "sentiment_distribution": {"negative": 8, "neutral": 3},
  "post_count": 11,
  "posts": ["post_id_1", "post_id_2"]
}
```

**`scored/{entity}_scored.json`** — array of risk-scored narratives:
```json
{
  "narrative_id": "pfizer_narrative_0",
  "title": "...",
  "risk_score": 76.2,
  "confidence": "medium",
  "confidence_band": [61.2, 91.2],
  "drivers": [
    {"name": "Language Signals", "score": 90.0, "weight": 0.30, "contribution": 27.0, "detail": {...}},
    {"name": "Volume",           "score": 85.0, "weight": 0.20, "contribution": 17.0, "detail": {...}}
  ],
  "evidence_posts": [
    {"post_id": "...", "text": "...", "author_id": "...", "followers": 134000, "likes": 45}
  ]
}
```

**`overrides.json`** — human corrections:
```json
{
  "entity_overrides": {
    "post_id_123": {"corrected_entity": "merck", "original": "pfizer", "timestamp": "..."}
  },
  "risk_overrides": {
    "pfizer_narrative_0": {"feedback": "Too High", "timestamp": "..."}
  }
}
```

## Guardrails

**Built into the codebase:**
- LLM entity outputs validated against catalog — hallucinated entity IDs filtered out
- Temperature = 0 for deterministic LLM outputs
- JSON parsing with markdown fence fallback (handles ```json wrapping)
- Narrative titles and summaries grounded in actual post content
- Risk score = formula-based composite (not LLM-generated number)
- Confidence bands communicate uncertainty (±8 to ±25 depending on sample size)
- `needs_review` flag on entity matches below 0.5 confidence
- Every LLM step has keyword fallback — pipeline works fully offline

## Cost Estimates (with API key)

| Step | Posts | Approx Cost |
|------|-------|-------------|
| Entity Resolution — Tier 2 (hard cases) | ~268 | ~$0.10 |
| Narrative Clustering (LLM topic labels) | ~1000 | ~$0.05 |
| LLM Audit (fuzzy validation) | ~100 | ~$0.02 |
| **Total** | | **~$0.17** |

Without an API key, the pipeline runs in fuzzy/keyword-only mode at zero cost.

## Project Files

```
├── app.py                     # Streamlit UI (sidebar + narrative drill-down + feedback)
├── run_pipeline.py            # CLI pipeline runner (--api-key, --limit)
├── entity_resolution.py       # Capability 1: fuzzy + LLM + audit
├── narrative_clustering.py    # Capability 2: LLM topic hashing + keyword fallback
├── risk_scoring.py            # Capability 3: weighted composite scoring
├── tests.py                   # Unit tests (data, imports, schemas, overrides)
├── requirements.txt           # rapidfuzz, scikit-learn, streamlit, anthropic
├── .env.example               # API key template
├── DECISIONS.md               # Design trade-offs and rationale
├── RiskRadar_Colab.ipynb      # Google Colab notebook
├── data/                      # Input files (user provides)
│   ├── posts.jsonl
│   ├── entities_seed.csv
│   └── authors.csv
└── outputs/                   # Pre-computed pipeline outputs
    ├── resolved_entities.jsonl
    ├── narratives/
    ├── scored/
    ├── overrides.json         # Human corrections (created by UI)
    └── feedback.jsonl         # Audit log (created by UI)
```

## Design Decisions

See [DECISIONS.md](DECISIONS.md) for detailed trade-offs: why fuzzy-first not LLM-only, why `text` not `text_altered`, why weighted composite over LLM scoring, tiered author scoring vs log-scale, percentile normalization, and the 2–4 week improvement roadmap.
