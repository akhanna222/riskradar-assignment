# RiskRadar — Social-Only Narrative Risk Triage

Entity resolution, narrative clustering, and explainable risk scoring on social media posts about pharmaceutical entities.

## Setup

```bash
# Clone and enter
git clone https://github.com/abhishekpingale/riskradar-assignment.git
cd riskradar-assignment

# One-command setup (installs deps, runs tests, checks outputs)
bash setup.sh

# Launch the app
streamlit run app.py
```

Or step by step:

```bash
pip install -r requirements.txt
python tests.py             # verify everything works
streamlit run app.py        # launch UI
```

## API Key (Optional)

The pipeline works without an API key using fuzzy matching and keyword extraction. For better results, add an Anthropic API key:

1. Get a free key at **https://console.anthropic.com/settings/keys**
2. Copy `.env.example` to `.env` and paste your key
3. Re-run the pipeline: `python run_pipeline.py`

You can also paste the key directly in the Streamlit sidebar and click "Run Full Pipeline".

## Docker (Alternative)

```bash
docker build -t riskradar .
docker run -p 8501:8501 riskradar

# With API key
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... riskradar
```

Open http://localhost:8501

## How It Works

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

**Entity Resolution** — 3-tier hybrid: fuzzy alias matching (73% of posts) → LLM tagger for hard cases. Output: `{entity_id, confidence, resolution_method}` per post.

**Narrative Clustering** — LLM assigns canonical topic labels → GROUP BY topic. Keyword fallback: extract distinctive words + fuzzy merge similar labels. Output: `{narrative_id, title, summary, post_ids}`.

**Risk Scoring** — Weighted composite: Language(0.30) + Volume(0.20) + Engagement(0.20) + Velocity(0.15) + Author(0.15). Every score has 5 driver breakdowns, evidence posts, and confidence band.

**Human-in-the-Loop** — Analyst corrects entities or risk ratings in UI → writes to `overrides.json` → applied live on app reload AND on next pipeline run.

## Project Files

```
├── app.py                     # Streamlit UI
├── run_pipeline.py            # One-shot pipeline runner
├── tests.py                   # Unit tests (run before app)
├── setup.sh                   # One-command setup
├── entity_resolution.py       # Capability 1
├── narrative_clustering.py    # Capability 2
├── risk_scoring.py            # Capability 3
├── Dockerfile                 # Docker option
├── requirements.txt
├── .env.example               # API key template
├── data/                      # Input files
│   ├── posts.jsonl
│   ├── entities_seed.csv
│   └── authors.csv
└── outputs/                   # Pipeline outputs
    ├── resolved_entities.jsonl
    ├── narratives/
    ├── scored/
    ├── overrides.json         # Human corrections (created by UI)
    └── feedback.jsonl         # Audit log (created by UI)
```
