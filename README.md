# RiskRadar — Social-Only Narrative Risk Triage

Entity resolution, narrative clustering, and explainable risk scoring on social media posts about pharmaceutical entities.

## Setup

```bash
git clone <your-repo-url>
cd riskradar

# Place input data files in data/ folder:
#   data/posts.jsonl
#   data/entities_seed.csv
#   data/authors.csv

bash setup.sh           # installs deps, checks data, runs tests
streamlit run app.py    # launch UI (uses pre-computed outputs)
```

The app ships with **pre-computed outputs** — it works immediately without an API key or re-running the pipeline.

## Re-running the Pipeline (Optional)

This will **overwrite** the `outputs/` folder. You need an Anthropic API key for LLM-enhanced mode.

```bash
# 1. Get a key at https://console.anthropic.com/settings/keys
# 2. Add it to .env
echo "ANTHROPIC_API_KEY=sk-ant-your-key" > .env

# 3. Run
python run_pipeline.py
```

Without a key, the pipeline runs in fuzzy/keyword-only mode (free, lower quality).
You can also re-run from the Streamlit sidebar — paste your key and click "Run Full Pipeline".

## Docker (Alternative)

```bash
docker build -t riskradar .
docker run -p 8501:8501 riskradar
# With API key: docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... riskradar
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

**Entity Resolution** — Fuzzy alias matching (73% of posts) → LLM tagger for hard cases.

**Narrative Clustering** — LLM assigns canonical topic labels → GROUP BY topic. Keyword fallback for no-API mode.

**Risk Scoring** — Weighted composite: Language(0.30) + Volume(0.20) + Engagement(0.20) + Velocity(0.15) + Author(0.15). Each score has driver breakdowns, evidence posts, and confidence band.

**Human-in-the-Loop** — Analyst corrects entities or risk ratings in UI → `overrides.json` → applied live on reload + on next pipeline run.

## Project Files

```
├── app.py                     # Streamlit UI
├── run_pipeline.py            # Pipeline runner
├── tests.py                   # Unit tests (run before app)
├── setup.sh                   # Setup script
├── entity_resolution.py       # Capability 1
├── narrative_clustering.py    # Capability 2
├── risk_scoring.py            # Capability 3
├── Dockerfile                 # Docker option
├── requirements.txt
├── .env.example               # API key template
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
