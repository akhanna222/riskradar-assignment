#!/bin/bash
# ──────────────────────────────────────────────────────────────
# RiskRadar Setup
# ──────────────────────────────────────────────────────────────

set -e

echo "================================"
echo " RiskRadar — Setup"
echo "================================"

# 1. Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt -q

# 2. Set up .env (if not exists)
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[2/4] Created .env from .env.example"
    echo "      → Edit .env to add your Anthropic API key (optional)"
    echo "      → Get a key at: https://console.anthropic.com/settings/keys"
else
    echo "[2/4] .env already exists — skipping"
fi

# 3. Run tests
echo ""
echo "[3/4] Running tests..."
python tests.py 2>&1 | tail -5

# 4. Check if outputs exist
echo ""
if [ -f outputs/scored/pfizer_scored.json ]; then
    echo "[4/4] Pre-computed outputs found — app is ready"
else
    echo "[4/4] No pre-computed outputs — running pipeline..."
    python run_pipeline.py
fi

echo ""
echo "================================"
echo " Setup complete!"
echo ""
echo " Run the app:"
echo "   streamlit run app.py"
echo ""
echo " Or re-run pipeline with LLM:"
echo "   1. Edit .env with your Anthropic key"
echo "   2. python run_pipeline.py"
echo "================================"
