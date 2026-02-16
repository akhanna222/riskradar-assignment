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
pip install -r requirements.txt -q 2>/dev/null || pip install -r requirements.txt -q --break-system-packages
echo "      Done."

# 2. Check data files — STOP if missing
echo ""
echo "[2/4] Checking data files..."
mkdir -p data

MISSING=0
for f in data/posts.jsonl data/entities_seed.csv data/authors.csv; do
    if [ ! -f "$f" ]; then
        echo "      ❌ MISSING: $f"
        MISSING=1
    else
        echo "      ✅ $f"
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "  ERROR: Required data files missing."
    echo ""
    echo "  Please place these files in the data/ folder:"
    echo "    data/posts.jsonl         (social media posts)"
    echo "    data/entities_seed.csv   (entity catalog)"
    echo "    data/authors.csv         (author profiles)"
    echo ""
    echo "  Then re-run:  bash setup.sh"
    exit 1
fi

# 3. Set up .env
echo ""
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[3/4] Created .env from .env.example"
else
    echo "[3/4] .env already exists"
fi

# 4. Run tests
echo ""
echo "[4/4] Running tests..."
python tests.py 2>&1 | tail -3

echo ""
echo "================================"
echo " Setup complete!"
echo ""
echo " Launch the app (uses pre-computed outputs):"
echo "   streamlit run app.py"
echo ""
echo " To re-run the pipeline from scratch:"
echo "   ⚠️  This will OVERWRITE the outputs/ folder."
echo "   ⚠️  You need an Anthropic API key for LLM mode."
echo "   1. Edit .env → add your key (https://console.anthropic.com/settings/keys)"
echo "   2. python run_pipeline.py"
echo "================================"
