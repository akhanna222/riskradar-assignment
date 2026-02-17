"""
Run the full RiskRadar pipeline: Entity Resolution → Narrative Clustering → Risk Scoring.

Usage:
    python run_pipeline.py                    # Fuzzy-only (no API key)
    python run_pipeline.py --api-key sk-...   # Full hybrid with LLM
    python run_pipeline.py --limit 50         # Test with fewer posts
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from entity_resolution import resolve_entities
from narrative_clustering import cluster_all_entities
from risk_scoring import score_all_entities


def main():
    parser = argparse.ArgumentParser(description="Run RiskRadar pipeline")
    parser.add_argument("--api-key", default=os.getenv("ANTHROPIC_API_KEY"),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N posts (for testing)")
    args = parser.parse_args()

    data_dir = Path("data")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # ── Validate input data ──
    required = {
        "posts":    data_dir / "posts.jsonl",
        "entities": data_dir / "entities_seed.csv",
        "authors":  data_dir / "authors.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        print(f"ERROR: Missing data files: {', '.join(missing)}")
        print(f"Place them in {data_dir}/:")
        for name, path in required.items():
            status = "✓" if path.exists() else "✗ MISSING"
            print(f"  {status}  {path}")
        sys.exit(1)

    mode = "LLM-enhanced" if args.api_key else "fuzzy/keyword-only"
    print(f"RiskRadar Pipeline — mode: {mode}")
    if args.limit:
        print(f"Limit: first {args.limit} posts")
    pipeline_start = time.time()

    # ── Stage 1: Entity Resolution ──
    print(f"\n{'='*60}")
    print("STAGE 1: Entity Resolution")
    print(f"{'='*60}")
    t0 = time.time()
    resolve_entities(
        posts_file=str(required["posts"]),
        entities_file=str(required["entities"]),
        api_key=args.api_key,
        limit=args.limit,
        output_file=str(output_dir / "resolved_entities.jsonl"),
    )
    print(f"  ⏱  {time.time() - t0:.1f}s")

    # ── Stage 2: Narrative Clustering ──
    print(f"\n{'='*60}")
    print("STAGE 2: Narrative Clustering")
    print(f"{'='*60}")
    t0 = time.time()
    cluster_all_entities(
        resolved_file=str(output_dir / "resolved_entities.jsonl"),
        api_key=args.api_key,
        output_dir=str(output_dir / "narratives"),
        min_cluster_size=2,
    )
    print(f"  ⏱  {time.time() - t0:.1f}s")

    # ── Stage 3: Risk Scoring ──
    print(f"\n{'='*60}")
    print("STAGE 3: Risk Scoring")
    print(f"{'='*60}")
    t0 = time.time()
    score_all_entities(
        narratives_dir=str(output_dir / "narratives"),
        posts_file=str(required["posts"]),
        authors_file=str(required["authors"]),
        output_dir=str(output_dir / "scored"),
    )
    print(f"  ⏱  {time.time() - t0:.1f}s")

    # ── Done ──
    total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE ({total:.1f}s)")
    print(f"{'='*60}")
    print(f"Outputs in: {output_dir}/")
    print(f"  resolved_entities.jsonl  — entity resolution results")
    print(f"  narratives/              — per-entity narrative clusters")
    print(f"  scored/                  — per-entity risk-scored narratives")
    print(f"\nRun the UI:  streamlit run app.py")


if __name__ == "__main__":
    main()
