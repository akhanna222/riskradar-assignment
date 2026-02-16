"""
Run the full RiskRadar pipeline: Entity Resolution → Narrative Clustering → Risk Scoring.

Usage:
    python run_pipeline.py                    # Fuzzy-only (no API key)
    python run_pipeline.py --api-key sk-...   # Full hybrid with LLM
"""

import argparse
import os
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

    print("=" * 60)
    print("STAGE 1: Entity Resolution")
    print("=" * 60)
    resolve_entities(
        posts_file=str(data_dir / "posts.jsonl"),
        entities_file=str(data_dir / "entities_seed.csv"),
        api_key=args.api_key,
        limit=args.limit,
        output_file=str(output_dir / "resolved_entities.jsonl"),
    )

    print("\n" + "=" * 60)
    print("STAGE 2: Narrative Clustering")
    print("=" * 60)
    cluster_all_entities(
        resolved_file=str(output_dir / "resolved_entities.jsonl"),
        api_key=args.api_key,
        output_dir=str(output_dir / "narratives"),
        min_cluster_size=2,
        merge_threshold=55,
    )

    print("\n" + "=" * 60)
    print("STAGE 3: Risk Scoring")
    print("=" * 60)
    score_all_entities(
        narratives_dir=str(output_dir / "narratives"),
        posts_file=str(data_dir / "posts.jsonl"),
        authors_file=str(data_dir / "authors.csv"),
        output_dir=str(output_dir / "scored"),
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Outputs in: {output_dir}/")
    print(f"  resolved_entities.jsonl  — entity resolution results")
    print(f"  narratives/              — per-entity narrative clusters")
    print(f"  scored/                  — per-entity risk-scored narratives")
    print(f"\nRun the UI:  streamlit run app.py")


if __name__ == "__main__":
    main()
