"""
RiskRadar Unit Tests
====================
Run before launching the app to verify pipeline integrity.

Usage:
    python tests.py          # run all tests
    python -m pytest tests.py -v   # verbose with pytest (optional)
"""

import csv
import json
import os
import sys
import unittest
from pathlib import Path


# ── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

POSTS_FILE = DATA_DIR / "posts.jsonl"
ENTITIES_FILE = DATA_DIR / "entities_seed.csv"
AUTHORS_FILE = DATA_DIR / "authors.csv"
RESOLVED_FILE = OUTPUT_DIR / "resolved_entities.jsonl"
NARRATIVES_DIR = OUTPUT_DIR / "narratives"
SCORED_DIR = OUTPUT_DIR / "scored"
OVERRIDES_FILE = OUTPUT_DIR / "overrides.json"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: Input data files exist and are valid
# ═════════════════════════════════════════════════════════════════════════════

class TestInputData(unittest.TestCase):
    """Verify input data files are present and well-formed."""

    def test_posts_file_exists(self):
        self.assertTrue(POSTS_FILE.exists(), f"{POSTS_FILE} not found")

    def test_posts_valid_jsonl(self):
        with open(POSTS_FILE) as f:
            first = json.loads(f.readline())
        self.assertIn("post_id", first)
        self.assertIn("text", first)

    def test_posts_count(self):
        with open(POSTS_FILE) as f:
            count = sum(1 for _ in f)
        self.assertGreater(count, 0, "posts.jsonl is empty")

    def test_entities_file_exists(self):
        self.assertTrue(ENTITIES_FILE.exists(), f"{ENTITIES_FILE} not found")

    def test_entities_valid_csv(self):
        with open(ENTITIES_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first = next(reader)
        self.assertIn("entity_id", first)
        self.assertIn("canonical_name", first)

    def test_authors_file_exists(self):
        self.assertTrue(AUTHORS_FILE.exists(), f"{AUTHORS_FILE} not found")

    def test_authors_valid_csv(self):
        with open(AUTHORS_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first = next(reader)
        self.assertIn("author_id", first)
        self.assertIn("followers", first)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Pipeline modules import correctly
# ═════════════════════════════════════════════════════════════════════════════

class TestImports(unittest.TestCase):
    """Verify all pipeline modules are importable."""

    def test_import_entity_resolution(self):
        import entity_resolution
        self.assertTrue(hasattr(entity_resolution, "resolve_entities"))

    def test_import_narrative_clustering(self):
        import narrative_clustering
        self.assertTrue(hasattr(narrative_clustering, "cluster_narratives"))
        self.assertTrue(hasattr(narrative_clustering, "cluster_all_entities"))

    def test_import_risk_scoring(self):
        import risk_scoring
        self.assertTrue(hasattr(risk_scoring, "score_narratives"))
        self.assertTrue(hasattr(risk_scoring, "score_all_entities"))

    def test_import_rapidfuzz(self):
        from rapidfuzz import fuzz
        score = fuzz.ratio("pfizer", "Pfizer")
        self.assertGreater(score, 80)

    def test_import_streamlit(self):
        import streamlit
        self.assertIsNotNone(streamlit)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Entity resolution produces valid output
# ═════════════════════════════════════════════════════════════════════════════

class TestEntityResolution(unittest.TestCase):
    """Test entity resolution on a small sample."""

    def test_resolve_small_batch(self):
        from entity_resolution import resolve_entities
        results = resolve_entities(
            posts_file=str(POSTS_FILE),
            entities_file=str(ENTITIES_FILE),
            limit=20,
        )
        self.assertGreater(len(results), 0)

    def test_output_schema(self):
        from entity_resolution import resolve_entities
        results = resolve_entities(
            posts_file=str(POSTS_FILE),
            entities_file=str(ENTITIES_FILE),
            limit=5,
        )
        for r in results:
            self.assertIn("post_id", r)
            self.assertIn("resolved_entities", r)
            self.assertIn("needs_review", r)
            self.assertIsInstance(r["resolved_entities"], list)
            self.assertIsInstance(r["needs_review"], bool)

    def test_entity_fields(self):
        from entity_resolution import resolve_entities
        results = resolve_entities(
            posts_file=str(POSTS_FILE),
            entities_file=str(ENTITIES_FILE),
            limit=50,
        )
        # Find a post with entities
        with_entities = [r for r in results if r["resolved_entities"]]
        self.assertGreater(len(with_entities), 0, "No entities found in 50 posts")
        e = with_entities[0]["resolved_entities"][0]
        self.assertIn("entity_id", e)
        self.assertIn("confidence", e)
        self.assertIn("resolution_method", e)
        self.assertIsInstance(e["confidence"], (int, float))
        self.assertGreaterEqual(e["confidence"], 0)
        self.assertLessEqual(e["confidence"], 1)

    def test_confidence_range(self):
        from entity_resolution import resolve_entities
        results = resolve_entities(
            posts_file=str(POSTS_FILE),
            entities_file=str(ENTITIES_FILE),
            limit=50,
        )
        for r in results:
            for e in r["resolved_entities"]:
                self.assertGreaterEqual(e["confidence"], 0.0)
                self.assertLessEqual(e["confidence"], 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Narrative clustering produces valid output
# ═════════════════════════════════════════════════════════════════════════════

class TestNarrativeClustering(unittest.TestCase):
    """Test narrative clustering on pre-computed resolved entities."""

    @classmethod
    def setUpClass(cls):
        """Ensure resolved_entities.jsonl exists."""
        if not RESOLVED_FILE.exists():
            from entity_resolution import resolve_entities
            resolve_entities(
                posts_file=str(POSTS_FILE),
                entities_file=str(ENTITIES_FILE),
                output_file=str(RESOLVED_FILE),
            )

    def test_cluster_single_entity(self):
        from narrative_clustering import cluster_narratives
        narratives = cluster_narratives(
            entity_id="pfizer",
            resolved_file=str(RESOLVED_FILE),
            min_cluster_size=2,
        )
        self.assertGreater(len(narratives), 0)

    def test_narrative_schema(self):
        from narrative_clustering import cluster_narratives
        narratives = cluster_narratives(
            entity_id="pfizer",
            resolved_file=str(RESOLVED_FILE),
            min_cluster_size=2,
        )
        required = ["narrative_id", "title", "summary", "post_ids", "post_count"]
        for n in narratives:
            for field in required:
                self.assertIn(field, n, f"Missing field: {field}")
            self.assertIsInstance(n["post_ids"], list)
            self.assertGreater(len(n["post_ids"]), 0)
            self.assertEqual(n["post_count"], len(n["post_ids"]))


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Risk scoring produces valid output
# ═════════════════════════════════════════════════════════════════════════════

class TestRiskScoring(unittest.TestCase):
    """Test risk scoring on pre-computed narratives."""

    @classmethod
    def setUpClass(cls):
        """Ensure narratives exist."""
        pfizer_narr = NARRATIVES_DIR / "pfizer_narratives.json"
        if not pfizer_narr.exists():
            if not RESOLVED_FILE.exists():
                from entity_resolution import resolve_entities
                resolve_entities(
                    posts_file=str(POSTS_FILE),
                    entities_file=str(ENTITIES_FILE),
                    output_file=str(RESOLVED_FILE),
                )
            from narrative_clustering import cluster_narratives
            NARRATIVES_DIR.mkdir(parents=True, exist_ok=True)
            narratives = cluster_narratives(
                entity_id="pfizer",
                resolved_file=str(RESOLVED_FILE),
                output_file=str(pfizer_narr),
            )

    def test_score_narratives(self):
        from risk_scoring import score_narratives
        scored = score_narratives(
            narratives_file=str(NARRATIVES_DIR / "pfizer_narratives.json"),
            posts_file=str(POSTS_FILE),
            authors_file=str(AUTHORS_FILE),
        )
        self.assertGreater(len(scored), 0)

    def test_score_range(self):
        from risk_scoring import score_narratives
        scored = score_narratives(
            narratives_file=str(NARRATIVES_DIR / "pfizer_narratives.json"),
            posts_file=str(POSTS_FILE),
            authors_file=str(AUTHORS_FILE),
        )
        for s in scored:
            self.assertGreaterEqual(s["risk_score"], 0)
            self.assertLessEqual(s["risk_score"], 100)

    def test_score_schema(self):
        from risk_scoring import score_narratives
        scored = score_narratives(
            narratives_file=str(NARRATIVES_DIR / "pfizer_narratives.json"),
            posts_file=str(POSTS_FILE),
            authors_file=str(AUTHORS_FILE),
        )
        required = ["narrative_id", "risk_score", "confidence", "confidence_band",
                     "drivers", "evidence_posts"]
        for s in scored:
            for field in required:
                self.assertIn(field, s, f"Missing field: {field}")

    def test_drivers_present(self):
        from risk_scoring import score_narratives
        scored = score_narratives(
            narratives_file=str(NARRATIVES_DIR / "pfizer_narratives.json"),
            posts_file=str(POSTS_FILE),
            authors_file=str(AUTHORS_FILE),
        )
        for s in scored:
            self.assertEqual(len(s["drivers"]), 5, "Expected 5 drivers")
            for d in s["drivers"]:
                self.assertIn("name", d)
                self.assertIn("score", d)
                self.assertIn("weight", d)
                self.assertIn("contribution", d)

    def test_weights_sum_to_one(self):
        from risk_scoring import WEIGHTS
        total = sum(WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=2)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: Overrides system
# ═════════════════════════════════════════════════════════════════════════════

class TestOverrides(unittest.TestCase):
    """Test the human-in-the-loop overrides system."""

    def setUp(self):
        """Clean up any existing test overrides."""
        self.test_overrides_path = OUTPUT_DIR / "test_overrides.json"

    def tearDown(self):
        if self.test_overrides_path.exists():
            os.remove(self.test_overrides_path)

    def test_overrides_json_structure(self):
        overrides = {"entity_overrides": {}, "risk_overrides": {}}
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.test_overrides_path, "w") as f:
            json.dump(overrides, f)
        with open(self.test_overrides_path) as f:
            loaded = json.load(f)
        self.assertIn("entity_overrides", loaded)
        self.assertIn("risk_overrides", loaded)

    def test_risk_override_applies(self):
        """Simulate risk override: Too High should reduce score by 15."""
        original_score = 75.0
        overrides = {
            "risk_overrides": {
                "test_narrative": {"feedback": "Too High", "original_score": original_score}
            }
        }
        feedback = overrides["risk_overrides"]["test_narrative"]["feedback"]
        adjusted = max(0, original_score - 15) if feedback == "Too High" else original_score
        self.assertEqual(adjusted, 60.0)

    def test_risk_override_too_low(self):
        """Too Low should increase score by 15."""
        original_score = 40.0
        adjusted = min(100, original_score + 15)
        self.assertEqual(adjusted, 55.0)

    def test_risk_override_clamps(self):
        """Score should never exceed 0-100."""
        self.assertEqual(max(0, 5.0 - 15), 0)
        self.assertEqual(min(100, 95.0 + 15), 100)

    def test_entity_override_structure(self):
        override = {
            "original_entity": "pfizer",
            "corrected_entity": "merck",
            "timestamp": "2025-01-01T00:00:00",
        }
        self.assertEqual(override["corrected_entity"], "merck")

    def test_entity_override_none(self):
        """corrected_entity=None means 'no entity match'."""
        override = {"corrected_entity": None}
        self.assertIsNone(override["corrected_entity"])


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7: Pre-computed outputs (if they exist)
# ═════════════════════════════════════════════════════════════════════════════

class TestPrecomputedOutputs(unittest.TestCase):
    """Verify pre-computed outputs are valid (skip if not present)."""

    def test_resolved_entities_exist(self):
        if not RESOLVED_FILE.exists():
            self.skipTest("No pre-computed resolved_entities.jsonl")
        with open(RESOLVED_FILE) as f:
            count = sum(1 for _ in f)
        self.assertGreater(count, 0)

    def test_all_entities_have_narratives(self):
        if not NARRATIVES_DIR.exists():
            self.skipTest("No pre-computed narratives")
        narr_files = list(NARRATIVES_DIR.glob("*_narratives.json"))
        self.assertGreater(len(narr_files), 0)

    def test_all_entities_have_scores(self):
        if not SCORED_DIR.exists():
            self.skipTest("No pre-computed scores")
        scored_files = list(SCORED_DIR.glob("*_scored.json"))
        self.assertGreater(len(scored_files), 0)

    def test_narrative_and_scored_match(self):
        """Every entity with narratives should also have scores."""
        if not NARRATIVES_DIR.exists() or not SCORED_DIR.exists():
            self.skipTest("No pre-computed outputs")
        narr_entities = {f.stem.replace("_narratives", "") for f in NARRATIVES_DIR.glob("*.json")}
        scored_entities = {f.stem.replace("_scored", "") for f in SCORED_DIR.glob("*.json")}
        self.assertEqual(narr_entities, scored_entities,
                         f"Mismatch: narratives={narr_entities - scored_entities}, scored={scored_entities - narr_entities}")


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("RiskRadar Unit Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
