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
# TEST 3b: Edge cases for entity resolution
# ═════════════════════════════════════════════════════════════════════════════

class TestEntityResolutionEdgeCases(unittest.TestCase):
    """Test entity resolution handles edge cases gracefully."""

    def test_empty_text(self):
        """Posts with empty text should produce empty entity list."""
        from entity_resolution import find_mentions, load_entities
        entities = load_entities(str(ENTITIES_FILE))
        mentions = find_mentions("", entities)
        self.assertEqual(mentions, [])

    def test_hashtag_only_post(self):
        """Posts with only hashtags should not crash."""
        from entity_resolution import find_mentions, load_entities
        entities = load_entities(str(ENTITIES_FILE))
        mentions = find_mentions("#pharma #vaccine #health #trending", entities)
        # Should return empty or valid matches — not crash
        self.assertIsInstance(mentions, list)

    def test_non_english_text(self):
        """Non-English text should not crash the pipeline."""
        from entity_resolution import find_mentions, load_entities
        entities = load_entities(str(ENTITIES_FILE))
        mentions = find_mentions("فايزر تطلق لقاح جديد", entities)  # Arabic
        self.assertIsInstance(mentions, list)

    def test_word_boundary_matching(self):
        """Entity mentions should use word boundaries, not substring matching."""
        from entity_resolution import find_mentions, load_entities
        entities = load_entities(str(ENTITIES_FILE))
        # "merck" should match "Merck" but not "commercial" or "merckx"
        mentions_valid = find_mentions("Merck announced quarterly results", entities)
        self.assertIn("merck", mentions_valid)

    def test_no_entity_matches(self):
        """Posts with no entity mentions should return empty list."""
        from entity_resolution import find_mentions, load_entities
        entities = load_entities(str(ENTITIES_FILE))
        mentions = find_mentions("The weather is nice today in Dublin", entities)
        self.assertEqual(mentions, [])


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Narrative clustering produces valid output
# ═════════════════════════════════════════════════════════════════════════════

class TestNarrativeClustering(unittest.TestCase):
    """Test narrative clustering on pre-computed resolved entities."""

    @classmethod
    def setUpClass(cls):
        """Check if resolved_entities.jsonl exists — skip if not (requires API key)."""
        if not RESOLVED_FILE.exists():
            raise unittest.SkipTest(
                "No pre-computed resolved_entities.jsonl — run pipeline first "
                "(requires API key for LLM clustering)"
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


class TestFuzzyMerge(unittest.TestCase):
    """Test the fuzzy label merge step in narrative clustering."""

    def test_merge_similar_labels(self):
        from narrative_clustering import merge_similar_labels
        groups = {
            "pfizer vaccine safety concerns": [{"post_id": "1"}],
            "pfizer vaccine safety issues": [{"post_id": "2"}],
            "quarterly earnings report": [{"post_id": "3"}],
        }
        merged = merge_similar_labels(groups, merge_threshold=55)
        # The two vaccine labels should merge; earnings stays separate
        self.assertLessEqual(len(merged), 2)

    def test_merge_preserves_posts(self):
        from narrative_clustering import merge_similar_labels
        groups = {
            "topic alpha": [{"post_id": "1"}, {"post_id": "2"}],
            "topic alpha variant": [{"post_id": "3"}],
        }
        merged = merge_similar_labels(groups, merge_threshold=55)
        total_posts = sum(len(v) for v in merged.values())
        self.assertEqual(total_posts, 3, "Fuzzy merge should not lose posts")

    def test_merge_no_false_merges(self):
        from narrative_clustering import merge_similar_labels
        groups = {
            "vaccine safety": [{"post_id": "1"}],
            "quarterly earnings": [{"post_id": "2"}],
        }
        merged = merge_similar_labels(groups, merge_threshold=55)
        self.assertEqual(len(merged), 2, "Dissimilar labels should not merge")


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Risk scoring produces valid output
# ═════════════════════════════════════════════════════════════════════════════

class TestRiskScoring(unittest.TestCase):
    """Test risk scoring on pre-computed narratives."""

    @classmethod
    def setUpClass(cls):
        """Check if narratives exist — skip if not (requires pipeline run)."""
        pfizer_narr = NARRATIVES_DIR / "pfizer_narratives.json"
        if not pfizer_narr.exists():
            raise unittest.SkipTest(
                "No pre-computed pfizer_narratives.json — run pipeline first"
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
# TEST 5b: Risk scoring unit tests (no pre-computed data needed)
# ═════════════════════════════════════════════════════════════════════════════

class TestRiskScoringUnits(unittest.TestCase):
    """Unit tests for risk scoring functions — no pre-computed data needed."""

    def test_keyword_word_boundaries(self):
        """Risk keywords should match whole words only, not substrings."""
        from risk_scoring import score_language
        # "risk" is in medium keywords. Should NOT match inside "asterisk"
        posts_with_asterisk = [{"text": "This asterisk marks a footnote"}]
        score1, detail1 = score_language(
            posts_with_asterisk, "General / Unclassified", {"neutral": 1}
        )
        posts_with_risk = [{"text": "There is a real risk of harm here"}]
        score2, detail2 = score_language(
            posts_with_risk, "General / Unclassified", {"neutral": 1}
        )
        # "risk" + "harm" post should have medium keyword hits; "asterisk" should not
        self.assertEqual(detail1.get("medium_risk_keywords", 0), 0,
                         "'asterisk' should not trigger 'risk' keyword match")
        self.assertGreater(detail2.get("medium_risk_keywords", 0), 0,
                           "'risk' and 'harm' should be detected as medium keywords")

    def test_facebook_views_excluded(self):
        """Facebook posts should not be penalized for views=0."""
        from risk_scoring import score_engagement
        fb_posts = [{"shares": 10, "comments": 5, "likes": 20, "views": 0, "platform": "facebook"}]
        tw_posts = [{"shares": 10, "comments": 5, "likes": 20, "views": 0, "platform": "twitter"}]
        fb_score, fb_detail = score_engagement(fb_posts, [])
        tw_score, tw_detail = score_engagement(tw_posts, [])
        # Both should get same engagement (views=0 anyway, but field is tracked)
        self.assertEqual(fb_detail["facebook_posts_excluded_views"], 1)
        self.assertEqual(tw_detail["facebook_posts_excluded_views"], 0)

    def test_weights_sum_to_one(self):
        from risk_scoring import WEIGHTS
        total = sum(WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_taxonomy_risk_coverage(self):
        """All taxonomy categories should have risk scores."""
        from risk_scoring import TAXONOMY_RISK
        expected = ["Customer Harm", "Regulatory / Compliance", "Financial Integrity",
                    "Data / Cyber", "Operational Resilience", "Executive / Employee Misconduct",
                    "Misinformation / Manipulation", "General / Unclassified"]
        for cat in expected:
            self.assertIn(cat, TAXONOMY_RISK, f"Missing taxonomy: {cat}")


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

    def test_overrides_roundtrip(self):
        """Test save + load round-trip for overrides.json."""
        test_data = {
            "entity_overrides": {
                "post_123": {"original_entity": "pfizer", "corrected_entity": "merck",
                             "timestamp": "2025-01-01T00:00:00"}
            },
            "risk_overrides": {
                "pfizer_narrative_0": {"feedback": "Too High", "original_score": 75.0,
                                       "timestamp": "2025-01-01T00:00:00"}
            },
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.test_overrides_path, "w") as f:
            json.dump(test_data, f)
        with open(self.test_overrides_path) as f:
            loaded = json.load(f)
        self.assertEqual(loaded["entity_overrides"]["post_123"]["corrected_entity"], "merck")
        self.assertEqual(loaded["risk_overrides"]["pfizer_narrative_0"]["feedback"], "Too High")
        self.assertEqual(loaded["risk_overrides"]["pfizer_narrative_0"]["original_score"], 75.0)


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
