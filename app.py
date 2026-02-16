"""
RiskRadar — Narrative Risk Triage Prototype
Streamlit app for entity selection, narrative browsing, risk scoring, and feedback.
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from collections import Counter

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
FEEDBACK_FILE = OUTPUT_DIR / "feedback.jsonl"

POSTS_FILE = DATA_DIR / "posts.jsonl"
ENTITIES_FILE = DATA_DIR / "entities_seed.csv"
AUTHORS_FILE = DATA_DIR / "authors.csv"
RESOLVED_FILE = OUTPUT_DIR / "resolved_entities.jsonl"
NARRATIVES_DIR = OUTPUT_DIR / "narratives"
SCORED_DIR = OUTPUT_DIR / "scored"


# ── Data Loading (cached) ──────────────────────────────────────────────────

@st.cache_data
def load_entities():
    entities = []
    with open(ENTITIES_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entities.append(row)
    return entities


@st.cache_data
def load_resolved():
    results = []
    with open(RESOLVED_FILE) as f:
        for line in f:
            results.append(json.loads(line))
    return results


@st.cache_data
def load_posts_index():
    index = {}
    with open(POSTS_FILE) as f:
        for line in f:
            p = json.loads(line)
            index[str(p["post_id"])] = p
    return index


@st.cache_data
def load_scored(entity_id):
    path = SCORED_DIR / f"{entity_id}_scored.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


@st.cache_data
def get_entity_stats(entity_id, resolved):
    """Get post count and confidence distribution for an entity."""
    posts = []
    confidences = []
    for r in resolved:
        for e in r.get("resolved_entities", []):
            if e["entity_id"] == entity_id:
                posts.append(r)
                confidences.append(e.get("confidence", 0.95))
    return len(posts), confidences


def save_feedback(feedback_entry):
    """Append feedback to feedback.jsonl."""
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")


# ── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="RiskRadar", page_icon="🔴", layout="wide")
st.title("RiskRadar — Narrative Risk Triage")

entities = load_entities()
resolved = load_resolved()
posts_index = load_posts_index()
entity_ids = sorted(set(e["entity_id"] for e in entities))


# ── Sidebar: Entity Selection (Screen A) ────────────────────────────────────

st.sidebar.header("Entity Selection")

entity_labels = {e["entity_id"]: f"{e['canonical_name']} ({e['entity_type']})" for e in entities}
selected_entity = st.sidebar.selectbox(
    "Select Entity",
    entity_ids,
    format_func=lambda x: entity_labels.get(x, x),
)

# Entity overview stats
post_count, confidences = get_entity_stats(selected_entity, resolved)
st.sidebar.metric("Matched Posts", post_count)

if confidences:
    avg_conf = sum(confidences) / len(confidences)
    st.sidebar.metric("Avg Confidence", f"{avg_conf:.2f}")

    conf_high = sum(1 for c in confidences if c >= 0.8)
    conf_med = sum(1 for c in confidences if 0.5 <= c < 0.8)
    conf_low = sum(1 for c in confidences if c < 0.5)
    st.sidebar.caption(f"High: {conf_high} · Medium: {conf_med} · Low: {conf_low}")

st.sidebar.divider()

# ── Sidebar: Re-run Pipeline ────────────────────────────────────────────────

st.sidebar.header("Re-run Pipeline")
st.sidebar.caption(
    "Provide an Anthropic API key to re-run with LLM-enhanced "
    "entity resolution, topic labeling, and narrative summarization. "
    "Without a key, the pipeline uses fuzzy/keyword fallback."
)

api_key_env = os.getenv("ANTHROPIC_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "Anthropic API Key",
    value=api_key_env,
    type="password",
    placeholder="sk-ant-... (or set in .env)",
)

if st.sidebar.button("Run Full Pipeline", type="primary"):
    api_key = api_key_input if api_key_input else None
    mode = "LLM-enhanced" if api_key else "fuzzy/keyword only"
    st.sidebar.info(f"Running pipeline ({mode})...")

    try:
        from entity_resolution import resolve_entities
        from narrative_clustering import cluster_all_entities
        from risk_scoring import score_all_entities

        with st.spinner("Stage 1/3: Entity Resolution..."):
            resolve_entities(
                posts_file=str(POSTS_FILE),
                entities_file=str(ENTITIES_FILE),
                api_key=api_key,
                output_file=str(RESOLVED_FILE),
            )

        with st.spinner("Stage 2/3: Narrative Clustering..."):
            cluster_all_entities(
                resolved_file=str(RESOLVED_FILE),
                api_key=api_key,
                output_dir=str(NARRATIVES_DIR),
                min_cluster_size=2,
                merge_threshold=55,
            )

        with st.spinner("Stage 3/3: Risk Scoring..."):
            score_all_entities(
                narratives_dir=str(NARRATIVES_DIR),
                posts_file=str(POSTS_FILE),
                authors_file=str(AUTHORS_FILE),
                output_dir=str(SCORED_DIR),
            )

        st.sidebar.success("Pipeline complete! Refresh the page to see updated results.")
        st.cache_data.clear()

    except Exception as e:
        st.sidebar.error(f"Pipeline error: {e}")

st.sidebar.divider()
st.sidebar.caption("RiskRadar Prototype · Built for Lead DS Challenge")


# ── Main: Narrative List (Screen B) ─────────────────────────────────────────

scored = load_scored(selected_entity)

if not scored:
    st.warning(f"No scored narratives found for {selected_entity}.")
    st.stop()

st.subheader(f"Narratives for {entity_labels.get(selected_entity, selected_entity)}")
st.caption(f"{len(scored)} narratives · {post_count} matched posts")

# Narrative table
for i, narr in enumerate(scored):
    score = narr["risk_score"]

    # Color code by risk
    if score >= 70:
        color = "🔴"
    elif score >= 50:
        color = "🟡"
    else:
        color = "🟢"

    conf = narr.get("confidence", "?")
    band = narr.get("confidence_band", [0, 100])
    taxonomy = narr.get("taxonomy_label", "")
    title = narr["title"]

    # Top driver tags
    drivers = narr.get("drivers", [])
    driver_tags = " · ".join(
        f"{d['name']}({d['score']:.0f})" for d in drivers[:3]
    )

    with st.expander(f"{color} **{score:.0f}** — {title}  ·  _{taxonomy}_  ·  {narr['post_count']} posts"):

        # ── Screen C: Narrative Detail ──────────────────────────────

        # Summary
        st.markdown(f"**Summary:** {narr.get('summary', 'N/A')}")

        # Score + Confidence
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", f"{score:.1f} / 100")
        col2.metric("Confidence", conf)
        col3.metric("Band", f"{band[0]:.0f} – {band[1]:.0f}")

        # Driver breakdown
        st.markdown("**Score Drivers:**")
        for d in drivers:
            pct = d["contribution"]
            st.markdown(
                f"- **{d['name']}**: {d['score']:.0f}/100 "
                f"(weight {d['weight']:.0f}× → {pct:.1f} pts)"
            )
            detail = d.get("detail", {})
            if d["name"] == "Language Signals":
                st.caption(
                    f"  Taxonomy base: {detail.get('taxonomy_base_score', '?')} · "
                    f"Negative ratio: {detail.get('negative_ratio', '?')} · "
                    f"High-risk keywords: {detail.get('high_risk_keywords', 0)} · "
                    f"Medium-risk keywords: {detail.get('medium_risk_keywords', 0)}"
                )
            elif d["name"] == "Engagement":
                st.caption(
                    f"  Shares: {detail.get('total_shares', 0)} · "
                    f"Comments: {detail.get('total_comments', 0)} · "
                    f"Likes: {detail.get('total_likes', 0)} · "
                    f"Percentile: {detail.get('percentile', '?')}"
                )
            elif d["name"] == "Volume":
                st.caption(
                    f"  Posts: {detail.get('post_count', '?')} · "
                    f"Percentile: {detail.get('percentile', '?')}"
                )
            elif d["name"] == "Velocity":
                st.caption(
                    f"  Posts/day: {detail.get('posts_per_day', '?')} · "
                    f"Span: {detail.get('span_days', '?')} days · "
                    f"Accelerating: {detail.get('accelerating', '?')}"
                )
            elif d["name"] == "Author Influence":
                st.caption(
                    f"  Max followers: {detail.get('max_followers', '?')} · "
                    f"Unique authors: {detail.get('unique_authors', '?')}"
                )

        # Evidence posts
        evidence = narr.get("evidence_posts", [])
        if evidence:
            st.markdown("**Evidence Posts:**")
            for ep in evidence[:5]:
                url = ep.get("url", "")
                handle = ep.get("handle", "")
                meta = f"👤 {handle}" if handle else ""
                meta += f" · ❤️ {ep.get('likes', 0)} · 🔄 {ep.get('shares', 0)} · 💬 {ep.get('comments', 0)}"
                if ep.get("followers"):
                    meta += f" · 👥 {ep['followers']:,} followers"

                st.markdown(f"> {ep['text'][:250]}...")
                st.caption(meta)
                if url:
                    st.caption(f"[Link]({url})")

        # ── Screen D: Feedback Capture ──────────────────────────────

        st.divider()
        st.markdown("**Feedback:**")

        fcol1, fcol2 = st.columns(2)

        with fcol1:
            risk_feedback = st.radio(
                "Risk rating accurate?",
                ["Correct", "Too High", "Too Low"],
                key=f"risk_{narr['narrative_id']}",
                horizontal=True,
            )
            if st.button("Submit Risk Feedback", key=f"riskbtn_{narr['narrative_id']}"):
                save_feedback({
                    "type": "risk_rating",
                    "narrative_id": narr["narrative_id"],
                    "entity_id": selected_entity,
                    "original_score": score,
                    "feedback": risk_feedback,
                    "timestamp": datetime.now().isoformat(),
                })
                st.success("Risk feedback saved.")

        with fcol2:
            entity_options = ["Correct"] + entity_ids + ["none"]
            entity_correction = st.selectbox(
                "Correct entity?",
                entity_options,
                key=f"entity_{narr['narrative_id']}",
            )
            if st.button("Submit Entity Feedback", key=f"entbtn_{narr['narrative_id']}"):
                save_feedback({
                    "type": "entity_correction",
                    "narrative_id": narr["narrative_id"],
                    "entity_id": selected_entity,
                    "corrected_entity": entity_correction,
                    "timestamp": datetime.now().isoformat(),
                })
                st.success("Entity feedback saved.")


# ── Footer ──────────────────────────────────────────────────────────────────

st.divider()
feedback_count = 0
if FEEDBACK_FILE.exists():
    with open(FEEDBACK_FILE) as f:
        feedback_count = sum(1 for _ in f)

st.caption(
    f"Pipeline: entity_resolution.py → narrative_clustering.py → risk_scoring.py · "
    f"Feedback entries: {feedback_count} · "
    f"Data: {len(resolved)} posts, {len(entity_ids)} entities"
)
