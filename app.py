"""
RiskRadar — Narrative Risk Triage Prototype
Streamlit app for entity selection, narrative browsing, risk scoring, and feedback.
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path

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
OVERRIDES_FILE = OUTPUT_DIR / "overrides.json"

POSTS_FILE = DATA_DIR / "posts.jsonl"
ENTITIES_FILE = DATA_DIR / "entities_seed.csv"
AUTHORS_FILE = DATA_DIR / "authors.csv"
RESOLVED_FILE = OUTPUT_DIR / "resolved_entities.jsonl"
NARRATIVES_DIR = OUTPUT_DIR / "narratives"
SCORED_DIR = OUTPUT_DIR / "scored"


# ── Overrides: Human-in-the-loop corrections ────────────────────────────────

def load_overrides():
    """Load overrides.json — the live file the pipeline reads."""
    if OVERRIDES_FILE.exists():
        with open(OVERRIDES_FILE) as f:
            return json.load(f)
    return {"entity_overrides": {}, "risk_overrides": {}}


def save_overrides(overrides):
    """Write overrides.json — immediately available to pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OVERRIDES_FILE, "w") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False)


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
def load_scored_raw(entity_id):
    """Load raw scored narratives from disk."""
    path = SCORED_DIR / f"{entity_id}_scored.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def load_scored(entity_id):
    """Load scored narratives and apply any overrides.json corrections live."""
    scored = load_scored_raw(entity_id)
    overrides = load_overrides()

    risk_overrides = overrides.get("risk_overrides", {})
    entity_overrides = overrides.get("entity_overrides", {})

    results = []
    for narr in scored:
        n = dict(narr)  # shallow copy

        # Apply risk overrides
        nid = n["narrative_id"]
        if nid in risk_overrides:
            feedback = risk_overrides[nid].get("feedback", "")
            original = n["risk_score"]
            if feedback == "Too High":
                n["risk_score"] = max(0, original - 15)
            elif feedback == "Too Low":
                n["risk_score"] = min(100, original + 15)
            n["analyst_override"] = feedback
            n["original_score_before_override"] = original

        # Check if any posts in this narrative have entity overrides
        overridden_posts = [
            pid for pid in n.get("post_ids", [])
            if str(pid) in entity_overrides
        ]
        if overridden_posts:
            n["has_entity_overrides"] = len(overridden_posts)

        results.append(n)

    # Re-sort by (possibly adjusted) risk score
    results.sort(key=lambda s: -s["risk_score"])
    return results


@st.cache_data
def get_entity_stats(entity_id, resolved):
    """Get post count, confidence distribution, and LLM audit stats for an entity."""
    posts = []
    confidences = []
    llm_audit = {"agrees": 0, "disagrees": 0, "not_audited": 0, "human": 0}
    for r in resolved:
        for e in r.get("resolved_entities", []):
            if e["entity_id"] == entity_id:
                posts.append(r)
                confidences.append(e.get("confidence", 0.95))
                agrees = e.get("llm_agrees")
                if agrees is True:
                    llm_audit["agrees"] += 1
                elif agrees is False:
                    llm_audit["disagrees"] += 1
                elif agrees == "human_override":
                    llm_audit["human"] += 1
                else:
                    llm_audit["not_audited"] += 1
    return len(posts), confidences, llm_audit


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
post_count, confidences, llm_audit = get_entity_stats(selected_entity, resolved)
st.sidebar.metric("Matched Posts", post_count)

if confidences:
    avg_conf = sum(confidences) / len(confidences)
    st.sidebar.metric("Avg Confidence", f"{avg_conf:.2f}")

    conf_high = sum(1 for c in confidences if c >= 0.8)
    conf_med = sum(1 for c in confidences if 0.5 <= c < 0.8)
    conf_low = sum(1 for c in confidences if c < 0.5)
    st.sidebar.caption(f"High: {conf_high} · Medium: {conf_med} · Low: {conf_low}")

# LLM Audit stats (if audit was run)
audited = llm_audit["agrees"] + llm_audit["disagrees"]
if audited > 0:
    rate = llm_audit["agrees"] / audited
    st.sidebar.metric("LLM Audit Agree Rate", f"{rate:.0%}")
    st.sidebar.caption(
        f"Audited: {audited} · Agrees: {llm_audit['agrees']} · "
        f"Disagrees: {llm_audit['disagrees']} · Not audited: {llm_audit['not_audited']}"
    )

st.sidebar.divider()

# ── Sidebar: Risk Taxonomy Reference ───────────────────────────────────────

with st.sidebar.expander("Risk Taxonomy Reference"):
    st.markdown("""
**1. Regulatory / Compliance** — Alleged breaches, fines, investigations, misconduct claims

**2. Financial Integrity** — Fraud, money laundering, market manipulation, mis-selling

**3. Customer Harm** — Poor treatment, unfair practices, discrimination, widespread complaints

**4. Data / Cyber** — Breach claims, leaks, ransomware, insecure systems

**5. Operational Resilience** — Outages, service failure, systemic disruption

**6. Executive / Employee Misconduct** — Leadership scandal, harassment, unethical behaviour

**7. Misinformation / Manipulation** — Coordinated campaigns, synthetic or misleading media narratives
""")

# ── Sidebar: Re-run Pipeline ────────────────────────────────────────────────

st.sidebar.header("Re-run Pipeline")
st.sidebar.caption(
    "Provide an Anthropic API key to re-run with LLM-enhanced "
    "entity resolution, topic labeling, and narrative summarization. "
    "Without a key, the pipeline uses fuzzy/keyword fallback."
)

api_key_env = os.getenv("ANTHROPIC_API_KEY", "")
has_env_key = bool(api_key_env)

if has_env_key:
    st.sidebar.success("API key loaded from .env")
    api_key_input = st.sidebar.text_input(
        "Anthropic API Key (override .env)",
        value="",
        type="password",
        placeholder="Leave blank to use .env key",
    )
else:
    api_key_input = st.sidebar.text_input(
        "Anthropic API Key",
        value="",
        type="password",
        placeholder="sk-ant-... (or set in .env)",
    )

if st.sidebar.button("Run Full Pipeline", type="primary"):
    # Use typed key if provided, else fall back to .env key
    api_key = api_key_input if api_key_input else (api_key_env if has_env_key else None)
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


# ── Main: Cross-Entity Overview ─────────────────────────────────────────────

# Load all scored narratives for overview
all_scored_flat = []
for eid in entity_ids:
    for narr in load_scored(eid):
        narr["_entity_label"] = entity_labels.get(eid, eid)
        all_scored_flat.append(narr)
all_scored_flat.sort(key=lambda n: -n["risk_score"])

# Top risks across all entities
st.subheader("Top Risk Narratives (All Entities)")
top_n = min(10, len(all_scored_flat))
for narr in all_scored_flat[:top_n]:
    s = narr["risk_score"]
    color = "🔴" if s >= 70 else ("🟡" if s >= 50 else "🟢")
    st.markdown(
        f"{color} **{s:.0f}** — {narr['title'][:50]}  ·  "
        f"_{narr['_entity_label']}_  ·  {narr.get('taxonomy_label', '')}  ·  "
        f"{narr['post_count']} posts"
    )

st.divider()

# ── Main: Per-Entity Narrative List (Screen B) ──────────────────────────────

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

        # Override indicators
        if narr.get("analyst_override"):
            orig = narr.get("original_score_before_override", score)
            st.warning(
                f"Analyst override active: '{narr['analyst_override']}' — "
                f"original score was {orig:.0f}, adjusted to {score:.0f}"
            )
        if narr.get("has_entity_overrides"):
            st.info(f"{narr['has_entity_overrides']} post(s) in this narrative have entity corrections pending pipeline re-run")

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

        # Evidence posts with per-post entity feedback
        evidence = narr.get("evidence_posts", [])
        if evidence:
            st.markdown("**Evidence Posts:**")
            for ep_idx, ep in enumerate(evidence[:5]):
                url = ep.get("url", "")
                handle = ep.get("handle", "")
                meta = f"👤 {handle}" if handle else ""
                meta += f" · ❤️ {ep.get('likes', 0)} · 🔄 {ep.get('shares', 0)} · 💬 {ep.get('comments', 0)}"
                if ep.get("followers"):
                    meta += f" · 👥 {ep['followers']:,} followers"

                ep_col1, ep_col2 = st.columns([4, 1])
                with ep_col1:
                    st.markdown(f"> {ep['text'][:250]}...")
                    st.caption(meta)
                    if url:
                        st.caption(f"[Link]({url})")
                with ep_col2:
                    # Per-post entity correction (spec requirement)
                    post_entity_options = ["✓ Correct"] + entity_ids + ["none"]
                    post_correction = st.selectbox(
                        "Entity?",
                        post_entity_options,
                        key=f"postent_{narr['narrative_id']}_{ep['post_id']}_{ep_idx}",
                        label_visibility="collapsed",
                    )
                    if post_correction != "✓ Correct":
                        overrides = load_overrides()
                        overrides.setdefault("entity_overrides", {})[str(ep["post_id"])] = {
                            "original_entity": selected_entity,
                            "corrected_entity": post_correction if post_correction != "none" else None,
                            "timestamp": datetime.now().isoformat(),
                        }
                        save_overrides(overrides)
                        save_feedback({
                            "type": "post_entity_correction",
                            "post_id": ep["post_id"],
                            "narrative_id": narr["narrative_id"],
                            "original_entity": selected_entity,
                            "corrected_entity": post_correction,
                            "timestamp": datetime.now().isoformat(),
                        })
                        st.caption(f"→ {post_correction}")

        # ── Screen D: Feedback Capture ──────────────────────────────

        st.divider()
        st.markdown("**Feedback:**")

        # Load current overrides
        overrides = load_overrides()

        # Show if this narrative already has overrides
        existing_risk = overrides.get("risk_overrides", {}).get(narr["narrative_id"])
        if existing_risk:
            st.info(f"Analyst override active: rated '{existing_risk['feedback']}' (original: {existing_risk['original_score']:.0f})")

        fcol1, fcol2 = st.columns(2)

        with fcol1:
            risk_feedback = st.radio(
                "Risk rating accurate?",
                ["Correct", "Too High", "Too Low"],
                key=f"risk_{narr['narrative_id']}",
                horizontal=True,
            )
            if st.button("Submit Risk Feedback", key=f"riskbtn_{narr['narrative_id']}"):
                entry = {
                    "narrative_id": narr["narrative_id"],
                    "entity_id": selected_entity,
                    "original_score": score,
                    "feedback": risk_feedback,
                    "timestamp": datetime.now().isoformat(),
                }
                # Write to overrides.json (live state)
                if risk_feedback != "Correct":
                    overrides.setdefault("risk_overrides", {})[narr["narrative_id"]] = entry
                else:
                    overrides.get("risk_overrides", {}).pop(narr["narrative_id"], None)
                save_overrides(overrides)
                # Append to feedback.jsonl (audit log)
                save_feedback({"type": "risk_rating", **entry})
                st.cache_data.clear()
                st.success("Saved to overrides.json — pipeline will use this on next run.")

        with fcol2:
            entity_options = ["Correct"] + entity_ids + ["none"]
            entity_correction = st.selectbox(
                "Correct entity?",
                entity_options,
                key=f"entity_{narr['narrative_id']}",
            )
            if st.button("Submit Entity Feedback", key=f"entbtn_{narr['narrative_id']}"):
                entry = {
                    "narrative_id": narr["narrative_id"],
                    "entity_id": selected_entity,
                    "corrected_entity": entity_correction,
                    "timestamp": datetime.now().isoformat(),
                }
                # Write entity overrides keyed by post_ids in this narrative
                if entity_correction != "Correct":
                    for pid in narr.get("post_ids", [])[:50]:
                        overrides.setdefault("entity_overrides", {})[str(pid)] = {
                            "original_entity": selected_entity,
                            "corrected_entity": entity_correction if entity_correction != "none" else None,
                            "timestamp": datetime.now().isoformat(),
                        }
                save_overrides(overrides)
                save_feedback({"type": "entity_correction", **entry})
                st.cache_data.clear()
                st.success("Saved to overrides.json — pipeline will use this on next run.")


# ── Footer ──────────────────────────────────────────────────────────────────

st.divider()
feedback_count = 0
if FEEDBACK_FILE.exists():
    with open(FEEDBACK_FILE) as f:
        feedback_count = sum(1 for _ in f)

final_overrides = load_overrides()
entity_override_count = len(final_overrides.get("entity_overrides", {}))
risk_override_count = len(final_overrides.get("risk_overrides", {}))

st.caption(
    f"Pipeline: entity_resolution.py → narrative_clustering.py → risk_scoring.py · "
    f"Feedback log: {feedback_count} entries · "
    f"Active overrides: {entity_override_count} entity, {risk_override_count} risk · "
    f"Data: {len(resolved)} posts, {len(entity_ids)} entities"
)
