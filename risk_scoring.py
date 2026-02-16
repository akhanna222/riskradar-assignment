"""
Narrative Risk Scoring Pipeline
================================================================================

PURPOSE:
    For each narrative (cluster of posts about an entity), compute a risk score
    (0-100) reflecting the likelihood of reputational harm. Every score comes
    with a structured explanation: top drivers, evidence posts, and confidence.

SCORING MODEL:
    risk_score = (
        0.20 * volume_score      +   # How many posts — is this a big conversation?
        0.15 * velocity_score    +   # How fast is it growing?
        0.20 * engagement_score  +   # Likes/shares/comments — is it amplified?
        0.15 * author_score      +   # High-follower accounts driving this?
        0.30 * language_score        # Taxonomy + sentiment — is the content harmful?
    )

    Language gets highest weight (0.30) because a narrative about "CEO fraud"
    is inherently riskier than "nice earnings" regardless of volume.

WHY THESE WEIGHTS:
    - Language (0.30): Content IS the risk. "Side effects killed my mother" is
      dangerous at any volume. Taxonomy category is the strongest signal.
    - Volume (0.20): More posts = more visibility = more risk. But volume alone
      can be gamed by bots, so it's not the top weight.
    - Engagement (0.20): Shares/comments indicate real amplification, not just
      posting. High engagement means the narrative is resonating.
    - Velocity (0.15): A spike matters — 50 posts in 1 day is scarier than
      50 posts over 30 days. But velocity is noisy with small samples.
    - Author (0.15): One post from a 1M-follower journalist outweighs 100
      posts from anonymous accounts. But follower data is often missing.

NORMALIZATION:
    Each sub-score is 0-100, computed using percentile rank across ALL
    narratives in the dataset. This avoids arbitrary thresholds — we're saying
    "this narrative's engagement is in the 85th percentile of all narratives."

    Percentile rank formula: score = (rank / total_narratives) * 100

    Language score uses a different approach: direct mapping from taxonomy
    category and sentiment ratio, since these have inherent risk meaning
    regardless of relative position.

EXPLAINABILITY (per narrative):
    {
        "risk_score": 73,
        "confidence": "medium",
        "confidence_band": [65, 81],
        "drivers": [
            {"name": "Language Signals", "score": 87, "weight": 0.30,
             "detail": "Taxonomy: Customer Harm. 72% negative sentiment. ..."},
            {"name": "Volume", "score": 65, "weight": 0.20,
             "detail": "25 posts (75th percentile). ..."},
            ...
        ],
        "evidence_posts": [top 5 posts by engagement * risk signal],
        "audit_trail": {full computation details}
    }

STEP-BY-STEP:

    Step 1: LOAD DATA — Join posts.jsonl + authors.csv + narratives
    Step 2: COMPUTE SUB-SCORES — Volume, velocity, engagement, author, language
    Step 3: COMBINE — Weighted sum → risk_score (0-100)
    Step 4: EXPLAIN — Top drivers + evidence posts + confidence band
    Step 5: OUTPUT — Scored narratives sorted by risk_score descending

USAGE:
    from risk_scoring import score_narratives

    scored = score_narratives(
        narratives_file="narratives/pfizer_narratives.json",
        posts_file="posts.jsonl",
        authors_file="authors.csv",
    )

    # Or score all entities at once
    from risk_scoring import score_all_entities
    all_scored = score_all_entities(
        narratives_dir="narratives/",
        posts_file="posts.jsonl",
        authors_file="authors.csv",
    )

DEPENDENCIES:
    Required: none (pure Python + stdlib)
"""

import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTS — EXPLICIT AND AUDITABLE
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS = {
    "volume":     0.20,
    "velocity":   0.15,
    "engagement": 0.20,
    "author":     0.15,
    "language":   0.30,
}

# Engagement weights: shares amplify reach more than likes
ENGAGEMENT_WEIGHTS = {"shares": 3.0, "comments": 2.0, "likes": 1.0, "views": 0.1}

# Taxonomy risk tiers: how inherently risky is each category
TAXONOMY_RISK = {
    "Customer Harm":                    90,
    "Regulatory / Compliance":          85,
    "Financial Integrity":              75,
    "Executive / Employee Misconduct":  80,
    "Data / Cyber":                     85,
    "Operational Resilience":           70,
    "Misinformation / Manipulation":    80,
    "General / Unclassified":           20,
}

# Language risk keywords — presence amplifies language score
RISK_KEYWORDS = {
    "high": ["death", "died", "killed", "fraud", "lawsuit", "sued", "arrested",
             "investigation", "recalled", "ban", "scandal", "criminal", "indicted",
             "whistleblower", "coverup", "victim", "hospitalized", "emergency"],
    "medium": ["adverse", "side effect", "risk", "harm", "injury", "complaint",
               "safety", "contaminated", "toxic", "resign", "misconduct",
               "conspiracy", "censored", "manipulation", "breach"],
    "low": ["concern", "question", "worry", "issue", "problem", "trouble",
            "warning", "alert", "caution", "review"],
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD + JOIN DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_posts_index(posts_file):
    """Load posts into a dict keyed by post_id for fast lookup."""
    index = {}
    with open(posts_file) as f:
        for line in f:
            p = json.loads(line)
            index[str(p["post_id"])] = p
    return index


def load_authors_index(authors_file):
    """Load authors into a dict keyed by author_id."""
    index = {}
    with open(authors_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = str(row["author_id"])
            followers = 0
            try:
                followers = int(row.get("followers", 0) or 0)
            except (ValueError, TypeError):
                pass
            index[aid] = {
                "author_id": aid,
                "handle": row.get("handle", ""),
                "followers": followers,
            }
    return index


def enrich_narrative_posts(narrative, posts_index, authors_index):
    """Join narrative post_ids with full post data + author data."""
    enriched = []
    for pid in narrative["post_ids"]:
        post = posts_index.get(str(pid))
        if not post:
            continue
        author = authors_index.get(str(post.get("author_id", "")), {})
        enriched.append({
            "post_id": str(pid),
            "text": post.get("text", "")[:500],
            "created_at": post.get("created_at", ""),
            "platform": post.get("platform", ""),
            "author_id": post.get("author_id", ""),
            "likes": post.get("likes", 0) or 0,
            "shares": post.get("shares", 0) or 0,
            "comments": post.get("comments", 0) or 0,
            "views": post.get("views", 0) or 0,
            "url": post.get("url", ""),
            "followers": author.get("followers", 0),
            "handle": author.get("handle", ""),
        })
    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: COMPUTE SUB-SCORES
# ══════════════════════════════════════════════════════════════════════════════

def score_volume(posts, all_narrative_sizes):
    """
    Volume score based on percentile rank of post count.
    More posts = higher volume score.
    """
    count = len(posts)
    if not all_narrative_sizes:
        return 0, {"post_count": count}

    rank = sum(1 for s in all_narrative_sizes if s <= count)
    score = (rank / len(all_narrative_sizes)) * 100
    return min(score, 100), {
        "post_count": count,
        "percentile": round(score, 1),
    }


def score_velocity(posts):
    """
    Velocity score based on how concentrated posts are in time.
    All posts in 1 day = 100. Spread over 30 days = lower.
    Also detects acceleration (posts speeding up).
    """
    if len(posts) < 2:
        return 0, {"reason": "insufficient posts for velocity"}

    timestamps = []
    for p in posts:
        try:
            ts = datetime.fromisoformat(p["created_at"].replace("Z", "+00:00"))
            timestamps.append(ts)
        except (ValueError, KeyError):
            continue

    if len(timestamps) < 2:
        return 0, {"reason": "insufficient timestamps"}

    timestamps.sort()
    span = (timestamps[-1] - timestamps[0]).total_seconds()
    span_days = max(span / 86400, 0.01)  # avoid division by zero
    posts_per_day = len(timestamps) / span_days

    # Score: higher posts_per_day = higher velocity
    # 10+ posts/day = score ~90, 1 post/day = ~30, 0.1 posts/day = ~5
    velocity_raw = min(posts_per_day / 10, 1.0) * 80

    # Acceleration bonus: are recent posts faster than older?
    if len(timestamps) >= 4:
        mid = len(timestamps) // 2
        first_half_span = (timestamps[mid] - timestamps[0]).total_seconds() / 86400
        second_half_span = (timestamps[-1] - timestamps[mid]).total_seconds() / 86400
        first_rate = mid / max(first_half_span, 0.01)
        second_rate = (len(timestamps) - mid) / max(second_half_span, 0.01)
        if second_rate > first_rate * 1.5:
            velocity_raw += 20  # accelerating

    score = min(velocity_raw, 100)
    return score, {
        "posts_per_day": round(posts_per_day, 2),
        "span_days": round(span_days, 1),
        "accelerating": score > velocity_raw - 20,
    }


def score_engagement(posts, all_narrative_engagements):
    """
    Engagement score based on weighted engagement metrics.
    shares × 3 + comments × 2 + likes × 1 + views × 0.1
    Percentile ranked against all narratives.
    """
    total_weighted = 0
    for p in posts:
        total_weighted += (
            p.get("shares", 0) * ENGAGEMENT_WEIGHTS["shares"]
            + p.get("comments", 0) * ENGAGEMENT_WEIGHTS["comments"]
            + p.get("likes", 0) * ENGAGEMENT_WEIGHTS["likes"]
            + p.get("views", 0) * ENGAGEMENT_WEIGHTS["views"]
        )

    avg_weighted = total_weighted / max(len(posts), 1)

    # Percentile rank
    if all_narrative_engagements:
        rank = sum(1 for e in all_narrative_engagements if e <= total_weighted)
        score = (rank / len(all_narrative_engagements)) * 100
    else:
        score = 50

    total_shares = sum(p.get("shares", 0) for p in posts)
    total_comments = sum(p.get("comments", 0) for p in posts)
    total_likes = sum(p.get("likes", 0) for p in posts)
    total_views = sum(p.get("views", 0) for p in posts)

    return min(score, 100), {
        "total_weighted_engagement": round(total_weighted, 1),
        "avg_per_post": round(avg_weighted, 1),
        "total_shares": total_shares,
        "total_comments": total_comments,
        "total_likes": total_likes,
        "total_views": total_views,
        "percentile": round(score, 1),
    }


def score_author(posts, all_narrative_author_scores):
    """
    Author influence score based on follower counts.
    max_followers: one influential author can drive everything.
    unique_authors: broader spread = harder to contain.
    """
    followers = [p.get("followers", 0) for p in posts]
    followers_nonzero = [f for f in followers if f > 0]

    if not followers_nonzero:
        return 20, {
            "reason": "no follower data available",
            "unique_authors": len(set(p.get("author_id", "") for p in posts)),
        }

    max_f = max(followers_nonzero)
    median_f = sorted(followers_nonzero)[len(followers_nonzero) // 2]
    unique = len(set(p.get("author_id", "") for p in posts))

    # Composite: 60% max followers (log scale), 20% median, 20% unique count
    max_score = min(math.log10(max(max_f, 1)) / 7 * 100, 100)  # 10M followers = 100
    median_score = min(math.log10(max(median_f, 1)) / 5 * 100, 100)
    unique_score = min(unique / 20 * 100, 100)  # 20+ authors = 100

    raw = 0.6 * max_score + 0.2 * median_score + 0.2 * unique_score

    # Percentile rank
    if all_narrative_author_scores:
        rank = sum(1 for s in all_narrative_author_scores if s <= raw)
        score = (rank / len(all_narrative_author_scores)) * 100
    else:
        score = raw

    return min(score, 100), {
        "max_followers": max_f,
        "median_followers": median_f,
        "unique_authors": unique,
        "raw_score": round(raw, 1),
        "percentile": round(score, 1),
    }


def score_language(posts, taxonomy_label, sentiment_dist):
    """
    Language risk score from taxonomy category + sentiment + keyword density.
    This is NOT percentile-ranked — taxonomy has inherent risk meaning.
    """
    # Base score from taxonomy category
    taxonomy_score = TAXONOMY_RISK.get(taxonomy_label, 30)

    # Sentiment modifier: more negative = higher risk
    total_sent = sum(sentiment_dist.values()) or 1
    neg_ratio = sentiment_dist.get("negative", 0) / total_sent
    pos_ratio = sentiment_dist.get("positive", 0) / total_sent
    # Range: -20 (all positive) to +10 (all negative)
    sentiment_modifier = (neg_ratio * 10) - (pos_ratio * 20)

    # Keyword density: count risk keywords in posts
    all_text = " ".join(p.get("text", "").lower() for p in posts)
    high_hits = sum(1 for kw in RISK_KEYWORDS["high"] if kw in all_text)
    med_hits = sum(1 for kw in RISK_KEYWORDS["medium"] if kw in all_text)
    low_hits = sum(1 for kw in RISK_KEYWORDS["low"] if kw in all_text)

    keyword_density = min((high_hits * 3 + med_hits * 2 + low_hits * 1) / max(len(posts), 1), 10)
    keyword_modifier = keyword_density * 2  # 0 to 20

    score = max(0, min(100, taxonomy_score + sentiment_modifier + keyword_modifier))

    return score, {
        "taxonomy_label": taxonomy_label,
        "taxonomy_base_score": taxonomy_score,
        "negative_ratio": round(neg_ratio, 2),
        "positive_ratio": round(pos_ratio, 2),
        "sentiment_modifier": round(sentiment_modifier, 1),
        "high_risk_keywords": high_hits,
        "medium_risk_keywords": med_hits,
        "low_risk_keywords": low_hits,
        "keyword_modifier": round(keyword_modifier, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: COMBINE + CONFIDENCE
# ══════════════════════════════════════════════════════════════════════════════

def compute_confidence(posts, sub_scores):
    """
    Estimate confidence in the risk score.
    Low confidence when: few posts, mixed signals, missing data.
    """
    n = len(posts)
    factors = []

    # Sample size: more posts = more confidence
    if n >= 20:
        factors.append(1.0)
    elif n >= 10:
        factors.append(0.8)
    elif n >= 5:
        factors.append(0.6)
    else:
        factors.append(0.3)

    # Signal consistency: do all sub-scores agree?
    scores = [v for k, v in sub_scores.items() if isinstance(v, (int, float))]
    if scores:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        consistency = max(0, 1 - std / 50)  # lower std = more consistent
        factors.append(consistency)

    # Author data availability
    has_followers = sum(1 for p in posts if p.get("followers", 0) > 0)
    data_quality = has_followers / max(n, 1)
    factors.append(max(data_quality, 0.3))

    avg_confidence = sum(factors) / len(factors)

    if avg_confidence >= 0.7:
        label = "high"
        band_width = 8
    elif avg_confidence >= 0.5:
        label = "medium"
        band_width = 15
    else:
        label = "low"
        band_width = 25

    return label, band_width, round(avg_confidence, 2)


def select_evidence_posts(posts, language_detail):
    """
    Pick top 5 evidence posts — highest engagement × risk signal.
    These are the posts an analyst should read first.
    """
    scored = []
    for p in posts:
        eng = (p.get("shares", 0) * 3 + p.get("comments", 0) * 2
               + p.get("likes", 0) + p.get("views", 0) * 0.01)
        # Boost posts with risk keywords
        text_l = p.get("text", "").lower()
        keyword_boost = sum(2 for kw in RISK_KEYWORDS["high"] if kw in text_l)
        keyword_boost += sum(1 for kw in RISK_KEYWORDS["medium"] if kw in text_l)
        total = eng + keyword_boost * 10
        scored.append((total, p))

    scored.sort(key=lambda x: -x[0])
    return [
        {
            "post_id": p["post_id"],
            "text": p["text"][:300],
            "url": p.get("url", ""),
            "platform": p.get("platform", ""),
            "likes": p.get("likes", 0),
            "shares": p.get("shares", 0),
            "comments": p.get("comments", 0),
            "followers": p.get("followers", 0),
            "handle": p.get("handle", ""),
        }
        for _, p in scored[:5]
    ]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: SCORE ONE NARRATIVE
# ══════════════════════════════════════════════════════════════════════════════

def score_narrative(narrative, enriched_posts, all_sizes, all_engagements, all_author_scores):
    """Compute risk score for a single narrative with full explainability."""

    vol_score, vol_detail = score_volume(enriched_posts, all_sizes)
    vel_score, vel_detail = score_velocity(enriched_posts)
    eng_score, eng_detail = score_engagement(enriched_posts, all_engagements)
    auth_score, auth_detail = score_author(enriched_posts, all_author_scores)
    lang_score, lang_detail = score_language(
        enriched_posts,
        narrative.get("taxonomy_label", "General / Unclassified"),
        narrative.get("sentiment_distribution", {}),
    )

    # Weighted combination
    risk_score = (
        WEIGHTS["volume"]     * vol_score
        + WEIGHTS["velocity"]   * vel_score
        + WEIGHTS["engagement"] * eng_score
        + WEIGHTS["author"]     * auth_score
        + WEIGHTS["language"]   * lang_score
    )
    risk_score = round(min(max(risk_score, 0), 100), 1)

    # Confidence
    sub_scores = {
        "volume": vol_score, "velocity": vel_score, "engagement": eng_score,
        "author": auth_score, "language": lang_score,
    }
    confidence_label, band_width, confidence_raw = compute_confidence(enriched_posts, sub_scores)

    # Build drivers list (sorted by contribution)
    drivers = [
        {"name": "Language Signals", "score": round(lang_score, 1),
         "weight": WEIGHTS["language"],
         "contribution": round(WEIGHTS["language"] * lang_score, 1),
         "detail": lang_detail},
        {"name": "Volume", "score": round(vol_score, 1),
         "weight": WEIGHTS["volume"],
         "contribution": round(WEIGHTS["volume"] * vol_score, 1),
         "detail": vol_detail},
        {"name": "Engagement", "score": round(eng_score, 1),
         "weight": WEIGHTS["engagement"],
         "contribution": round(WEIGHTS["engagement"] * eng_score, 1),
         "detail": eng_detail},
        {"name": "Velocity", "score": round(vel_score, 1),
         "weight": WEIGHTS["velocity"],
         "contribution": round(WEIGHTS["velocity"] * vel_score, 1),
         "detail": vel_detail},
        {"name": "Author Influence", "score": round(auth_score, 1),
         "weight": WEIGHTS["author"],
         "contribution": round(WEIGHTS["author"] * auth_score, 1),
         "detail": auth_detail},
    ]
    drivers.sort(key=lambda d: -d["contribution"])

    # Evidence posts
    evidence = select_evidence_posts(enriched_posts, lang_detail)

    return {
        "narrative_id": narrative["narrative_id"],
        "entity_id": narrative["entity_id"],
        "title": narrative["title"],
        "summary": narrative["summary"],
        "taxonomy_label": narrative.get("taxonomy_label", ""),
        "post_count": narrative["post_count"],
        "risk_score": risk_score,
        "confidence": confidence_label,
        "confidence_band": [
            round(max(risk_score - band_width, 0), 1),
            round(min(risk_score + band_width, 100), 1),
        ],
        "confidence_raw": confidence_raw,
        "drivers": drivers,
        "evidence_posts": evidence,
        "sentiment_distribution": narrative.get("sentiment_distribution", {}),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def score_narratives(narratives_file, posts_file, authors_file, output_file=None):
    """
    Score all narratives in a file.

    Args:
        narratives_file:  path to entity_narratives.json
        posts_file:       path to posts.jsonl (original data)
        authors_file:     path to authors.csv
        output_file:      save scored narratives as JSON

    Returns:
        list of scored narrative dicts, sorted by risk_score descending
    """
    # Load data
    with open(narratives_file) as f:
        narratives = json.load(f)

    posts_index = load_posts_index(posts_file)
    authors_index = load_authors_index(authors_file)

    if not narratives:
        return []

    # Enrich all narratives with full post + author data
    all_enriched = {}
    for narr in narratives:
        enriched = enrich_narrative_posts(narr, posts_index, authors_index)
        all_enriched[narr["narrative_id"]] = enriched

    # Pre-compute percentile baselines across all narratives
    all_sizes = [len(e) for e in all_enriched.values()]
    all_engagements = []
    all_author_raw = []
    for enriched in all_enriched.values():
        total_eng = sum(
            p.get("shares", 0) * 3 + p.get("comments", 0) * 2
            + p.get("likes", 0) + p.get("views", 0) * 0.1
            for p in enriched
        )
        all_engagements.append(total_eng)

        followers = [p.get("followers", 0) for p in enriched if p.get("followers", 0) > 0]
        if followers:
            max_f = max(followers)
            median_f = sorted(followers)[len(followers) // 2]
            unique = len(set(p.get("author_id", "") for p in enriched))
            raw = 0.6 * min(math.log10(max(max_f, 1)) / 7 * 100, 100) + \
                  0.2 * min(math.log10(max(median_f, 1)) / 5 * 100, 100) + \
                  0.2 * min(unique / 20 * 100, 100)
            all_author_raw.append(raw)
        else:
            all_author_raw.append(20)

    # Score each narrative
    scored = []
    for narr in narratives:
        enriched = all_enriched[narr["narrative_id"]]
        if not enriched:
            continue
        result = score_narrative(narr, enriched, all_sizes, all_engagements, all_author_raw)
        scored.append(result)

    # Apply human risk overrides from overrides.json (if exists)
    overrides_path = Path(narratives_file).parent.parent / "overrides.json"
    if overrides_path.exists():
        with open(overrides_path) as f:
            overrides = json.load(f)
        risk_overrides = overrides.get("risk_overrides", {})
        if risk_overrides:
            applied = 0
            for s in scored:
                nid = s["narrative_id"]
                if nid in risk_overrides:
                    feedback = risk_overrides[nid].get("feedback", "")
                    original = s["risk_score"]
                    if feedback == "Too High":
                        s["risk_score"] = max(0, original - 15)
                        s["analyst_override"] = "too_high"
                    elif feedback == "Too Low":
                        s["risk_score"] = min(100, original + 15)
                        s["analyst_override"] = "too_low"
                    s["original_score_before_override"] = original
                    applied += 1
            if applied:
                print(f"  Applied {applied} human risk overrides from overrides.json")

    scored.sort(key=lambda s: -s["risk_score"])

    if output_file:
        with open(output_file, "w") as f:
            json.dump(scored, f, indent=2, ensure_ascii=False)

    # Summary
    entity_id = narratives[0]["entity_id"] if narratives else "?"
    print(f"\n  {'='*75}")
    print(f"  RISK SCORES: {entity_id} ({len(scored)} narratives)")
    print(f"  {'='*75}")
    print(f"  {'#':<3} {'Score':>5} {'Conf':<6} {'Title':<40} {'Taxonomy'}")
    print(f"  {'-'*75}")
    for i, s in enumerate(scored, 1):
        print(f"  {i:<3} {s['risk_score']:>5.1f} {s['confidence']:<6} "
              f"{s['title'][:39]:<40} {s['taxonomy_label']}")
    print(f"  {'='*75}")

    return scored


def score_all_entities(narratives_dir, posts_file, authors_file, output_dir=None):
    """Score narratives for all entities."""
    narr_dir = Path(narratives_dir)
    all_scored = {}

    for narr_file in sorted(narr_dir.glob("*_narratives.json")):
        entity_id = narr_file.stem.replace("_narratives", "")
        out_file = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out_file = str(Path(output_dir) / f"{entity_id}_scored.json")

        scored = score_narratives(
            str(narr_file), posts_file, authors_file, output_file=out_file,
        )
        all_scored[entity_id] = scored

    # Grand summary
    print(f"\n{'='*75}")
    print(f"TOP 15 RISKIEST NARRATIVES (ALL ENTITIES)")
    print(f"{'='*75}")
    all_flat = [(eid, s) for eid, slist in all_scored.items() for s in slist]
    all_flat.sort(key=lambda x: -x[1]["risk_score"])
    print(f"{'#':<3} {'Score':>5} {'Entity':<20} {'Title':<35} {'Taxonomy'}")
    print(f"{'-'*75}")
    for i, (eid, s) in enumerate(all_flat[:15], 1):
        print(f"{i:<3} {s['risk_score']:>5.1f} {eid:<20} "
              f"{s['title'][:34]:<35} {s['taxonomy_label']}")
    print(f"{'='*75}")

    return all_scored
