"""
Narrative Clustering Pipeline
================================================================================

PURPOSE:
    For a chosen entity, cluster its matched posts into "narratives" — distinct
    events, themes, or storylines. Each narrative gets a title, summary, and
    list of member post_ids.

ARCHITECTURE:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  Step 1: FILTER — Get all posts for the selected entity             │
    │  Step 2: TOPIC LABELING — Assign a canonical topic to each post     │
    │          Mode A: LLM labels (semantic, consistent, best quality)    │
    │          Mode B: Keyword extraction (free, fast, decent baseline)   │
    │  Step 3: GROUP BY TOPIC — Posts with same/similar topic = 1 cluster │
    │          Mode A: LLM uses consistent labels → direct groupby        │
    │          Mode B: Fuzzy-merge similar keyword labels                 │
    │  Step 4: SUMMARIZE — Title + summary per narrative                  │
    │          Mode A: LLM generates grounded title + summary             │
    │          Mode B: Top keywords as title, first post as summary       │
    └──────────────────────────────────────────────────────────────────────┘

WHY NOT TF-IDF / BM25:
    Both are bag-of-words — they treat "vaccine side effects" and "adverse
    reactions after injection" as completely different because no words overlap.
    Short social media posts don't have enough word overlap for BoW methods
    to form meaningful clusters. Result: one mega-cluster + dust.

WHY LLM TOPIC HASHING:
    The LLM understands that "Pfizer shot gave me blood clots" and "adverse
    cardiac events post-vaccination" are the SAME topic. When instructed to
    use canonical labels, the LLM produces consistent topic strings that we
    can simply GROUP BY. The LLM IS the clustering engine — no distance
    matrix, no threshold tuning, no hyperparameters.

WHY KEYWORD FALLBACK:
    When there's no API key, we extract the most distinctive n-grams from
    each post and use fuzzy string matching to group similar labels. Not as
    good as LLM, but gives a workable baseline with clear improvement path.

STEP-BY-STEP:

    Step 1: FILTER POSTS (get_entity_posts)
        Pull all posts where resolved_entities contains entity_id.
        Strip hashtags/URLs/mentions/emojis. Skip if < 30 chars remain
        or > 40% of text is hashtags (spam filter).

    Step 2: TOPIC LABELING
        LLM mode (label_posts_llm):
            Send posts in batches of 15 to Claude Haiku. Prompt instructs:
            - Assign a 3-8 word canonical topic label
            - Use CONSISTENT labels across posts (same event = same label)
            - Also return sentiment (negative/neutral/positive) and
              risk taxonomy category
            Cost: ~$0.05 for 200 posts on Haiku

        Keyword mode (label_posts_keyword):
            Extract top 3-4 distinctive words after aggressive stop-word
            removal. Match against taxonomy keyword lexicon for category.

    Step 3: GROUP INTO NARRATIVES
        LLM mode: Direct groupby on topic label string. LLM consistency
            means "covid vaccine safety concerns" always comes back as
            exactly that string, so groupby works.

        Keyword mode: Fuzzy merge using rapidfuzz. If two topic labels
            have >= 70% token overlap, merge them into one group.
            Example: "vaccine clinical trials" and "pfizer clinical trials"
            share 2/3 tokens → merge.

    Step 4: TITLE + SUMMARY
        LLM mode: Send cluster posts to Claude → grounded title + summary.
        Keyword mode: Most common topic label as title, first post text
            as summary.

OUTPUT PER NARRATIVE:
    {
        "narrative_id": "pfizer_narrative_1",
        "entity_id": "pfizer",
        "title": "COVID Vaccine Side Effect Concerns",
        "summary": "Multiple posts discuss adverse reactions...",
        "taxonomy_label": "Customer Harm",
        "post_count": 25,
        "post_ids": ["123", "456", ...],
        "sentiment_distribution": {"negative": 15, "neutral": 8, "positive": 2},
        "topic_labels": ["covid vaccine side effects"],
        "sample_posts": [{"post_id": "123", "text": "...", "url": "..."}]
    }

USAGE:
    from narrative_clustering import cluster_narratives

    # With LLM (best quality)
    narratives = cluster_narratives("pfizer", "resolved_entities.jsonl", api_key="sk-ant-...")

    # Without LLM (keyword fallback)
    narratives = cluster_narratives("pfizer", "resolved_entities.jsonl")

    # All entities
    from narrative_clustering import cluster_all_entities
    all_narr = cluster_all_entities("resolved_entities.jsonl", api_key="sk-ant-...")

DEPENDENCIES:
    Required:  rapidfuzz (for keyword label merging)
    Optional:  anthropic (for LLM mode)
    Optional:  scikit-learn (only if you want TF-IDF sub-clustering)
"""

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from rapidfuzz import fuzz


# ══════════════════════════════════════════════════════════════════════════════
# RISK TAXONOMY
# ══════════════════════════════════════════════════════════════════════════════

TAXONOMY_KEYWORDS = {
    "Regulatory / Compliance": [
        "fda", "ema", "regulation", "regulatory", "fine", "fined", "investigation",
        "compliance", "lawsuit", "sue", "sued", "court", "penalty", "violation",
        "approved", "approval", "recalled", "ban", "banned", "patent", "settlement",
        "whistleblower", "subpoena",
    ],
    "Financial Integrity": [
        "fraud", "laundering", "manipulation", "stock", "earnings", "revenue",
        "profit", "investor", "sec", "trading", "price", "market", "shares",
        "billion", "million", "deal", "acquisition", "merger", "buyout",
        "dividend", "valuation", "quarterly",
    ],
    "Customer Harm": [
        "side effect", "adverse", "death", "died", "dying", "injury", "harm",
        "patient", "safety", "risk", "cancer", "treatment", "allergic", "reaction",
        "blood clot", "cardiac", "heart", "stroke", "seizure", "contaminated",
        "toxic", "hospitalized", "suffering", "victim", "dangerous",
    ],
    "Data / Cyber": [
        "breach", "hack", "hacked", "data leak", "ransomware", "cyber", "privacy",
    ],
    "Operational Resilience": [
        "outage", "shortage", "supply", "delay", "disruption", "recall",
        "manufacturing", "plant", "contamination", "quality control",
    ],
    "Executive / Employee Misconduct": [
        "ceo", "executive", "fired", "scandal", "harassment", "misconduct",
        "resign", "resigned", "leadership",
    ],
    "Misinformation / Manipulation": [
        "conspiracy", "misinformation", "fake", "propaganda", "hoax",
        "coverup", "cover-up", "suppressed", "censored", "depopulation",
        "bioweapon", "plandemic", "experimental", "shill",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: FILTER POSTS
# ══════════════════════════════════════════════════════════════════════════════

def get_entity_posts(entity_id, resolved_file):
    """Get all posts for an entity, filtering spam/noise."""
    posts = []
    with open(resolved_file) as f:
        for line in f:
            r = json.loads(line)
            entity_ids = [e["entity_id"] for e in r.get("resolved_entities", [])]
            if entity_id not in entity_ids:
                continue

            text = r.get("text", "")
            clean = re.sub(r'#\S+', '', text)
            clean = re.sub(r'https?://\S+', '', clean)
            clean = re.sub(r'@\w+', '', clean)
            clean = re.sub(r'[^\w\s.,!?\'-]', ' ', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()

            if len(clean) < 30:
                continue
            hashtag_chars = sum(len(m) for m in re.findall(r'#\S+', text))
            if len(text) > 0 and hashtag_chars / len(text) > 0.40:
                continue

            confidence = next(
                (e.get("confidence", 0.95) for e in r["resolved_entities"]
                 if e["entity_id"] == entity_id), 0.95,
            )
            posts.append({
                "post_id": r["post_id"],
                "text": text,
                "clean_text": clean[:500],
                "confidence": confidence,
            })
    return posts


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2a: KEYWORD TOPIC LABELING (fallback)
# ══════════════════════════════════════════════════════════════════════════════

STOP_WORDS = {
    "the", "and", "for", "are", "not", "but", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "who", "how", "its", "let",
    "may", "say", "she", "too", "use", "him", "his", "get", "did", "got",
    "now", "new", "way", "day", "any", "see", "own", "here", "then", "each",
    "make", "over", "such", "take", "well", "only", "come", "many", "want",
    "much", "need", "know", "even", "give", "back", "them", "after", "year",
    "also", "look", "still", "every", "think", "other", "going", "being",
    "people", "really", "right", "where", "thing", "those", "first", "does",
    "told", "says", "said", "done", "made", "sure", "just", "like", "this",
    "that", "with", "from", "have", "been", "were", "will", "about", "they",
    "their", "than", "what", "when", "more", "some", "would", "could",
    "should", "into", "your", "very", "most", "http", "https",
    "everyone", "follow", "viral", "post", "share", "comment", "page",
    "love", "life", "good", "great", "best", "morning", "night",
    "hello", "today", "tonight", "guys",
}


def classify_taxonomy(text_lower):
    """Assign taxonomy category via keyword matching."""
    scores = {}
    for cat, keywords in TAXONOMY_KEYWORDS.items():
        score = sum(len(kw.split()) for kw in keywords if kw in text_lower)
        if score > 0:
            scores[cat] = score
    return max(scores, key=scores.get) if scores else "General / Unclassified"


def extract_topic_keywords(text_lower):
    """Extract top distinctive words as topic label."""
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    words = [w for w in words if w not in STOP_WORDS]
    top = [w for w, _ in Counter(words).most_common(4)]
    return " ".join(top) if top else "general"


def label_posts_keyword(posts):
    """Label all posts with topic + taxonomy using keywords."""
    labels = {}
    for p in posts:
        text_lower = p["clean_text"].lower()
        taxonomy = classify_taxonomy(text_lower)
        topic = extract_topic_keywords(text_lower)

        neg = ["death", "died", "harm", "fraud", "scandal", "lawsuit", "adverse",
               "risk", "dangerous", "fake", "conspiracy", "killed", "toxic",
               "suffering", "victim", "ban", "hoax"]
        pos = ["approved", "success", "breakthrough", "partnership", "growth",
               "profit", "cure", "effective", "safe", "innovation"]
        n = sum(1 for w in neg if w in text_lower)
        po = sum(1 for w in pos if w in text_lower)
        sentiment = "negative" if n > po else ("positive" if po > n else "neutral")

        labels[p["post_id"]] = {
            "topic": topic,
            "taxonomy": taxonomy,
            "sentiment": sentiment,
        }
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2b: LLM TOPIC LABELING (primary)
# ══════════════════════════════════════════════════════════════════════════════

LABEL_PROMPT = """You are labeling social media posts about {entity_id} for narrative clustering.

CRITICAL: Use CONSISTENT topic labels. If two posts discuss the same event/theme,
they MUST get the EXACT SAME topic string. Think of topic as a "bucket name".

Examples of GOOD consistent labels:
  - "covid vaccine safety concerns"  (not sometimes "vaccine side effects" and sometimes "jab risks")
  - "quarterly earnings and stock price"  (not sometimes "financial results" and sometimes "stock movement")
  - "FDA approval of new drug"  (not sometimes "regulatory approval" and sometimes "FDA news")

For each post return:
- topic: 3-8 word canonical topic label (CONSISTENT across similar posts)
- taxonomy: Regulatory / Compliance | Financial Integrity | Customer Harm | Data / Cyber | Operational Resilience | Executive / Employee Misconduct | Misinformation / Manipulation | General / Unclassified
- sentiment: negative | neutral | positive

POSTS:
{posts_block}

Return ONLY a JSON array in same order:
[{{"post_id":"...","topic":"...","taxonomy":"...","sentiment":"..."}}]"""


def label_posts_llm(posts, entity_id, client, batch_size=15):
    """Label posts with topic + taxonomy using LLM in batches."""
    all_labels = {}

    # First pass: label all posts
    for i in range(0, len(posts), batch_size):
        batch = posts[i:i + batch_size]
        posts_block = "\n\n".join(
            f"[{p['post_id']}]: {p['clean_text'][:250]}"
            for p in batch
        )

        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                temperature=0,
                messages=[{"role": "user", "content": LABEL_PROMPT.format(
                    entity_id=entity_id, posts_block=posts_block
                )}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```"))

            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    pid = str(item.get("post_id", ""))
                    all_labels[pid] = {
                        "topic": item.get("topic", "general discussion"),
                        "taxonomy": item.get("taxonomy", "General / Unclassified"),
                        "sentiment": item.get("sentiment", "neutral"),
                    }
        except Exception:
            for p in batch:
                all_labels[p["post_id"]] = label_posts_keyword([p])[p["post_id"]]

        if (i // batch_size + 1) % 5 == 0:
            print(f"    Labeled {min(i + batch_size, len(posts))}/{len(posts)} posts")
        time.sleep(0.15)

    # Fill any gaps
    for p in posts:
        if p["post_id"] not in all_labels:
            all_labels[p["post_id"]] = label_posts_keyword([p])[p["post_id"]]

    return all_labels


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3a: GROUP BY TOPIC — LLM MODE (direct groupby)
# ══════════════════════════════════════════════════════════════════════════════

def group_by_topic_direct(posts, labels):
    """Group posts by exact topic label. Works when LLM gives consistent labels."""
    groups = defaultdict(list)
    for p in posts:
        topic = labels.get(p["post_id"], {}).get("topic", "general")
        groups[topic].append(p)
    return dict(groups)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3b: GROUP BY TOPIC — KEYWORD MODE (fuzzy merge)
# ══════════════════════════════════════════════════════════════════════════════

def fuzzy_merge_groups(groups, merge_threshold=65):
    """
    Merge groups whose topic labels are similar using token-based fuzzy matching.
    "vaccine clinical trials" and "pfizer clinical trial data" → merge.
    """
    topic_keys = list(groups.keys())
    merged = {}
    used = set()

    for i, key_a in enumerate(topic_keys):
        if key_a in used:
            continue
        current_posts = list(groups[key_a])
        current_label = key_a

        for j in range(i + 1, len(topic_keys)):
            key_b = topic_keys[j]
            if key_b in used:
                continue
            # Token sort ratio handles word order differences
            score = fuzz.token_sort_ratio(key_a, key_b)
            if score >= merge_threshold:
                current_posts.extend(groups[key_b])
                used.add(key_b)

        merged[current_label] = current_posts
        used.add(key_a)

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4a: SUMMARIZE — KEYWORD MODE
# ══════════════════════════════════════════════════════════════════════════════

def build_title_keyword(topic_label, taxonomy):
    """Make a readable title from keyword topic label."""
    title = topic_label.title()
    if len(title) < 10:
        title = f"{taxonomy}: {title}"
    return title


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4b: SUMMARIZE — LLM MODE
# ══════════════════════════════════════════════════════════════════════════════

SUMMARY_PROMPT = """Summarize this cluster of {count} social media posts about {entity_id}.

TOPIC: {topic}

SAMPLE POSTS:
{posts_block}

Return ONLY JSON:
{{"title":"5-12 word specific grounded title","summary":"1-2 sentence summary citing specific claims from posts"}}"""


def summarize_llm(cluster_posts, entity_id, topic, client):
    """Generate grounded title + summary for a narrative cluster."""
    posts_block = "\n".join(
        f"- {p['clean_text'][:180]}"
        for p in cluster_posts[:12]
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(
                count=len(cluster_posts), entity_id=entity_id,
                topic=topic, posts_block=posts_block,
            )}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```"))
        parsed = json.loads(raw)
        return parsed.get("title", topic.title()), parsed.get("summary", "")
    except Exception:
        return topic.title(), cluster_posts[0]["clean_text"][:200] if cluster_posts else ""


# ══════════════════════════════════════════════════════════════════════════════
# BUILD NARRATIVE OBJECTS
# ══════════════════════════════════════════════════════════════════════════════

def build_narrative(cluster_posts, entity_id, idx, topic_label, labels, client=None):
    """Build complete narrative dict from a cluster of posts."""
    sentiments = [labels.get(p["post_id"], {}).get("sentiment", "neutral")
                  for p in cluster_posts]
    taxonomies = [labels.get(p["post_id"], {}).get("taxonomy", "General / Unclassified")
                  for p in cluster_posts]
    tax_counts = Counter(taxonomies)
    dominant_taxonomy = tax_counts.most_common(1)[0][0] if tax_counts else "General / Unclassified"

    if client:
        title, summary = summarize_llm(cluster_posts, entity_id, topic_label, client)
    else:
        title = build_title_keyword(topic_label, dominant_taxonomy)
        summary = cluster_posts[0]["clean_text"][:250] if cluster_posts else ""

    return {
        "narrative_id": f"{entity_id}_narrative_{idx}",
        "entity_id": entity_id,
        "title": title,
        "summary": summary,
        "taxonomy_label": dominant_taxonomy,
        "sentiment_distribution": dict(Counter(sentiments)),
        "post_count": len(cluster_posts),
        "post_ids": [p["post_id"] for p in cluster_posts],
        "topic_labels": [topic_label],
        "sample_posts": [
            {"post_id": p["post_id"], "text": p["text"][:300], "url": p.get("url", "")}
            for p in cluster_posts[:5]
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def cluster_narratives(entity_id, resolved_file, api_key=None,
                       min_cluster_size=2, merge_threshold=65,
                       output_file=None):
    """
    Cluster posts for one entity into narratives.

    Args:
        entity_id:         entity to cluster (e.g. "pfizer")
        resolved_file:     path to resolved_entities.jsonl
        api_key:           Anthropic key (None = keyword mode)
        min_cluster_size:  minimum posts per narrative
        merge_threshold:   fuzzy merge threshold for keyword mode (0-100)
        output_file:       save narratives JSON

    Returns:
        list of narrative dicts sorted by post_count descending
    """
    # Step 1: Filter
    posts = get_entity_posts(entity_id, resolved_file)
    print(f"Entity '{entity_id}': {len(posts)} posts after filtering")
    if not posts:
        return []

    # Step 2: Label
    client = None
    if api_key:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        print(f"  Step 2: LLM topic labeling ({len(posts)} posts)...")
        labels = label_posts_llm(posts, entity_id, client)
    else:
        print(f"  Step 2: Keyword topic labeling...")
        labels = label_posts_keyword(posts)

    # Step 3: Group
    print(f"  Step 3: Grouping by topic...")
    if api_key:
        groups = group_by_topic_direct(posts, labels)
    else:
        raw_groups = group_by_topic_direct(posts, labels)
        groups = fuzzy_merge_groups(raw_groups, merge_threshold=merge_threshold)

    print(f"    Raw groups: {len(groups)}")

    # Filter small groups → collect into "Other"
    significant = {}
    other_posts = []
    for topic, group_posts in groups.items():
        if len(group_posts) >= min_cluster_size:
            significant[topic] = group_posts
        else:
            other_posts.extend(group_posts)

    if len(other_posts) >= min_cluster_size:
        significant["miscellaneous topics"] = other_posts

    print(f"    After filtering (min_size={min_cluster_size}): {len(significant)} groups")

    # Step 4: Build narratives
    print(f"  Step 4: Generating titles + summaries...")
    narratives = []
    for idx, (topic, group_posts) in enumerate(significant.items(), 1):
        narrative = build_narrative(group_posts, entity_id, idx, topic,
                                    labels, client=client)
        narratives.append(narrative)
        time.sleep(0.05)

    narratives.sort(key=lambda n: -n["post_count"])

    if output_file:
        with open(output_file, "w") as f:
            json.dump(narratives, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {output_file}")

    # Summary
    print(f"\n  {'='*70}")
    print(f"  NARRATIVES: {entity_id} ({len(narratives)} narratives, {len(posts)} posts)")
    print(f"  {'='*70}")
    print(f"  {'#':<3} {'Title':<45} {'Posts':>5}  Taxonomy")
    print(f"  {'-'*70}")
    for i, n in enumerate(narratives, 1):
        print(f"  {i:<3} {n['title'][:44]:<45} {n['post_count']:>5}  {n['taxonomy_label']}")
    print(f"  {'='*70}")

    return narratives


def cluster_all_entities(resolved_file, api_key=None, output_dir=None,
                         min_cluster_size=2, merge_threshold=65):
    """Run narrative clustering for ALL entities."""
    entity_ids = set()
    with open(resolved_file) as f:
        for line in f:
            for e in json.loads(line).get("resolved_entities", []):
                entity_ids.add(e["entity_id"])

    print(f"Clustering {len(entity_ids)} entities\n")
    all_narratives = {}

    for eid in sorted(entity_ids):
        out_file = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out_file = str(Path(output_dir) / f"{eid}_narratives.json")

        all_narratives[eid] = cluster_narratives(
            entity_id=eid, resolved_file=resolved_file,
            api_key=api_key, min_cluster_size=min_cluster_size,
            merge_threshold=merge_threshold, output_file=out_file,
        )
        print()

    total = sum(len(v) for v in all_narratives.values())
    print(f"TOTAL: {total} narratives across {len(all_narratives)} entities")
    return all_narratives
