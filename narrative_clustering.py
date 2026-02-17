"""
Narrative Clustering — LLM topic hashing.
LLM labels each post with canonical topic → GROUP BY → build narratives.
"""

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


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
# STEP 2: LLM TOPIC LABELING
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
    """Label posts with topic + taxonomy + sentiment using LLM in batches."""
    all_labels = {}

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
        except Exception as e:
            # Safe default — assign neutral general label so pipeline continues
            print(f"    LLM batch error: {e} — assigning defaults for {len(batch)} posts")
            for p in batch:
                all_labels[p["post_id"]] = {
                    "topic": "general discussion",
                    "taxonomy": "General / Unclassified",
                    "sentiment": "neutral",
                }

        if (i // batch_size + 1) % 5 == 0:
            print(f"    Labeled {min(i + batch_size, len(posts))}/{len(posts)} posts")
        time.sleep(0.15)

    # Fill any gaps (posts LLM missed in response)
    for p in posts:
        if p["post_id"] not in all_labels:
            all_labels[p["post_id"]] = {
                "topic": "general discussion",
                "taxonomy": "General / Unclassified",
                "sentiment": "neutral",
            }

    return all_labels


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: GROUP BY TOPIC
# ══════════════════════════════════════════════════════════════════════════════

def group_by_topic(posts, labels):
    """Group posts by exact topic label. Works when LLM gives consistent labels."""
    groups = defaultdict(list)
    for p in posts:
        topic = labels.get(p["post_id"], {}).get("topic", "general")
        groups[topic].append(p)
    return dict(groups)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: SUMMARIZE (LLM)
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

    title, summary = summarize_llm(cluster_posts, entity_id, topic_label, client)

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
                       min_cluster_size=2, output_file=None):
    """
    Cluster posts for one entity into narratives.

    Args:
        entity_id:         entity to cluster (e.g. "pfizer")
        resolved_file:     path to resolved_entities.jsonl
        api_key:           Anthropic API key
        min_cluster_size:  minimum posts per narrative
        output_file:       save narratives JSON

    Returns:
        list of narrative dicts sorted by post_count descending
    """
    # Step 1: Filter
    posts = get_entity_posts(entity_id, resolved_file)
    print(f"Entity '{entity_id}': {len(posts)} posts after filtering")
    if not posts:
        return []

    # Step 2: Label with LLM
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    print(f"  Step 2: LLM topic labeling ({len(posts)} posts)...")
    labels = label_posts_llm(posts, entity_id, client)

    # Step 3: Group by topic
    print(f"  Step 3: Grouping by topic...")
    groups = group_by_topic(posts, labels)
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
                         min_cluster_size=2):
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
            output_file=out_file,
        )
        print()

    total = sum(len(v) for v in all_narratives.values())
    print(f"TOTAL: {total} narratives across {len(all_narratives)} entities")
    return all_narratives
