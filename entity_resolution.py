"""
Entity Resolution — Fuzzy-first, LLM-second pipeline.
Tier 1: rapidfuzz alias matching (73% coverage, free, deterministic)
Tier 2: Claude Haiku for hard cases (abbreviations, product-parent links)
Merge: priority fuzzy > LLM, confidence 0-1 numeric, flag <0.5 for review
"""

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from rapidfuzz import fuzz


# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_entities(path):
    """
    STEP 1a: Load entity catalog from CSV.

    Reads entities_seed.csv and builds alias lists for each entity.
    Canonical name is always included as the first alias (lowercased).
    Pipe-delimited aliases from CSV are split and appended.
    Duplicates are removed while preserving insertion order.

    Example CSV row:
        entity_id=bristol_myers_squibb, canonical_name=Bristol Myers Squibb,
        aliases=Bristol-Myers Squibb|Bristol Myers Squibb Co.
    Produces:
        aliases = ["bristol myers squibb", "bristol-myers squibb", "bristol myers squibb co."]
    """
    entities = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            canonical = str(row["canonical_name"]).strip()
            aliases_raw = row.get("aliases", "")
            aliases = []
            if isinstance(aliases_raw, str) and aliases_raw.strip():
                aliases = [a.strip().lower() for a in aliases_raw.split("|") if a.strip()]
            all_aliases = list(dict.fromkeys([canonical.lower()] + aliases))
            entities.append({
                "entity_id": str(row["entity_id"]),
                "canonical_name": canonical,
                "entity_type": str(row["entity_type"]),
                "aliases": all_aliases,
            })
    return entities


def load_posts(path):
    """
    STEP 1b: Load posts from JSONL.

    Each line is a JSON object with post_id, text, platform, engagement fields.
    IMPORTANT: We use the 'text' field for entity resolution, NOT 'text_altered'.
    The text_altered field is a data quality trap — entity names are replaced
    with generic descriptions (e.g. "AstraZeneca" → "the British-Swedish
    pharmaceutical company"), which would cause ~60% of matches to be lost.
    """
    posts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                posts.append(json.loads(line))
    return posts


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1a — FUZZY ALIAS MATCHING (always available, no heavy deps)
# ══════════════════════════════════════════════════════════════════════════════

def find_mentions(text, entities):
    """
    STEP 2: Mention Detection — find candidate entity mentions in text.

    Word-boundary-aware search: lowercase the entire post text, then check
    if each entity alias appears as a whole word/phrase. Uses regex word
    boundaries to prevent false positives like "merck" inside "commercial".

    Example: text = "Merck's Keytruda showed strong results"
        → finds ["merck", "keytruda"] (two separate mentions)

    Returns deduplicated list of matched alias strings.
    """
    import re
    text_l = text.lower()
    mentions = []
    for e in entities:
        for alias in e["aliases"]:
            # Word boundary match: prevents "merck" matching inside "commercial"
            # \b handles possessives like "Merck's" correctly
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_l):
                mentions.append(alias)
    return list(set(mentions))


def link_mention_fuzzy(mention, entities, fuzzy_threshold=90):
    """
    STEP 3: Fuzzy Alias Linking — resolve a mention to an entity.

    Uses rapidfuzz.fuzz.ratio() which computes Levenshtein-based similarity
    (0-100 scale). For each mention, we compare against ALL aliases of ALL
    entities and collect those scoring >= threshold (default 90).

    Decision logic:
        - Exactly 1 entity matches above threshold → "linked" (confident match)
        - Multiple entities match above threshold → "ambiguous" (needs disambiguation)
        - No entity matches above threshold → "unresolved" (fuzzy can't help)

    We keep the best score per entity_id so that if an entity has 3 aliases
    and 2 of them match, we only count that entity once (with the highest score).

    Example:
        mention = "bristol myers squibb"
        alias "bristol myers squibb" → score 100 → entity bristol_myers_squibb
        → returns status: "linked", score: 100
    """
    m = mention.lower()
    hits = []
    for e in entities:
        for alias in e["aliases"]:
            score = fuzz.ratio(m, alias)
            if score >= fuzzy_threshold:
                hits.append((e, score))

    unique = {}
    for e, score in hits:
        if e["entity_id"] not in unique or score > unique[e["entity_id"]][1]:
            unique[e["entity_id"]] = (e, score)

    if len(unique) == 1:
        entity, score = list(unique.values())[0]
        return {
            "status": "linked",
            "method": "fuzzy_alias",
            "entity": entity,
            "score": score,
        }
    if len(unique) > 1:
        return {
            "status": "ambiguous",
            "method": "fuzzy_alias",
            "candidates": [e for e, _ in unique.values()],
        }
    return {"status": "unresolved", "method": "fuzzy_alias"}


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — FUZZY MATCHING
# ══════════════════════════════════════════════════════════════════════════════
def resolve_post_tier1(post, entities):
    """
    STEP 5: Resolve all entities in a single post using Tier 1 methods.

    Orchestrates mention detection → fuzzy linking for a single post.
    pipeline for one post. For each detected mention:
        1. Try fuzzy linking first (fast, deterministic)

    Output per entity:
        - status: linked / ambiguous / unresolved
        - resolution_method: fuzzy_alias
        - entity_id, canonical_name, entity_type (if linked)
        - candidates (if ambiguous)
    """
    mentions = find_mentions(post.get("text", ""), entities)
    results = []

    for m in mentions:
        result = link_mention_fuzzy(m, entities)

        output = {
            "mention_text": m,
            "status": result["status"],
            "resolution_method": result["method"],
        }
        if result["status"] == "linked":
            e = result["entity"]
            output["entity_id"] = e["entity_id"]
            output["canonical_name"] = e["canonical_name"]
            output["entity_type"] = e["entity_type"]
            output["fuzzy_score"] = result.get("score", 100)
        if result["status"] == "ambiguous":
            output["candidates"] = [
                {"entity_id": c["entity_id"], "canonical_name": c["canonical_name"]}
                for c in result.get("candidates", [])
            ]
        results.append(output)

    return results


def run_tier1(posts, entities):
    """Run Tier 1 on all posts."""
    print(f"Tier 1: Resolving {len(posts)} posts...")
    results = []
    for i, post in enumerate(posts):
        resolved = resolve_post_tier1(post, entities)
        results.append({
            "post_id": post["post_id"],
            "text": post.get("text", ""),
            "resolved_entities": resolved,
            "needs_review": any(r["status"] != "linked" for r in resolved),
        })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(posts)} done")

    linked = sum(1 for r in results if r["resolved_entities"] and not r["needs_review"])
    needs_review = sum(1 for r in results if r["needs_review"])
    no_entity = sum(1 for r in results if not r["resolved_entities"])
    print(f"Tier 1: {linked} linked | {needs_review} need review | {no_entity} no entity")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — LLM TAGGER (hard cases only)
# ══════════════════════════════════════════════════════════════════════════════

LLM_PROMPT = """You are an entity tagger for pharmaceutical risk monitoring.

ENTITY CATALOG:
{catalog}

POST TEXT:
{text}

Find every reference to a cataloged entity. For each, return:
- entity_id (from catalog)
- mention_text (exact words from post)
- evidence_type: exact_name | known_alias | abbreviation | product_of_parent | contextual_inference | unclear
- reasoning (one sentence)

Rules:
- "Merck Handbook" = book, not company. Tag as contextual_inference with explanation.
- "BMS" in battery context = Battery Management System, not Bristol Myers.
- When a drug implies its manufacturer AND manufacturer is in catalog, tag BOTH.
- If no entities found, return []

Return ONLY a JSON array. No markdown. No preamble.
[{{"entity_id":"...","mention_text":"...","evidence_type":"...","reasoning":"..."}}]"""


def build_catalog_prompt(entities):
    lines = []
    for e in entities:
        aliases = "|".join(e["aliases"]) if e["aliases"] else "none"
        lines.append(f"- {e['entity_id']}: {e['canonical_name']} ({e['entity_type']}) aliases=[{aliases}]")
    return "\n".join(lines)


def call_llm(client, text, catalog_prompt):
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0,
        messages=[{"role": "user", "content": LLM_PROMPT.format(catalog=catalog_prompt, text=text[:4000])}],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```"))
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        return []


def run_tier2(posts, tier1_results, entities, api_key=None):
    from anthropic import Anthropic

    valid_ids = {e["entity_id"] for e in entities}
    catalog_prompt = build_catalog_prompt(entities)
    tier1_by_id = {str(r["post_id"]): r for r in tier1_results}

    # Only posts that need help
    hard_posts = []
    for post in posts:
        pid = str(post["post_id"])
        t1 = tier1_by_id.get(pid, {})
        if t1.get("needs_review", False) or not t1.get("resolved_entities", []):
            hard_posts.append(post)

    if not hard_posts:
        print("Tier 2: No hard cases — skipping LLM")
        return {}

    print(f"Tier 2: LLM tagging {len(hard_posts)} hard cases...")
    client = Anthropic(api_key=api_key) if api_key else Anthropic()

    llm_results = {}
    for i, post in enumerate(hard_posts):
        tags = call_llm(client, post["text"], catalog_prompt)
        llm_results[str(post["post_id"])] = [t for t in tags if t.get("entity_id") in valid_ids]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(hard_posts)} done")
        time.sleep(0.1)

    found = sum(1 for v in llm_results.values() if v)
    print(f"Tier 2: LLM found entities in {found}/{len(hard_posts)} hard cases")
    return llm_results


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT — LLM validates fuzzy matches (optional, runs after merge)
# ══════════════════════════════════════════════════════════════════════════════

AUDIT_PROMPT = """You are auditing entity resolution results. For each post below, fuzzy matching linked it to the given entity. Judge whether each match is CORRECT or WRONG.

ENTITY CATALOG:
{catalog}

POSTS TO AUDIT (JSON array):
{batch}

For each post, return:
- post_id
- entity_id
- verdict: "correct" or "wrong"
- reason: one sentence why

Return ONLY a JSON array. No markdown.
[{{"post_id":"...","entity_id":"...","verdict":"correct|wrong","reason":"..."}}]"""


def audit_fuzzy_with_llm(merged, entities, api_key, sample_size=100):
    """
    LLM-as-judge: audit a sample of fuzzy-resolved entities.
    Sets llm_agrees = True/False on audited posts, leaves "not_audited" on unaudited.

    Args:
        merged:       list of merged results (modified in place)
        entities:     entity catalog list
        api_key:      Anthropic API key
        sample_size:  max posts to audit (controls cost)

    Returns:
        dict with audit stats {total, correct, wrong, agree_rate}
    """
    from anthropic import Anthropic

    # Collect fuzzy-only posts (where llm_agrees is "not_audited")
    fuzzy_posts = []
    for rec in merged:
        for ent in rec.get("resolved_entities", []):
            if ent.get("source") == "fuzzy_alias" and ent.get("llm_agrees") == "not_audited":
                fuzzy_posts.append({
                    "post_id": str(rec["post_id"]),
                    "text": rec.get("text", "")[:300],
                    "entity_id": ent["entity_id"],
                    "mention_text": ent.get("mention_text", ""),
                })

    if not fuzzy_posts:
        print("Audit: No fuzzy-only matches to audit")
        return {"total": 0, "correct": 0, "wrong": 0, "agree_rate": 1.0}

    # Sample if too many
    import random
    if len(fuzzy_posts) > sample_size:
        random.seed(42)
        fuzzy_posts = random.sample(fuzzy_posts, sample_size)

    print(f"Audit: LLM reviewing {len(fuzzy_posts)} fuzzy matches...")
    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    catalog_prompt = build_catalog_prompt(entities)

    # Batch into groups of 15 for efficient LLM calls
    verdicts = {}  # (post_id, entity_id) -> True/False
    batch_size = 15
    for i in range(0, len(fuzzy_posts), batch_size):
        batch = fuzzy_posts[i:i + batch_size]
        batch_json = json.dumps([{
            "post_id": p["post_id"],
            "text": p["text"],
            "entity_id": p["entity_id"],
            "mention_text": p["mention_text"],
        } for p in batch], ensure_ascii=False)

        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                temperature=0,
                messages=[{"role": "user", "content": AUDIT_PROMPT.format(
                    catalog=catalog_prompt, batch=batch_json
                )}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.split("\n") if not l.strip().startswith("```"))
            try:
                results = json.loads(raw)
            except json.JSONDecodeError:
                start, end = raw.find("["), raw.rfind("]")
                if start != -1 and end != -1:
                    results = json.loads(raw[start:end + 1])
                else:
                    results = []

            for r in results:
                pid = str(r.get("post_id", ""))
                eid = r.get("entity_id", "")
                verdict = r.get("verdict", "").lower()
                if pid and eid and verdict in ("correct", "wrong"):
                    verdicts[(pid, eid)] = verdict == "correct"
        except Exception as e:
            print(f"  Audit batch error: {e}")

        if (i + batch_size) % 60 == 0:
            print(f"  {min(i + batch_size, len(fuzzy_posts))}/{len(fuzzy_posts)} audited")
        time.sleep(0.15)

    # Apply verdicts to merged results
    applied = 0
    for rec in merged:
        pid = str(rec["post_id"])
        for ent in rec.get("resolved_entities", []):
            key = (pid, ent.get("entity_id", ""))
            if key in verdicts:
                ent["llm_agrees"] = verdicts[key]
                applied += 1

    correct = sum(1 for v in verdicts.values() if v)
    wrong = sum(1 for v in verdicts.values() if not v)
    rate = correct / max(len(verdicts), 1)

    print(f"Audit: {applied} matches reviewed | {correct} correct, {wrong} wrong | agree rate: {rate:.1%}")
    return {"total": len(verdicts), "correct": correct, "wrong": wrong, "agree_rate": round(rate, 3)}


def merge_results(tier1_results, llm_results):
    merged = []

    for t1 in tier1_results:
        pid = str(t1["post_id"])
        text = t1.get("text", "")[:500]

        t1_entities = t1.get("resolved_entities", [])
        t1_linked = {r["entity_id"]: r for r in t1_entities if r.get("status") == "linked"}
        t1_ambiguous = [r for r in t1_entities if r.get("status") == "ambiguous"]

        llm_tags = llm_results.get(pid, [])
        llm_by_id = {tag["entity_id"]: tag for tag in llm_tags}

        final_entities = []
        seen_ids = set()

        # Keep all Tier 1 linked (primary)
        llm_was_run = len(llm_tags) > 0  # LLM only runs on posts fuzzy couldn't resolve
        for eid, r in t1_linked.items():
            # llm_agrees: True if LLM also found this entity, "not_audited" if LLM wasn't run
            if llm_was_run:
                llm_agrees = eid in llm_by_id
            else:
                llm_agrees = "not_audited"  # LLM not invoked — fuzzy was sufficient
            final_entities.append({
                "entity_id": eid,
                "canonical_name": r.get("canonical_name", ""),
                "entity_type": r.get("entity_type", ""),
                "mention_text": r.get("mention_text", ""),
                "confidence": round(r.get("fuzzy_score", 95) / 100, 2),
                "confidence_label": "high",
                "resolution_method": r.get("resolution_method", ""),
                "source": "fuzzy_alias",
                "llm_agrees": llm_agrees,
            })
            seen_ids.add(eid)

        # Resolve ambiguous with LLM
        for r in t1_ambiguous:
            candidates = r.get("candidates", [])
            candidate_ids = {c["entity_id"] for c in candidates}
            llm_pick = None
            for cid in candidate_ids:
                if cid in llm_by_id:
                    llm_pick = llm_by_id[cid]
                    break
            if llm_pick and llm_pick["entity_id"] not in seen_ids:
                final_entities.append({
                    "entity_id": llm_pick["entity_id"],
                    "mention_text": r.get("mention_text", ""),
                    "confidence": 0.70,
                    "confidence_label": "medium",
                    "resolution_method": "fuzzy_ambiguous_llm_resolved",
                    "source": "llm_disambiguation",
                    "llm_agrees": True,  # LLM picked this from ambiguous candidates
                    "evidence_type": llm_pick.get("evidence_type", ""),
                    "reasoning": llm_pick.get("reasoning", ""),
                })
                seen_ids.add(llm_pick["entity_id"])
            elif candidates and candidates[0]["entity_id"] not in seen_ids:
                final_entities.append({
                    "entity_id": candidates[0]["entity_id"],
                    "canonical_name": candidates[0].get("canonical_name", ""),
                    "mention_text": r.get("mention_text", ""),
                    "confidence": 0.35,
                    "confidence_label": "low",
                    "resolution_method": "fuzzy_ambiguous_unresolved",
                    "source": "fuzzy_alias",
                    "llm_agrees": "not_audited",
                    "needs_review": True,
                })
                seen_ids.add(candidates[0]["entity_id"])

        # Add LLM-only entities (fuzzy missed)
        for eid, tag in llm_by_id.items():
            if eid not in seen_ids:
                final_entities.append({
                    "entity_id": eid,
                    "mention_text": tag.get("mention_text", ""),
                    "confidence": 0.65,
                    "confidence_label": "medium",
                    "resolution_method": "llm_only",
                    "source": "llm",
                    "llm_agrees": True,  # LLM is the source — it agrees by definition
                    "evidence_type": tag.get("evidence_type", ""),
                    "reasoning": tag.get("reasoning", ""),
                })
                seen_ids.add(eid)

        merged.append({
            "post_id": pid,
            "text": text,
            "resolved_entities": final_entities,
            "needs_review": any(e.get("needs_review") or e.get("confidence", 1.0) < 0.5 for e in final_entities),
        })

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def resolve_entities(posts_file, entities_file, api_key=None,
                     limit=None, output_file=None):
    """
    Full hybrid entity resolution.

    Args:
        posts_file:      path to posts.jsonl
        entities_file:   path to entities_seed.csv
        api_key:         Anthropic key (None = skip LLM, fuzzy only)
        limit:           process first N posts
        output_file:     save merged results as JSONL

    Returns:
        list of merged results per post
    """
    entities = load_entities(entities_file)
    posts = load_posts(posts_file)
    if limit:
        posts = posts[:limit]

    # Tier 1
    tier1 = run_tier1(posts, entities)

    # Tier 2
    llm_results = {}
    if api_key:
        llm_results = run_tier2(posts, tier1, entities, api_key=api_key)
    else:
        print("Tier 2: No API key — skipping LLM")

    # Tier 3
    final = merge_results(tier1, llm_results)

    # Audit: LLM validates fuzzy matches (optional, same API key)
    audit_stats = None
    if api_key:
        audit_stats = audit_fuzzy_with_llm(final, entities, api_key, sample_size=100)

    # Apply human overrides from overrides.json (if exists)
    overrides_file = Path(output_file).parent / "overrides.json" if output_file else None
    if overrides_file and overrides_file.exists():
        with open(overrides_file) as f:
            overrides = json.load(f)
        entity_overrides = overrides.get("entity_overrides", {})
        if entity_overrides:
            applied = 0
            for r in final:
                pid = str(r["post_id"])
                if pid in entity_overrides:
                    override = entity_overrides[pid]
                    corrected = override.get("corrected_entity")
                    if corrected is None:
                        # Human said "none" — remove all entities
                        r["resolved_entities"] = []
                    else:
                        # Human corrected to a specific entity
                        r["resolved_entities"] = [{
                            "entity_id": corrected,
                            "canonical_name": corrected,
                            "entity_type": "Unknown",
                            "mention_text": "human_override",
                            "confidence": 1.0,
                            "confidence_label": "high",
                            "resolution_method": "human_override",
                            "source": "human_override",
                            "llm_agrees": "human_override",
                        }]
                    r["needs_review"] = False
                    applied += 1
            if applied:
                print(f"Applied {applied} human entity overrides from overrides.json")

    if output_file:
        with open(output_file, "w") as f:
            for r in final:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nSaved to {output_file}")

    # Summary
    source_counts = defaultdict(int)
    confidence_counts = defaultdict(int)
    entity_counts = defaultdict(int)
    for r in final:
        for e in r["resolved_entities"]:
            source_counts[e.get("source", "?")] += 1
            confidence_counts[e.get("confidence", "?")] += 1
            entity_counts[e["entity_id"]] += 1

    total = sum(source_counts.values())
    posts_with = sum(1 for r in final if r["resolved_entities"])
    review = sum(1 for r in final if r["needs_review"])

    print(f"\n{'='*60}")
    print(f"FINAL: {len(final)} posts | {posts_with} with entities | {review} need review | {total} mentions")
    print(f"\nSource:      ", dict(source_counts))
    print(f"Confidence:  ", dict(confidence_counts))
    print(f"\nTop entities:")
    for eid, c in sorted(entity_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {eid:<30} {c}")
    print(f"{'='*60}")

    return final
