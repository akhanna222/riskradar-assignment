"""
Entity Resolution Pipeline — Hybrid (Fuzzy/Embedding Primary + LLM Secondary)
================================================================================

PURPOSE:
    Given a set of social media posts and a catalog of known entities (companies,
    products), detect which entities are mentioned in each post and resolve them
    to their canonical entity_id. This is the foundation for downstream narrative
    clustering and risk scoring.

ARCHITECTURE OVERVIEW:
    The pipeline runs in 3 tiers, each building on the previous:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  TIER 1 (Primary): Fuzzy Alias + Embedding Matching               │
    │  Runs on ALL posts. No API calls. Fast, deterministic, auditable. │
    │  → Handles ~73% of posts (exact/fuzzy string matches)             │
    ├─────────────────────────────────────────────────────────────────────┤
    │  TIER 2 (Secondary): LLM Tagger (Claude Haiku)                    │
    │  Runs ONLY on hard cases from Tier 1:                             │
    │    - Posts where Tier 1 found NO entities (268/1000 in test)       │
    │    - Posts where Tier 1 returned "ambiguous" (multiple candidates) │
    │  → Catches abbreviations ($PFE), product-parent links, context    │
    ├─────────────────────────────────────────────────────────────────────┤
    │  TIER 3: Merge + Source Tracking                                   │
    │  Combines Tier 1 + Tier 2 results. Every entity carries:          │
    │    - source: where it came from (fuzzy_embedding / llm / llm_dis) │
    │    - confidence: high / medium / low                               │
    │    - llm_agrees: whether LLM confirmed the fuzzy match            │
    │  → Full audit trail for every resolution decision                  │
    └─────────────────────────────────────────────────────────────────────┘

STEP-BY-STEP LOGIC:

    Step 1: LOAD DATA
        - load_entities(): Read entities_seed.csv. For each entity, build a list
          of aliases by combining canonical_name (lowercased) + pipe-delimited
          aliases from the CSV. Deduplicates aliases preserving order.
          Example: "Bristol Myers Squibb" with aliases "Bristol-Myers Squibb|Bristol Myers Squibb Co."
          → aliases = ["bristol myers squibb", "bristol-myers squibb", "bristol myers squibb co."]
        - load_posts(): Read posts.jsonl line by line. Each post has post_id,
          text, platform, engagement fields, etc.
          IMPORTANT: We use the 'text' field, NOT 'text_altered'. The text_altered
          field is a data trap — it replaces entity names with generic descriptions
          (e.g., "AstraZeneca" → "the British-Swedish pharmaceutical company").

    Step 2: TIER 1a — MENTION DETECTION (find_mentions)
        For each post, scan the lowercased text for exact substring matches
        against all entity aliases. This is a simple "is alias in text?" check.
        Example: Post text contains "Merck" → mention "merck" detected.
        Returns a deduplicated list of detected mention strings.

    Step 3: TIER 1a — FUZZY LINKING (link_mention_fuzzy)
        For each detected mention, compare it against ALL aliases of ALL entities
        using rapidfuzz.fuzz.ratio() (Levenshtein-based similarity, 0-100).
        - If ratio >= 90 (threshold) with exactly ONE entity → status: "linked"
        - If ratio >= 90 with MULTIPLE entities → status: "ambiguous" (e.g., "BMS"
          could match Bristol Myers Squibb or a battery management system)
        - If no alias scores >= 90 → status: "unresolved"
        We keep the best score per entity_id to handle cases where an entity has
        multiple aliases that all match.

    Step 4: TIER 1b — EMBEDDING FALLBACK (optional, link_mention_embedding)
        If fuzzy matching returns "unresolved" or "ambiguous" AND sentence-transformers
        is installed (use_embeddings=True), we try semantic similarity:
        - Pre-compute normalized embeddings for ALL aliases using all-MiniLM-L6-v2
        - Encode the mention text into the same embedding space
        - Compute cosine similarity against all alias embeddings
        - If top similarity >= 0.55 AND the gap to second-best >= 0.03 → "linked"
        - If gap < 0.03 (two entities are equally close) → "ambiguous"
        - If top score < 0.55 → "unresolved"
        This catches misspellings and paraphrases that exact/fuzzy matching misses.
        NOTE: Requires ~500MB RAM for model loading. Skip on memory-constrained envs.

    Step 5: TIER 1 OUTPUT (resolve_post_tier1 → run_tier1)
        For each post, produce:
        {
            "post_id": "...",
            "text": "...",
            "resolved_entities": [
                {"entity_id": "merck", "mention_text": "merck", "status": "linked",
                 "resolution_method": "fuzzy_alias", "canonical_name": "Merck", ...},
            ],
            "needs_review": false  // true if any entity is ambiguous/unresolved
        }
        Tier 1 summary stats: linked / needs_review / no_entity counts.

    Step 6: TIER 2 — LLM TAGGER (run_tier2, only if api_key provided)
        Selects ONLY the hard cases from Tier 1:
        - Posts where needs_review=True (ambiguous matches)
        - Posts where resolved_entities is empty (no matches found)
        For each hard post, sends the post text + full entity catalog to Claude
        Haiku via the Anthropic API. The LLM returns structured JSON:
        [{"entity_id": "...", "mention_text": "...", "evidence_type": "...",
          "reasoning": "..."}]
        Evidence types: exact_name, known_alias, abbreviation, product_of_parent,
        contextual_inference, unclear.
        The prompt includes guardrails:
        - "Merck Handbook" should be tagged as contextual_inference (book reference)
        - "BMS" in battery context should NOT match Bristol Myers Squibb
        - Product mentions should also tag the parent manufacturer
        We filter results to only valid entity_ids from the catalog (prevents
        hallucinated entities).

    Step 7: TIER 3 — MERGE (merge_results)
        For each post, combine Tier 1 and Tier 2 results with clear priority:

        Priority 1 — Tier 1 linked entities (source: "fuzzy_embedding"):
            These are the primary results. Confidence: "high".
            We also check if the LLM agreed (llm_agrees: true/false) for
            cross-validation.

        Priority 2 — Ambiguous cases resolved by LLM (source: "llm_disambiguation"):
            If Tier 1 found multiple candidates and the LLM picked one of them,
            we use the LLM's choice. Confidence: "medium".
            If no LLM available, we take the first candidate with confidence: "low"
            and flag needs_review: true.

        Priority 3 — LLM-only entities (source: "llm"):
            Entities the LLM found that fuzzy matching completely missed.
            Typically: abbreviations ($PFE → Pfizer), product-to-parent links
            (Tremfya → Johnson & Johnson), contextual references.
            Confidence: "medium".

    Step 8: OUTPUT
        Final output per post:
        {
            "post_id": "122227002926249609",
            "text": "UPDATE: All Good!! Gettin that Pacemaker checkup...",
            "resolved_entities": [
                {
                    "entity_id": "opdivo",
                    "canonical_name": "Opdivo",
                    "entity_type": "Product",
                    "mention_text": "opdivo",
                    "confidence": "high",
                    "resolution_method": "fuzzy_alias",
                    "source": "fuzzy_embedding",
                    "llm_agrees": true
                },
                {
                    "entity_id": "bristol_myers_squibb",
                    "mention_text": "Opdivo",
                    "confidence": "medium",
                    "resolution_method": "llm_only",
                    "source": "llm",
                    "evidence_type": "product_of_parent",
                    "reasoning": "Opdivo is manufactured by Bristol Myers Squibb"
                }
            ],
            "needs_review": false
        }
        Saved as JSONL (one JSON object per line) if output_file is specified.

WHY THIS DESIGN:
    - Fuzzy is primary because it's fast (1000 posts/sec), free, deterministic,
      and auditable. Same input always gives same output.
    - LLM is secondary because it's slow (~100ms/post), costs money ($0.10 for
      268 posts on Haiku), and non-deterministic. But it catches things fuzzy
      can't: abbreviations, product-parent relationships, contextual disambiguation.
    - The merge layer ensures every entity resolution decision has a traceable
      source. An auditor can ask "why did you tag this post with Pfizer?" and
      get: "fuzzy_alias match with score 100, LLM confirmed."

WHAT THE LLM CATCHES THAT FUZZY MISSES:
    - "$PFE" → Pfizer (ticker symbols)
    - "J&J" → Johnson & Johnson (abbreviations not in aliases)
    - "Tremfya approved" → Johnson & Johnson (product-to-parent inference)
    - "their covid vaccine" → contextual_inference (no explicit entity name)
    - "Merck Handbook" → correctly tagged as book reference, not the company

WHAT FUZZY CATCHES THAT LLM MIGHT MISS:
    - Consistent, reproducible results (LLM can vary between runs)
    - No API dependency (works offline)
    - No cost (LLM charges per token)

DEPENDENCIES:
    Required:  rapidfuzz
    Optional:  sentence-transformers (for embedding similarity)
    Optional:  anthropic (for LLM tier)

USAGE:
    from entity_resolution import resolve_entities

    # Mode 1: Fuzzy only (no API key, no embeddings)
    final = resolve_entities("posts.jsonl", "entities_seed.csv")

    # Mode 2: Fuzzy + Embedding (Colab / machine with RAM)
    final = resolve_entities("posts.jsonl", "entities_seed.csv", use_embeddings=True)

    # Mode 3: Full hybrid (Fuzzy + Embedding + LLM for hard cases)
    final = resolve_entities("posts.jsonl", "entities_seed.csv",
                             use_embeddings=True, api_key="sk-ant-...")

TEST RESULTS (on 1000 pharma social media posts):
    Tier 1 (fuzzy only): 732 posts linked, 268 no entity, 0 ambiguous
    997 total entity mentions, all 19 catalog entities found
    Top: Pfizer (212), AstraZeneca (180), Merck (101), Eliquis (77)
"""

import csv
import json
import time
from collections import defaultdict
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

    Simple substring search: lowercase the entire post text, then check if
    each entity alias appears as a substring. This is intentionally simple
    and fast — the linking step (Step 3) handles disambiguation.

    Example: text = "Merck's Keytruda showed strong results"
        → finds ["merck", "keytruda"] (two separate mentions)

    Returns deduplicated list of matched alias strings.
    """
    text_l = text.lower()
    mentions = []
    for e in entities:
        for alias in e["aliases"]:
            if alias in text_l:
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
# TIER 1b — EMBEDDING SIMILARITY (optional, needs sentence-transformers)
# ══════════════════════════════════════════════════════════════════════════════

def build_embedding_index(entities, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    STEP 4a: Build Embedding Index (optional, requires sentence-transformers).

    Loads the all-MiniLM-L6-v2 model (~80MB) and encodes ALL entity aliases
    into 384-dimensional normalized vectors. These are stored in memory for
    fast cosine similarity lookup during linking.

    This is a one-time cost at pipeline startup. The index is reused for all posts.
    Returns (model, embeddings_matrix, entity_meta_list) or (None, None, None)
    if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer(model_name)
        alias_texts = []
        entity_meta = []
        for e in entities:
            for alias in e["aliases"]:
                alias_texts.append(alias)
                entity_meta.append(e)
        embeddings = model.encode(alias_texts, normalize_embeddings=True)
        return model, embeddings, entity_meta
    except ImportError:
        print("  sentence-transformers not installed — skipping embeddings")
        return None, None, None


def link_mention_embedding(mention, model, alias_embeddings, entity_meta,
                           embed_threshold=0.55, ambiguity_margin=0.03):
    """
    STEP 4b: Embedding Similarity Linking — semantic fallback for fuzzy failures.

    Encodes the mention text into the same embedding space as the alias index,
    then computes cosine similarity against all pre-computed alias embeddings.

    Decision logic:
        - Top similarity >= 0.55 AND gap to second >= 0.03 → "linked"
        - Top similarity >= 0.55 BUT gap to second < 0.03 → "ambiguous"
          (two entities are semantically equidistant — can't decide)
        - Top similarity < 0.55 → "unresolved" (nothing is close enough)

    This catches cases fuzzy matching misses:
        - Misspellings: "Astrazenecca" → AstraZeneca (embedding similarity ~0.85)
        - Semantic variants: "BMS drugs" → Bristol Myers Squibb
    """
    from sentence_transformers import util
    emb = model.encode(mention.lower(), normalize_embeddings=True)
    sims = util.cos_sim(emb, alias_embeddings)[0]
    top2 = sims.topk(k=min(2, sims.shape[0]))

    top_score = float(top2.values[0])
    top_idx = int(top2.indices[0])

    if top_score < embed_threshold:
        return {"status": "unresolved", "method": "embedding"}

    if len(top2.values) > 1:
        second_score = float(top2.values[1])
        if (top_score - second_score) < ambiguity_margin:
            return {"status": "ambiguous", "method": "embedding"}

    return {
        "status": "linked",
        "method": "embedding",
        "entity": entity_meta[top_idx],
        "score": top_score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — COMBINED FUZZY + EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def resolve_post_tier1(post, entities, emb_model=None, emb_embeddings=None, emb_meta=None):
    """
    STEP 5: Resolve all entities in a single post using Tier 1 methods.

    Orchestrates the mention detection → fuzzy linking → embedding fallback
    pipeline for one post. For each detected mention:
        1. Try fuzzy linking first (fast, deterministic)
        2. If fuzzy returns "unresolved" or "ambiguous" AND embeddings are
           available, try embedding similarity as a fallback
        3. If embedding returns "linked", override the fuzzy result

    Output per entity:
        - status: linked / ambiguous / unresolved
        - resolution_method: fuzzy_alias / embedding
        - entity_id, canonical_name, entity_type (if linked)
        - candidates (if ambiguous)
    """
    mentions = find_mentions(post.get("text", ""), entities)
    results = []

    for m in mentions:
        result = link_mention_fuzzy(m, entities)

        # If fuzzy failed and embeddings available, try embedding
        if result["status"] in ("unresolved", "ambiguous") and emb_model is not None:
            emb_result = link_mention_embedding(m, emb_model, emb_embeddings, emb_meta)
            if emb_result["status"] == "linked":
                result = emb_result

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


def run_tier1(posts, entities, use_embeddings=False):
    """Run Tier 1 on all posts."""
    emb_model, emb_embeddings, emb_meta = None, None, None
    if use_embeddings:
        print("Tier 1: Building embedding index...")
        emb_model, emb_embeddings, emb_meta = build_embedding_index(entities)

    print(f"Tier 1: Resolving {len(posts)} posts...")
    results = []
    for i, post in enumerate(posts):
        resolved = resolve_post_tier1(post, entities, emb_model, emb_embeddings, emb_meta)
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
# TIER 3 — MERGE
# ══════════════════════════════════════════════════════════════════════════════

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
        for eid, r in t1_linked.items():
            final_entities.append({
                "entity_id": eid,
                "canonical_name": r.get("canonical_name", ""),
                "entity_type": r.get("entity_type", ""),
                "mention_text": r.get("mention_text", ""),
                "confidence": round(r.get("fuzzy_score", 95) / 100, 2),
                "confidence_label": "high",
                "resolution_method": r.get("resolution_method", ""),
                "source": "fuzzy_embedding",
                "llm_agrees": eid in llm_by_id,
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
                    "source": "fuzzy_embedding",
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
                     use_embeddings=False, limit=None, output_file=None):
    """
    Full hybrid entity resolution.

    Args:
        posts_file:      path to posts.jsonl
        entities_file:   path to entities_seed.csv
        api_key:         Anthropic key (None = skip LLM, fuzzy only)
        use_embeddings:  True = load sentence-transformers (needs RAM)
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
    tier1 = run_tier1(posts, entities, use_embeddings=use_embeddings)

    # Tier 2
    llm_results = {}
    if api_key:
        llm_results = run_tier2(posts, tier1, entities, api_key=api_key)
    else:
        print("Tier 2: No API key — skipping LLM")

    # Tier 3
    final = merge_results(tier1, llm_results)

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
