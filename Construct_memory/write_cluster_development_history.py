"""
write_cluster_development_history.py
------------------------------------
Read the tree-structured refined memory library, write a STATIC development
history for each leaf theme, then write aggregate histories for non-leaf nodes
that directly contain leaf children. Each aggregate history writes per-point
history only for its direct leaf children, and its overall summary can also
integrate already generated child aggregate histories.

The final output keeps leaf themes in the original `themes` list so retrieval
continues to operate on leaf-level memory. Aggregate histories are stored
separately in `aggregate_themes` for visualization and tree exploration. Each
memory unit has:
  - title
  - description
  - development_history
  - development_history_structured
  - members

When writing the history, the script also looks back into all_memories.json to
retrieve each source paper's title, summary, final decision, and peer-review
signals (meta-review + reviewer comments/scores) so the LLM can better judge
historical standing and innovation significance.

Inputs:
  - <project-root>/memory_database/refined_memory_output/refined_memory_library.json
  - <project-root>/memory_output/all_memories.json

Outputs:
  <project-root>/memory_database/final_memory_output/
    final_theme_<theme_id>.json
    aggregate_theme_<theme_id>.json
    final_memory_library.json

Useful subset testing:
  python write_cluster_development_history.py --max-themes 10
"""

import os
import re
import json
import time
import sys
import argparse
import traceback
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONSTRUCT_MEMORY_DIR = str(PROJECT_ROOT / "Construct_memory")
REFINED_MEMORY_OUTPUT_DIR = str(PROJECT_ROOT / "memory_database" / "refined_memory_output")
FINAL_MEMORY_OUTPUT_DIR = str(PROJECT_ROOT / "memory_database" / "final_memory_output")
MEMORY_OUTPUT_DIR = str(PROJECT_ROOT / "memory_output")
LOG_DIR = str(PROJECT_ROOT / "logs")


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(CONSTRUCT_MEMORY_DIR, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_client(config):
    return openai.OpenAI(
        api_key=config["api"]["openai_api_key"],
        base_url=config["api"]["openai_base_url"],
        timeout=config["api"].get("openai_timeout", 240.0),
    )


def extract_response_text(response):
    output_text = getattr(response, "output_text", None)
    if output_text and str(output_text).strip():
        return str(output_text)

    try:
        data = response.model_dump()
    except Exception:
        data = None

    if isinstance(data, dict):
        output_items = data.get("output", [])
        text_parts = []
        for item in output_items:
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    text_parts.append(content["text"])
        if text_parts:
            return "\n".join(text_parts)

    raise ValueError("Could not extract text from Responses API response")


def strip_json_fence(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def call_llm_json(client, config, system_prompt, user_prompt):
    model = config["llm_config"]["model"]
    temperature = config["llm_config"]["temperature"]
    max_retries = config["llm_config"]["max_retries"]
    retry_delay = config["llm_config"]["retry_delay"]

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
                ],
                text={"format": {"type": "json_object"}, "verbosity": "low"},
                temperature=temperature,
                stream=False,
            )
            text = strip_json_fence(extract_response_text(response))
            if not text:
                raise ValueError("Empty LLM response")
            return json.loads(text)
        except Exception as e:
            last_error = e
            print(f"[WARN] LLM JSON call failed attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    raise RuntimeError(f"LLM JSON call failed after {max_retries} attempts: {last_error}")


def safe_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def format_scores(scores):
    if not isinstance(scores, dict) or not scores:
        return "Not available"

    preferred_order = [
        "Overall",
        "Confidence",
        "Novelty",
        "Soundness",
        "Solid",
        "Presentation",
        "Clarity",
        "Reproducibility",
        "Impact",
    ]
    ordered_keys = [key for key in preferred_order if key in scores]
    ordered_keys.extend(sorted(key for key in scores.keys() if key not in ordered_keys))

    parts = []
    for key in ordered_keys:
        value = scores.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "Not available"


def format_meta_review(meta_review):
    if not isinstance(meta_review, list) or not meta_review:
        return "None"

    lines = []
    for item in meta_review:
        role = safe_text(item.get("role")) or "MetaReview"
        comment = safe_text(item.get("comment"))
        if comment:
            lines.append(f"- {role}: {comment}")
    return "\n".join(lines) if lines else "None"


def format_reviewer_reviews(reviewer_reviews):
    if not isinstance(reviewer_reviews, list) or not reviewer_reviews:
        return "None"

    lines = []
    for item in reviewer_reviews:
        reviewer_index = item.get("reviewer_index")
        role = safe_text(item.get("role")) or "Reviewer"
        scores_text = format_scores(item.get("scores"))
        comment = safe_text(item.get("review"))
        reviewer_label = f"{role} {reviewer_index}" if reviewer_index is not None else role
        if comment:
            lines.append(f"- {reviewer_label} | Scores: {scores_text} | Comment: {comment}")
        else:
            lines.append(f"- {reviewer_label} | Scores: {scores_text}")
    return "\n".join(lines) if lines else "None"


def normalize_history_structure(raw_history):
    if not isinstance(raw_history, dict):
        raise ValueError("Structured development history must be a JSON object")

    normalized = {
        "theme_overview": safe_text(raw_history.get("theme_overview")),
        "innovation_points": [],
        "overall_evolution_summary": safe_text(raw_history.get("overall_evolution_summary")),
    }

    innovation_points = raw_history.get("innovation_points", [])
    if not isinstance(innovation_points, list):
        innovation_points = []

    for item in innovation_points:
        if not isinstance(item, dict):
            continue

        evaluation = item.get("evaluation", {})
        if not isinstance(evaluation, dict):
            evaluation = {}

        normalized["innovation_points"].append({
            "item_number": item.get("item_number"),
            "year": item.get("year"),
            "paper_title": safe_text(item.get("paper_title")),
            "innovation_content": safe_text(item.get("innovation_content")),
            "evaluation": {
                "rating": safe_text(evaluation.get("rating")),
                "assessment": safe_text(
                    evaluation.get("assessment")
                    or evaluation.get("historical_position")
                    or evaluation.get("historical_value")
                    or evaluation.get("judgment_basis")
                ),
            },
        })

    return normalized


def render_structured_history_as_text(structured_history):
    if not isinstance(structured_history, dict):
        return safe_text(structured_history)

    sections = []

    theme_overview = safe_text(structured_history.get("theme_overview"))
    if theme_overview:
        sections.append(f"Theme overview: {theme_overview}")

    for point in structured_history.get("innovation_points", []):
        if not isinstance(point, dict):
            continue

        evaluation = point.get("evaluation", {})
        if not isinstance(evaluation, dict):
            evaluation = {}

        point_lines = []
        year = point.get("year")
        paper_title = safe_text(point.get("paper_title")) or "Unknown paper"
        item_number = point.get("item_number")
        header = f"[{year}] {paper_title}" if year is not None else paper_title
        if item_number is not None:
            header += f" | Item {item_number}"
        point_lines.append(header)

        innovation_content = safe_text(point.get("innovation_content"))
        if innovation_content:
            point_lines.append(f"Innovation content: {innovation_content}")

        rating = safe_text(evaluation.get("rating"))
        assessment = safe_text(evaluation.get("assessment"))

        if rating:
            point_lines.append(f"Evaluation rating: {rating}")
        if assessment:
            point_lines.append(f"Evaluation: {assessment}")

        sections.append("\n".join(point_lines))

    overall_evolution_summary = safe_text(structured_history.get("overall_evolution_summary"))
    if overall_evolution_summary:
        sections.append(f"Overall evolution summary: {overall_evolution_summary}")

    return "\n\n".join(sections)


def load_refined_library(path=None):
    if path is None:
        path = os.path.join(REFINED_MEMORY_OUTPUT_DIR, "refined_memory_library.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded refined library: {path}")
    print(f"[INFO] Coarse clusters in current file: {len(data.get('clusters', []))}")
    return data


def load_original_memory(path=None):
    if path is None:
        path = os.path.join(MEMORY_OUTPUT_DIR, "all_memories.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded original memory: {path}")
    print(f"[INFO] Original memory entries: {len(data)}")
    return data


def build_memory_lookup(memories):
    by_source = {}
    by_paper_id = {}
    for mem in memories:
        metadata = mem.get("metadata", {})
        source_filename = metadata.get("source_filename")
        paper_id = mem.get("paper_id")
        if source_filename:
            by_source[source_filename] = mem
        if paper_id:
            by_paper_id[paper_id] = mem
    return by_source, by_paper_id


def collect_leaf_themes(node, source_cluster_id, leaves):
    if not isinstance(node, dict):
        return

    children = node.get("children") or []
    is_leaf = bool(node.get("is_leaf")) or not children

    if is_leaf:
        leaves.append({
            "theme_id": str(node.get("node_id") or f"{source_cluster_id}"),
            "source_cluster_id": source_cluster_id,
            "title": node.get("title", ""),
            "description": node.get("description", ""),
            "members": node.get("members", []),
        })
        return

    for child in children:
        collect_leaf_themes(child, source_cluster_id, leaves)


def flatten_final_themes(refined_library):
    themes = []
    for cluster in refined_library.get("clusters", []):
        cluster_id = cluster.get("cluster_id")

        # New format: each coarse cluster stores a recursive tree under "tree".
        tree = cluster.get("tree")
        if tree:
            collect_leaf_themes(tree, cluster_id, themes)
            continue

        # Backward-compatible fallback for older outputs.
        for subcategory in cluster.get("subcategories", []):
            themes.append({
                "theme_id": f"{cluster_id}_{subcategory.get('subcategory_id', 'C1')}",
                "source_cluster_id": cluster_id,
                "title": subcategory.get("title", ""),
                "description": subcategory.get("description", ""),
                "members": subcategory.get("members", []),
            })
    return themes


def collect_direct_leaf_parent_aggregate_themes(refined_library):
    """
    Collect aggregate nodes that directly own at least one leaf child.

    Each aggregate node writes per-point history only for its own direct leaf
    children. Non-leaf child branches are represented through the already
    generated nearest descendant aggregate histories and are only used when
    rewriting the aggregate node's overall chronological summary.
    """
    aggregate_lookup = {}

    def make_record(node, source_cluster_id, depth, direct_leaf_ids, descendant_leaf_ids, child_aggregate_ids):
        node_id = safe_text(node.get("node_id"))
        if not node_id:
            return
        aggregate_lookup[node_id] = {
            "theme_id": node_id,
            "source_cluster_id": source_cluster_id,
            "title": node.get("title", ""),
            "description": node.get("description", ""),
            "members": [],
            "history_scope": "aggregate",
            "aggregate_level": "aggregate",
            "aggregate_source": "direct_leaf_histories",
            "aggregation_strategy": "direct_leaf_plus_child_aggregates",
            "coverage_policy": "cover_direct_leaf_items",
            "tree_depth": depth,
            "direct_leaf_theme_ids": sorted(direct_leaf_ids),
            "descendant_leaf_theme_ids": sorted(descendant_leaf_ids),
            "child_aggregate_theme_ids": sorted(child_aggregate_ids),
        }

    def walk(node, source_cluster_id, depth):
        if not isinstance(node, dict):
            return set(), []

        children = [child for child in node.get("children", []) if isinstance(child, dict)]
        is_leaf = bool(node.get("is_leaf")) or not children
        node_id = safe_text(node.get("node_id")) or str(source_cluster_id)
        if is_leaf:
            return {node_id}, []

        direct_leaf_ids = set()
        descendant_leaf_ids = set()
        nearest_child_aggregate_ids = set()

        for child in children:
            child_children = [grandchild for grandchild in child.get("children", []) if isinstance(grandchild, dict)]
            child_is_leaf = bool(child.get("is_leaf")) or not child_children
            child_leaf_ids, child_nearest_aggregate_ids = walk(child, source_cluster_id, depth + 1)
            descendant_leaf_ids.update(child_leaf_ids)
            if child_is_leaf:
                direct_leaf_ids.update(child_leaf_ids)
            else:
                nearest_child_aggregate_ids.update(child_nearest_aggregate_ids)

        if direct_leaf_ids:
            make_record(
                node,
                source_cluster_id,
                depth,
                direct_leaf_ids,
                descendant_leaf_ids,
                nearest_child_aggregate_ids,
            )
            return descendant_leaf_ids, [node_id]

        return descendant_leaf_ids, sorted(nearest_child_aggregate_ids)

    for cluster in refined_library.get("clusters", []):
        cluster_id = cluster.get("cluster_id")
        tree = cluster.get("tree")
        if isinstance(tree, dict):
            walk(tree, cluster_id, 0)

    aggregate_themes = list(aggregate_lookup.values())
    aggregate_themes.sort(key=lambda x: (-int(x.get("tree_depth", 0)), x["theme_id"]))
    return aggregate_themes


def truncate_text(text, max_chars=700):
    text = safe_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def sort_members_by_year(members):
    return sorted(members, key=lambda x: ((x.get("year") is None), x.get("year") or 9999, x.get("title", "")))


def build_theme_source_material(theme, by_source, by_paper_id):
    records = []
    for idx, member in enumerate(theme.get("members", []), 1):
        record = dict(member)
        source_filename = member.get("source_filename")
        paper_id = member.get("paper_id")
        original_mem = by_source.get(source_filename) or by_paper_id.get(paper_id)
        record["paper_title"] = safe_text(member.get("title")) or "Unknown paper"
        record["paper_summary"] = ""
        record["final_decision"] = "Unknown"
        record["meta_review_text"] = "None"
        record["reviewer_reviews_text"] = "None"

        if original_mem is not None:
            record["paper_title"] = safe_text(original_mem.get("title")) or record["paper_title"]
            record["paper_summary"] = original_mem.get("summary") or ""
            record["final_decision"] = safe_text(original_mem.get("final_decision")) or "Unknown"
            review_comments = original_mem.get("review_comments") or {}
            record["meta_review_text"] = format_meta_review(review_comments.get("meta_review"))
            record["reviewer_reviews_text"] = format_reviewer_reviews(review_comments.get("reviewer_reviews"))

        record["theme_point_number"] = member.get("point_number") or idx
        records.append(record)

    records = sort_members_by_year(records)

    lines = []
    for idx, item in enumerate(records, 1):
        title = safe_text(item.get("paper_title")) or safe_text(item.get("title")) or "Unknown paper"
        year = item.get("year")
        summary = safe_text(item.get("paper_summary"))
        innovation = safe_text(item.get("point_text"))
        final_decision = safe_text(item.get("final_decision")) or "Unknown"
        meta_review_text = item.get("meta_review_text") or "None"
        reviewer_reviews_text = item.get("reviewer_reviews_text") or "None"
        lines.append(
            f"Item {idx}\n"
            f"Theme Point Number: {item.get('theme_point_number')}\n"
            f"Source Document: {title}\n"
            f"Paper Title: {title}\n"
            f"Publication Year: {year}\n"
            f"Source Filename: {item.get('source_filename')}\n"
            f"Original Point Index: {item.get('point_index')}\n"
            f"Paper Summary: {summary}\n"
            f"Innovation Point: {innovation}\n"
            f"Final Decision: {final_decision}\n"
            f"Meta-Review Comments (highest priority if they conflict with reviewer comments):\n{meta_review_text}\n"
            f"Reviewer Comments and Scores:\n{reviewer_reviews_text}"
        )

    return "\n\n".join(lines)


def build_parent_aggregate_source_material(aggregate_theme, leaf_history_lookup):
    records = []
    seen_member_keys = set()
    aggregate_members = []

    leaf_ids = aggregate_theme.get("direct_leaf_theme_ids") or aggregate_theme.get("descendant_leaf_theme_ids", [])
    for leaf_id in leaf_ids:
        leaf_theme = leaf_history_lookup.get(leaf_id)
        if not leaf_theme:
            continue

        for member in leaf_theme.get("members", []):
            member_key = (
                member.get("embedding_idx"),
                member.get("source_filename"),
                member.get("point_index"),
            )
            if member_key in seen_member_keys:
                continue
            seen_member_keys.add(member_key)
            aggregate_members.append(member)

        structured = leaf_theme.get("development_history_structured") or {}
        points = structured.get("innovation_points", [])
        if not isinstance(points, list):
            continue

        for point in points:
            if not isinstance(point, dict):
                continue
            evaluation = point.get("evaluation") or {}
            if not isinstance(evaluation, dict):
                evaluation = {}

            records.append({
                "leaf_theme_id": leaf_id,
                "leaf_theme_title": leaf_theme.get("title", ""),
                "year": point.get("year"),
                "paper_title": point.get("paper_title"),
                "innovation_content": point.get("innovation_content"),
                "rating": evaluation.get("rating"),
                "assessment": evaluation.get("assessment"),
            })

    records.sort(key=lambda x: ((x.get("year") is None), x.get("year") or 9999, safe_text(x.get("paper_title"))))

    lines = []
    for idx, record in enumerate(records, 1):
        lines.append(
            f"Aggregate Item {idx}\n"
            f"Leaf Theme ID: {record.get('leaf_theme_id')}\n"
            f"Leaf Theme Title: {safe_text(record.get('leaf_theme_title'))}\n"
            f"Publication Year: {record.get('year')}\n"
            f"Paper Title: {safe_text(record.get('paper_title')) or 'Unknown paper'}\n"
            f"Leaf History Innovation Content: {truncate_text(record.get('innovation_content'), 650)}\n"
            f"Leaf History Rating: {safe_text(record.get('rating')) or 'Unknown'}\n"
            f"Leaf History Assessment: {truncate_text(record.get('assessment'), 420)}"
        )

    aggregate_theme["members"] = sort_members_by_year(aggregate_members)
    return "\n\n".join(lines), records


def get_history_prompts(theme, source_material_text):
    system_prompt = (
        "You are an expert in structuring the development history of academic-paper innovation points. "
        "You are skilled at integrating publication time, paper titles, summaries, innovation claims, final decisions, "
        "and peer-review signals into a precise, high-detail, structured expert record. Your output will later be used "
        "for visualization, analysis, and as expert knowledge for another LLM to judge the novelty of a new innovation point "
        "and compare it against prior work. Therefore, you must preserve concrete technical content, accurately describe what "
        "each innovation point actually is, make the evolution of the category clear over time, and output a simple JSON object. "
        "You must cover every innovation point and must not omit any point. You should use review information selectively and "
        "critically: prioritize PC/AC meta-review over conflicting reviewer comments, use reviewer scores together with comment "
        "language and final decision to judge historical standing, ignore review comments that are irrelevant to the specific "
        "innovation point being discussed, and reason independently when the review evidence is absent, weak, or clearly unrelated."
    )

    user_prompt = f"""
You are given one category of academic innovation points, together with their source documents, publication years, paper titles, paper summaries, final decisions, and review comments.

Category metadata:
- Theme ID: {theme['theme_id']}
- Theme Title: {theme.get('title', '')}
- Theme Description: {theme.get('description', '')}

Your task:
1. Output ONLY a JSON object. Do not output markdown, explanation, or prose outside JSON.
2. The JSON object must follow this schema exactly:
{{
  "theme_overview": "Short structured overview of what this theme is about and how it evolves overall.",
  "innovation_points": [
    {{
      "item_number": 1,
      "year": 2024,
      "paper_title": "Exact paper title from source material",
      "innovation_content": "A detailed, faithful paragraph that should include: what the innovation point itself is; which paper it comes from; what that paper is broadly about / what problem it tackles; and, after checking earlier points in this theme, the concrete similarities and differences between this point and earlier related points in this subfield, especially at the method/content level when such comparison is supported by the source material.",
      "evaluation": {{
        "rating": "One of: major_breakthrough, important_advance, strong_extension, incremental_refinement, limited_contribution, mixed_or_uncertain",
        "assessment": "A complete evaluation paragraph that directly explains its historical standing, historical value, and why that judgment is reasonable."
      }}
    }}
  ],
  "overall_evolution_summary": "A detailed chronological narrative of the theme's historical trajectory."
}}
3. The `innovation_points` list must be in chronological order as much as possible.
4. For EACH innovation point, explain its concrete content in enough detail for downstream comparison, following the original innovation-point text closely. The `innovation_content` field should not only summarize the innovation point itself, but also state which paper it comes from and what that paper is broadly about in relation to this innovation point.
5. For EACH innovation point, also assess its historical standing in the line of work. Explain whether it appears to be a major breakthrough, an important advance, a strong extension, an incremental refinement, a limited contribution, or mixed/uncertain, and justify that judgment.
6. Use the paper summary, paper title, and innovation-point text as the primary evidence for understanding the innovation itself.
7. Use review comments, reviewer scores, meta-review, and final decision as auxiliary evidence for historical standing and significance, not as blind substitutes for reasoning.
8. Treat the final decision granularity as informative when available: for example, a stronger decision such as oral/spotlight generally signals higher impact or stronger recognition than poster, but still interpret that signal together with the actual innovation content and the rest of the evidence.
9. When review opinions conflict, prioritize PC/AC meta-review and the final decision over individual reviewer comments. If the meta-review explicitly indicates that a reviewer concern should be discounted or overridden, follow that signal.
10. Do NOT blindly use all review comments. For each innovation point, only use the parts of the review record that are actually relevant to that innovation point. If the review text mainly discusses unrelated parts of the paper or overall presentation issues, ignore those parts.
11. If review comments are absent, mixed, or not relevant to the innovation point, reason independently from the paper title, summary, innovation text, and surrounding chronology.
12. For each innovation point, check earlier points in this theme and explain the concrete similarities and differences within this subfield whenever the source material supports it. This comparison should be written inside `innovation_content`, especially focusing on methodological/content continuity, changes, refinements, extensions, or shifts.
13. Explain how later works build on, modify, extend, refine, or shift from earlier works whenever the source material supports such relations.
14. Make the structured history useful for later comparison analysis by clearly surfacing both continuity and distinction: what remains similar across points, what changes, what gets added, what gets refined, and what direction the line of work moves toward over time. If a point is related to earlier points, you should describe that directly inside `innovation_content` rather than creating a separate relation field.
15. You MUST cover all innovation points in the source material. Do not merge away, skip, or ignore any item, even if multiple points come from the same paper.
16. Make sure every Item in the source material is reflected somewhere in the final JSON. Before finishing, internally check that all Item numbers have been covered.
17. The `overall_evolution_summary` must NOT be a short placeholder. Write it as a self-contained chronological narrative that walks through the theme over time. It should mention the important years and papers, explain what the earliest works were doing, how later works changed direction, extended earlier ideas, or introduced new capabilities, and what the overall trajectory of this small field looks like. For leaf themes, it should still be concise enough to avoid repeating every detail, but it must clearly explain the temporal development rather than merely restating the theme.

Important constraints:
- Only use the provided source material.
- Preserve chronological order as much as possible.
- The result should be structured JSON, not prose outside JSON.
- Do not include extra keys beyond the schema above.
- The history should be detailed enough to support downstream novelty judgment, side-by-side comparison, timeline visualization, and expert-memory use.
- Do not say that some points were omitted, merged, or unavailable.
- When using review signals, do not quote them mechanically without interpretation; integrate them into your own expert judgment.

Source material:

{source_material_text}
""".strip()

    return system_prompt, user_prompt


def get_aggregate_history_prompts(theme, aggregate_source_material_text, source_kind, source_item_count):
    system_prompt = (
        "You are an expert at synthesizing concise, structured development histories for aggregate academic innovation themes. "
        "You will be given detailed leaf-level development histories for the leaf nodes that are directly attached to this aggregate node. "
        "Your task is not to re-analyze original papers. Instead, only use the provided leaf-level history material to write the aggregate node's own per-paper innovation history. "
        "You MUST cover every provided Aggregate Item exactly once, but each item should be much shorter than in the leaf histories: usually 1-2 sentences preserving year, source paper, condensed innovation content, and brief significance."
    )
    source_description = "direct leaf-level history material"
    coverage_rule = (
        f"4. Include every Aggregate Item from the source material exactly once in `innovation_points`. "
        f"The `innovation_points` list MUST contain exactly {source_item_count} objects."
    )
    source_ids_line = f"- Direct Leaf Theme IDs: {', '.join(theme.get('direct_leaf_theme_ids', []))}"

    user_prompt = f"""
You are given direct leaf-level development-history items belonging to one aggregate academic theme.

Aggregate theme metadata:
- Theme ID: {theme['theme_id']}
- Theme Title: {theme.get('title', '')}
- Theme Description: {theme.get('description', '')}
- Aggregate Source: direct leaf histories only
{source_ids_line}
- Number of source Aggregate Items: {source_item_count}

Your task:
1. Output ONLY a JSON object. Do not output markdown, explanation, or prose outside JSON.
2. Use ONLY the {source_description} below. Do not infer from original papers or add outside knowledge.
3. The JSON object must follow this schema exactly:
{{
  "theme_overview": "A concise overview of this aggregate node's direct leaf themes.",
  "innovation_points": [
    {{
      "item_number": 1,
      "year": 2024,
      "paper_title": "Exact paper title from the source material",
      "innovation_content": "One or two concise sentences summarizing what this paper contributed in this broader theme, including its relation to nearby earlier/later items when clear from the provided histories.",
      "evaluation": {{
        "rating": "One of: major_breakthrough, important_advance, strong_extension, incremental_refinement, limited_contribution, mixed_or_uncertain",
        "assessment": "One concise sentence summarizing historical standing/significance based only on the provided histories."
      }}
    }}
  ],
  "overall_evolution_summary": "A chronological narrative of the direct leaf items in this aggregate node."
}}
{coverage_rule}
5. Preserve chronological order as much as possible.
6. Keep each innovation point much shorter than a leaf-level history. Do not repeat detailed review evidence or long technical explanations.
7. Still preserve useful comparison signal: what each paper did, how the direction changed over time, and whether the item was major, important, incremental, limited, or mixed.
8. Do not include extra keys beyond the schema above.
9. This call only writes the aggregate node's own direct-leaf history. If the node also has non-leaf child branches, those child aggregate histories will be combined later when rewriting `overall_evolution_summary`; do not invent content for child branches that is not present in the source material here.

{source_description.capitalize()}:

{aggregate_source_material_text}
""".strip()

    return system_prompt, user_prompt


def process_single_theme(theme, by_source, by_paper_id, client, config):
    print(f"\n{'=' * 80}")
    print(f"[INFO] Writing history for theme {theme['theme_id']} | {theme.get('title')} | members={len(theme.get('members', []))}")
    print(f"{'=' * 80}")

    source_material_text = build_theme_source_material(theme, by_source, by_paper_id)
    system_prompt, user_prompt = get_history_prompts(theme, source_material_text)
    history_structured = normalize_history_structure(
        call_llm_json(client, config, system_prompt, user_prompt)
    )
    history_text = render_structured_history_as_text(history_structured)

    result = dict(theme)
    result["history_scope"] = "leaf"
    result["development_history"] = history_text
    result["development_history_structured"] = history_structured
    return result


def call_validated_aggregate_history(client, config, system_prompt, user_prompt, expected_count=None, require_exact_count=False, min_count=1):
    validation_attempts = min(3, max(1, int(config.get("llm_config", {}).get("max_retries", 3))))
    feedback = ""
    last_error = None

    for attempt in range(validation_attempts):
        prompt = user_prompt + feedback
        history_structured = normalize_history_structure(
            call_llm_json(client, config, system_prompt, prompt)
        )
        actual_count = len(history_structured.get("innovation_points", []))

        if require_exact_count and expected_count is not None and actual_count != expected_count:
            last_error = (
                f"Expected exactly {expected_count} innovation_points, "
                f"but got {actual_count}."
            )
        elif actual_count < min_count:
            last_error = f"Expected at least {min_count} innovation_points, but got {actual_count}."
        else:
            return history_structured

        print(f"[WARN] Aggregate history validation failed attempt {attempt + 1}/{validation_attempts}: {last_error}")
        feedback = (
            "\n\nValidation feedback for retry:\n"
            f"- {last_error}\n"
            "- Rewrite the full JSON object from scratch.\n"
            "- Keep the same schema and use only the provided source material.\n"
        )

    raise RuntimeError(f"Aggregate history validation failed after {validation_attempts} attempts: {last_error}")


def summarize_structured_history_for_prompt(theme):
    structured = theme.get("development_history_structured") or {}
    points = structured.get("innovation_points", [])
    if not isinstance(points, list):
        points = []

    compact_points = []
    for point in points:
        if not isinstance(point, dict):
            continue
        evaluation = point.get("evaluation") or {}
        if not isinstance(evaluation, dict):
            evaluation = {}
        compact_points.append({
            "item_number": point.get("item_number"),
            "year": point.get("year"),
            "paper_title": safe_text(point.get("paper_title")),
            "innovation_content": truncate_text(point.get("innovation_content"), 520),
            "rating": safe_text(evaluation.get("rating")),
            "assessment": truncate_text(evaluation.get("assessment"), 320),
        })

    return {
        "theme_id": safe_text(theme.get("theme_id")),
        "title": safe_text(theme.get("title")),
        "description": safe_text(theme.get("description")),
        "theme_overview": safe_text(structured.get("theme_overview")),
        "innovation_points": compact_points,
        "overall_evolution_summary": safe_text(structured.get("overall_evolution_summary")),
    }


def build_child_aggregate_display_payload(child_aggregate_themes):
    payload = []
    for child_theme in child_aggregate_themes:
        payload.append(summarize_structured_history_for_prompt(child_theme))
    return payload


def build_combined_aggregate_summary_source(aggregate_theme, child_aggregate_themes):
    return {
        "aggregate_theme": {
            "theme_id": safe_text(aggregate_theme.get("theme_id")),
            "title": safe_text(aggregate_theme.get("title")),
            "description": safe_text(aggregate_theme.get("description")),
            "direct_leaf_theme_ids": aggregate_theme.get("direct_leaf_theme_ids", []),
            "child_aggregate_theme_ids": aggregate_theme.get("child_aggregate_theme_ids", []),
        },
        "own_direct_leaf_history": summarize_structured_history_for_prompt(aggregate_theme),
        "child_aggregate_histories": [
            summarize_structured_history_for_prompt(child_theme)
            for child_theme in child_aggregate_themes
        ],
    }


def get_combined_aggregate_overall_summary_prompts(aggregate_theme, child_aggregate_themes):
    has_children = bool(child_aggregate_themes)
    coverage_rule = (
        "The overall summary MUST combine two kinds of evidence: this node's own direct-leaf history and the already generated child aggregate histories. "
        "Do not repeat child branches item-by-item if they are already summarized, but do explain their main historical role and how they connect to this node's own direct-leaf items."
        if has_children else
        "The overall summary should cover this node's own direct-leaf history in chronological order."
    )
    system_prompt = (
        "You are an expert at writing the final overall chronological summary for an aggregate academic innovation theme. "
        "You are given an aggregate node's own direct-leaf JSON plus any already generated child aggregate JSONs. "
        "Your job is to produce only a stronger `overall_evolution_summary` for the whole aggregate node. "
        "Do not rewrite the per-paper innovation_points."
    )
    source_json = json.dumps(
        build_combined_aggregate_summary_source(aggregate_theme, child_aggregate_themes),
        ensure_ascii=False,
        indent=2,
    )
    user_prompt = f"""
Rewrite ONLY the `overall_evolution_summary` for this aggregate node.

Rules:
1. Output ONLY a JSON object with exactly this schema:
{{
  "overall_evolution_summary": "..."
}}
2. Use only the provided aggregate JSON material.
3. Write a chronological narrative, not a short placeholder.
4. Mention important years and paper titles when they make the timeline clearer.
5. Explain how the theme starts, how later work changes or extends it, and how child branches relate to the node's own direct-leaf items.
6. {coverage_rule}
7. The summary should be detailed enough for expert-memory use and visualization, usually 8-18 sentences depending on the amount of material.

Aggregate JSON material:

{source_json}
""".strip()
    return system_prompt, user_prompt


def rewrite_aggregate_overall_summary_with_children(aggregate_theme, child_aggregate_themes, client, config):
    aggregate_theme = dict(aggregate_theme)
    aggregate_theme["child_aggregate_histories_for_display"] = build_child_aggregate_display_payload(child_aggregate_themes)

    system_prompt, user_prompt = get_combined_aggregate_overall_summary_prompts(aggregate_theme, child_aggregate_themes)
    validation_attempts = min(3, max(1, int(config.get("llm_config", {}).get("max_retries", 3))))
    feedback = ""
    for attempt in range(validation_attempts):
        response_json = call_llm_json(client, config, system_prompt, user_prompt + feedback)
        new_summary = safe_text(response_json.get("overall_evolution_summary"))
        if new_summary:
            updated_theme = dict(aggregate_theme)
            structured = dict(updated_theme.get("development_history_structured") or {})
            structured["overall_evolution_summary"] = new_summary
            updated_theme["development_history_structured"] = normalize_history_structure(structured)
            updated_theme["development_history"] = render_structured_history_as_text(updated_theme["development_history_structured"])
            return updated_theme

        print(
            f"[WARN] Empty aggregate overall summary for {aggregate_theme.get('theme_id')} "
            f"attempt {attempt + 1}/{validation_attempts}; retrying"
        )
        feedback = (
            "\n\nValidation feedback for retry:\n"
            "- Your previous response had an empty or missing `overall_evolution_summary`.\n"
            "- Return a non-empty chronological summary string.\n"
            "- Do not return markdown or extra keys.\n"
        )

    print(f"[WARN] Keeping existing aggregate overall summary for {aggregate_theme.get('theme_id')}")
    return aggregate_theme


def process_aggregate_theme(aggregate_theme, leaf_history_lookup, aggregate_history_lookup, client, config):
    print(f"\n{'=' * 80}")
    print(
        f"[INFO] Writing aggregate history for node {aggregate_theme['theme_id']} | "
        f"{aggregate_theme.get('title')} | direct_leaves={len(aggregate_theme.get('direct_leaf_theme_ids', []))} | "
        f"child_aggregates={len(aggregate_theme.get('child_aggregate_theme_ids', []))}"
    )
    print(f"{'=' * 80}")

    source_material_text, records = build_parent_aggregate_source_material(aggregate_theme, leaf_history_lookup)
    if not records:
        raise ValueError(f"Aggregate node {aggregate_theme['theme_id']} has no available leaf-history records")

    system_prompt, user_prompt = get_aggregate_history_prompts(
        aggregate_theme,
        source_material_text,
        source_kind="parent",
        source_item_count=len(records),
    )
    history_structured = call_validated_aggregate_history(
        client,
        config,
        system_prompt,
        user_prompt,
        expected_count=len(records),
        require_exact_count=True,
        min_count=len(records),
    )

    result = dict(aggregate_theme)
    result["history_scope"] = "aggregate"
    result["aggregate_level"] = "aggregate"
    result["aggregate_source"] = "direct_leaf_histories"
    result["aggregation_strategy"] = "direct_leaf_plus_child_aggregates"
    result["coverage_policy"] = "cover_direct_leaf_items"
    result["num_direct_leaf_themes"] = len(result.get("direct_leaf_theme_ids", []))
    result["num_child_aggregate_themes"] = len(result.get("child_aggregate_theme_ids", []))
    result["num_leaf_themes"] = len(result.get("descendant_leaf_theme_ids", []))
    result["num_aggregate_items"] = len(history_structured.get("innovation_points", []))
    result["development_history_structured"] = history_structured
    result["development_history"] = render_structured_history_as_text(history_structured)

    child_aggregate_themes = [
        aggregate_history_lookup[child_id]
        for child_id in result.get("child_aggregate_theme_ids", [])
        if child_id in aggregate_history_lookup
    ]
    return rewrite_aggregate_overall_summary_with_children(result, child_aggregate_themes, client, config)


def classify_history_kind(theme):
    if theme.get("history_scope") != "aggregate":
        return "leaf"
    return "aggregate"


def should_rewrite_overall_summary(theme, scope):
    history_kind = classify_history_kind(theme)
    if scope == "all":
        return True
    if scope == "aggregate":
        return history_kind == "aggregate"
    return history_kind == scope


def build_overall_summary_source(theme):
    structured = theme.get("development_history_structured") or {}
    points = structured.get("innovation_points", [])
    if not isinstance(points, list):
        points = []

    source_points = []
    for point in points:
        if not isinstance(point, dict):
            continue
        evaluation = point.get("evaluation") or {}
        if not isinstance(evaluation, dict):
            evaluation = {}
        source_points.append({
            "item_number": point.get("item_number"),
            "year": point.get("year"),
            "paper_title": safe_text(point.get("paper_title")),
            "innovation_content": safe_text(point.get("innovation_content")),
            "rating": safe_text(evaluation.get("rating")),
            "assessment": safe_text(evaluation.get("assessment")),
        })

    return {
        "theme_id": safe_text(theme.get("theme_id")),
        "title": safe_text(theme.get("title")),
        "description": safe_text(theme.get("description")),
        "history_kind": classify_history_kind(theme),
        "aggregate_level": safe_text(theme.get("aggregate_level")),
        "theme_overview": safe_text(structured.get("theme_overview")),
        "innovation_points": source_points,
        "current_overall_evolution_summary": safe_text(structured.get("overall_evolution_summary")),
    }


def get_overall_summary_rewrite_prompts(theme):
    history_kind = classify_history_kind(theme)
    if history_kind == "aggregate":
        coverage_rule = (
            "This is an aggregate theme. The new overall_evolution_summary is the most important field for downstream use. "
            "It should explicitly cover the direct-leaf innovation points in the provided JSON and, when child-branch summary information is present, explain how those child branches fit into the broader theme."
        )
        length_rule = "Write a detailed but readable timeline narrative, normally 8-18 sentences depending on the number of points."
    elif history_kind == "leaf":
        coverage_rule = (
            "This is a leaf theme. The summary should cover the complete local trajectory of the leaf theme and mention the main years "
            "and papers without repeating all low-level detail from each innovation point."
        )
        length_rule = "Write a clear timeline narrative, normally 5-12 sentences depending on the number of points."
    else:
        coverage_rule = (
            "This is an aggregate theme. Write a timeline narrative that covers the important stages and preserves useful expert-memory signal."
        )
        length_rule = "Write a detailed but readable timeline narrative."

    system_prompt = (
        "You are an expert at rewriting the overall chronological development summary for academic innovation-memory themes. "
        "You are given an existing structured history JSON that already contains per-paper innovation content and evaluations. "
        "Do not rewrite the per-paper items. Your only job is to produce a better `overall_evolution_summary`: a self-contained, "
        "chronological narrative that explains how this small field developed over time."
    )

    source_json = json.dumps(build_overall_summary_source(theme), ensure_ascii=False, indent=2)
    user_prompt = f"""
Rewrite ONLY the `overall_evolution_summary` for the following existing structured history.

Rules:
1. Output ONLY a JSON object with exactly this schema:
{{
  "overall_evolution_summary": "..."
}}
2. Use only the provided structured history JSON. Do not add outside knowledge.
3. Do not change, rewrite, or re-output `innovation_points`; only return the new summary string.
4. The summary must read like a chronological history of this theme, not like a short placeholder.
5. It should explicitly use years and paper titles when they help make the timeline clear.
6. It should explain what the earliest work was doing, what later work added or changed, and what direction the theme moved toward.
7. {coverage_rule}
8. {length_rule}

Existing structured history JSON:

{source_json}
""".strip()

    return system_prompt, user_prompt


def rewrite_overall_summary_only(theme, client, config):
    print(f"[INFO] Rewriting overall summary for {theme.get('theme_id')} | {theme.get('title')}")
    structured = dict(theme.get("development_history_structured") or {})
    if not structured.get("innovation_points"):
        raise ValueError(f"Theme {theme.get('theme_id')} has no structured innovation_points")

    system_prompt, user_prompt = get_overall_summary_rewrite_prompts(theme)
    validation_attempts = min(3, max(1, int(config.get("llm_config", {}).get("max_retries", 3))))
    feedback = ""
    new_summary = ""

    for attempt in range(validation_attempts):
        response_json = call_llm_json(client, config, system_prompt, user_prompt + feedback)
        new_summary = safe_text(response_json.get("overall_evolution_summary"))
        if new_summary:
            break

        print(
            f"[WARN] Empty overall summary for {theme.get('theme_id')} "
            f"attempt {attempt + 1}/{validation_attempts}; retrying with validation feedback"
        )
        feedback = (
            "\n\nValidation feedback for retry:\n"
            "- Your previous JSON response had an empty or missing `overall_evolution_summary`.\n"
            "- Return a non-empty `overall_evolution_summary` string.\n"
            "- Do not return an empty string, null, markdown, or any keys other than `overall_evolution_summary`.\n"
            "- Use the same provided structured history JSON and write the full chronological narrative now.\n"
        )

    if not new_summary:
        fallback_summary = safe_text(structured.get("overall_evolution_summary"))
        if fallback_summary:
            print(
                f"[WARN] Keeping existing overall summary for {theme.get('theme_id')} "
                "because all rewrite attempts returned empty summaries"
            )
            updated_theme = dict(theme)
            updated_theme["overall_summary_rewrite_failed_at"] = datetime.now().isoformat(timespec="seconds")
            return updated_theme
        raise ValueError(f"Theme {theme.get('theme_id')} returned an empty overall summary")

    updated_theme = dict(theme)
    structured["overall_evolution_summary"] = new_summary
    updated_theme["development_history_structured"] = normalize_history_structure(structured)
    updated_theme["development_history"] = render_structured_history_as_text(updated_theme["development_history_structured"])
    updated_theme["overall_summary_rewritten_at"] = datetime.now().isoformat(timespec="seconds")
    return updated_theme


def load_existing_final_library(path=None):
    if path is None:
        path = os.path.join(FINAL_MEMORY_OUTPUT_DIR, "final_memory_library.json")
    with open(path, "r", encoding="utf-8") as f:
        library = json.load(f)
    print(f"[INFO] Loaded existing final library for summary rewrite: {path}")
    return library


def rewrite_existing_overall_summaries(client, config, args):
    library = load_existing_final_library()
    leaf_themes = list(library.get("themes", []) or [])
    aggregate_themes = list(library.get("aggregate_themes", []) or [])

    target_specs = []
    for index, theme in enumerate(leaf_themes):
        if should_rewrite_overall_summary(theme, args.overall_summary_scope):
            target_specs.append(("leaf", index, theme))
    for index, theme in enumerate(aggregate_themes):
        if should_rewrite_overall_summary(theme, args.overall_summary_scope):
            target_specs.append(("aggregate", index, theme))

    if args.max_themes > 0:
        target_specs = target_specs[:args.max_themes]

    print(f"[INFO] Overall-summary rewrite scope: {args.overall_summary_scope}")
    print(f"[INFO] Overall summaries to rewrite: {len(target_specs)}")

    failures = []
    targets_lock = threading.Lock()

    def summary_worker(target_kind, index, theme):
        try:
            updated_theme = rewrite_overall_summary_only(theme, client, config)
            if target_kind == "leaf":
                save_single_theme(updated_theme)
            else:
                save_single_aggregate_theme(updated_theme)
            with targets_lock:
                if target_kind == "leaf":
                    leaf_themes[index] = updated_theme
                else:
                    aggregate_themes[index] = updated_theme
        except Exception as e:
            print(f"[ERROR] Failed overall-summary rewrite for {theme.get('theme_id')}: {e}")
            traceback.print_exc()
            with targets_lock:
                failures.append({"theme_id": theme.get("theme_id"), "error": str(e)})

    if target_specs:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(summary_worker, target_kind, index, theme)
                for target_kind, index, theme in target_specs
            ]
            for _ in as_completed(futures):
                pass

    save_library(leaf_themes, aggregate_themes)
    print(f"[INFO] Overall-summary rewrite failures: {len(failures)}")
    return failures


def load_existing_history_lookup():
    lookup = {}
    library_path = os.path.join(FINAL_MEMORY_OUTPUT_DIR, "final_memory_library.json")

    if os.path.exists(library_path):
        try:
            with open(library_path, "r", encoding="utf-8") as f:
                library = json.load(f)
            for theme in library.get("themes", []) or []:
                theme_id = safe_text(theme.get("theme_id"))
                if theme_id and theme.get("development_history_structured"):
                    lookup[theme_id] = theme
            for theme in library.get("aggregate_themes", []) or []:
                theme_id = safe_text(theme.get("theme_id"))
                if theme_id and theme.get("development_history_structured"):
                    lookup[theme_id] = theme
        except Exception as e:
            print(f"[WARN] Failed to load existing final library {library_path}: {e}")

    if os.path.isdir(FINAL_MEMORY_OUTPUT_DIR):
        for filename in os.listdir(FINAL_MEMORY_OUTPUT_DIR):
            if not (
                filename.startswith("final_theme_")
                or filename.startswith("aggregate_theme_")
            ) or not filename.endswith(".json"):
                continue
            path = os.path.join(FINAL_MEMORY_OUTPUT_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    theme = json.load(f)
                theme_id = safe_text(theme.get("theme_id"))
                if theme_id and theme.get("development_history_structured"):
                    lookup[theme_id] = theme
            except Exception as e:
                print(f"[WARN] Failed to load existing theme file {path}: {e}")

    print(f"[INFO] Existing histories available for reuse: {len(lookup)}")
    return lookup


def save_single_theme(theme):
    os.makedirs(FINAL_MEMORY_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(FINAL_MEMORY_OUTPUT_DIR, f"final_theme_{theme['theme_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(theme, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {path}")
    return path


def save_single_aggregate_theme(theme):
    os.makedirs(FINAL_MEMORY_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(FINAL_MEMORY_OUTPUT_DIR, f"aggregate_theme_{theme['theme_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(theme, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {path}")
    return path


def save_library(themes, aggregate_themes=None):
    aggregate_themes = aggregate_themes or []
    os.makedirs(FINAL_MEMORY_OUTPUT_DIR, exist_ok=True)
    library = {
        "num_themes": len(themes),
        "num_leaf_themes": len(themes),
        "num_aggregate_themes": len(aggregate_themes),
        "total_points": sum(len(t.get("members", [])) for t in themes),
        "themes": themes,
        "aggregate_themes": aggregate_themes,
    }
    path = os.path.join(FINAL_MEMORY_OUTPUT_DIR, "final_memory_library.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(library, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {path}")
    return path


def merge_reused_aggregate_history(existing_theme, candidate_theme):
    reused = dict(existing_theme)
    metadata_keys = [
        "source_cluster_id",
        "title",
        "description",
        "members",
        "history_scope",
        "aggregate_level",
        "aggregate_source",
        "aggregation_strategy",
        "coverage_policy",
        "direct_leaf_theme_ids",
        "descendant_leaf_theme_ids",
        "child_aggregate_theme_ids",
        "child_aggregate_histories_for_display",
        "tree_depth",
        "num_direct_leaf_themes",
        "num_child_aggregate_themes",
        "num_leaf_themes",
    ]
    for key in metadata_keys:
        if key in candidate_theme:
            if key == "members" and not candidate_theme.get("members"):
                continue
            reused[key] = candidate_theme[key]
    reused.setdefault("history_scope", "aggregate")
    return reused


def can_reuse_aggregate(existing_theme):
    if not existing_theme or not existing_theme.get("development_history_structured"):
        return False
    return (
        existing_theme.get("history_scope") == "aggregate"
        and existing_theme.get("aggregate_level") == "aggregate"
        and existing_theme.get("aggregate_source") == "direct_leaf_histories"
        and existing_theme.get("aggregation_strategy") == "direct_leaf_plus_child_aggregates"
        and existing_theme.get("coverage_policy") == "cover_direct_leaf_items"
    )


def main():
    parser = argparse.ArgumentParser(description="Write leaf histories and direct-leaf aggregate histories for refined themes")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--clusters", type=str, default="", help="Optional filter by source coarse cluster ids")
    parser.add_argument("--max-clusters", type=int, default=0, help="Only use themes coming from the first N coarse clusters (0 = all)")
    parser.add_argument("--max-themes", type=int, default=0, help="Only process the first N final themes after flattening (0 = all)")
    parser.add_argument("--workers", type=int, default=128, help="Number of parallel theme workers (default: 128)")
    parser.add_argument("--rewrite-leaf-histories", action="store_true", help="Rewrite leaf histories instead of reusing existing ones")
    parser.add_argument("--rewrite-aggregate-histories", action="store_true", help="Rewrite aggregate histories instead of reusing existing ones")
    parser.add_argument("--skip-aggregate-histories", action="store_true", help="Only write/reuse leaf histories and skip aggregate histories")
    parser.add_argument("--rewrite-overall-summaries-only", action="store_true", help="Only rewrite existing overall_evolution_summary fields; keep existing innovation_points unchanged")
    parser.add_argument(
        "--overall-summary-scope",
        choices=["all", "leaf", "aggregate"],
        default="all",
        help="Scope for --rewrite-overall-summaries-only (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"write_cluster_development_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout
    print(f"[INFO] Log file: {log_path}")

    config = load_config(args.config)
    client = build_client(config)

    if args.rewrite_overall_summaries_only:
        failures = rewrite_existing_overall_summaries(client, config, args)
        print(f"\n{'=' * 80}")
        print("DONE")
        print("  Mode:                       rewrite overall summaries only")
        print(f"  Scope:                      {args.overall_summary_scope}")
        print(f"  Failures:                   {len(failures)}")
        print(f"  Output dir:                 {FINAL_MEMORY_OUTPUT_DIR}")
        print(f"{'=' * 80}")
        return

    refined_library = load_refined_library()
    original_memory = load_original_memory()
    by_source, by_paper_id = build_memory_lookup(original_memory)
    existing_history_lookup = load_existing_history_lookup()

    clusters = refined_library.get("clusters", [])
    if args.clusters.strip():
        wanted = {int(x.strip()) for x in args.clusters.split(",") if x.strip()}
        clusters = [c for c in clusters if int(c["cluster_id"]) in wanted]
    elif args.max_clusters > 0:
        clusters = clusters[:args.max_clusters]

    themes = flatten_final_themes({"clusters": clusters})
    if args.max_themes > 0:
        themes = themes[:args.max_themes]

    print(f"[INFO] Selected source coarse clusters: {len(clusters)}")
    print(f"[INFO] Leaf themes to write/reuse: {len(themes)}")
    print(f"[INFO] Parallel workers: {args.workers}")

    results = []
    reused_leaf_count = 0
    failures = []
    results_lock = threading.Lock()
    failures_lock = threading.Lock()

    missing_leaf_themes = []
    for theme in themes:
        existing_theme = existing_history_lookup.get(theme.get("theme_id"))
        if existing_theme and not args.rewrite_leaf_histories:
            reused = dict(existing_theme)
            reused.setdefault("history_scope", "leaf")
            results.append(reused)
            reused_leaf_count += 1
        else:
            missing_leaf_themes.append(theme)

    print(f"[INFO] Reused existing leaf histories: {reused_leaf_count}")
    print(f"[INFO] Leaf histories requiring LLM generation: {len(missing_leaf_themes)}")

    def worker(theme):
        try:
            updated_theme = process_single_theme(theme, by_source, by_paper_id, client, config)
            save_single_theme(updated_theme)
            with results_lock:
                results.append(updated_theme)
        except Exception as e:
            print(f"[ERROR] Failed theme {theme.get('theme_id')}: {e}")
            traceback.print_exc()
            with failures_lock:
                failures.append({"theme_id": theme.get("theme_id"), "error": str(e)})

    if missing_leaf_themes:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker, theme) for theme in missing_leaf_themes]
            for _ in as_completed(futures):
                pass

    results.sort(key=lambda x: x["theme_id"])

    leaf_history_lookup = {theme["theme_id"]: theme for theme in results if theme.get("theme_id")}
    aggregate_results = []
    aggregate_failures = []
    reused_aggregate_count = 0

    if not args.skip_aggregate_histories:
        aggregate_candidates = collect_direct_leaf_parent_aggregate_themes({"clusters": clusters})
        available_leaf_ids = set(leaf_history_lookup.keys())
        filtered_aggregate_candidates = []
        for aggregate_theme in aggregate_candidates:
            leaf_ids = [
                leaf_id for leaf_id in aggregate_theme.get("direct_leaf_theme_ids", [])
                if leaf_id in available_leaf_ids
            ]
            if not leaf_ids:
                continue
            aggregate_theme["direct_leaf_theme_ids"] = leaf_ids
            filtered_aggregate_candidates.append(aggregate_theme)

        print(f"[INFO] Aggregate nodes to write/reuse: {len(filtered_aggregate_candidates)}")

        candidates_by_depth = {}
        for aggregate_theme in filtered_aggregate_candidates:
            depth = int(aggregate_theme.get("tree_depth", 0))
            candidates_by_depth.setdefault(depth, []).append(aggregate_theme)

        aggregate_history_lookup = {}
        aggregate_results_lock = threading.Lock()
        aggregate_failures_lock = threading.Lock()
        results_by_theme_id = {}

        def aggregate_worker(aggregate_theme, child_lookup_snapshot):
            try:
                updated_theme = process_aggregate_theme(
                    aggregate_theme,
                    leaf_history_lookup,
                    child_lookup_snapshot,
                    client,
                    config,
                )
                save_single_aggregate_theme(updated_theme)
                with aggregate_results_lock:
                    aggregate_history_lookup[updated_theme["theme_id"]] = updated_theme
                    results_by_theme_id[updated_theme["theme_id"]] = updated_theme
            except Exception as e:
                print(f"[ERROR] Failed aggregate theme {aggregate_theme.get('theme_id')}: {e}")
                traceback.print_exc()
                with aggregate_failures_lock:
                    aggregate_failures.append({"theme_id": aggregate_theme.get("theme_id"), "error": str(e)})

        for depth in sorted(candidates_by_depth.keys(), reverse=True):
            depth_candidates = sorted(candidates_by_depth[depth], key=lambda x: x["theme_id"])
            missing_aggregate_themes = []
            reused_at_depth = 0

            for aggregate_theme in depth_candidates:
                child_ids = [
                    child_id for child_id in aggregate_theme.get("child_aggregate_theme_ids", [])
                    if child_id in aggregate_history_lookup
                ]
                aggregate_theme["child_aggregate_theme_ids"] = child_ids

                existing_aggregate = existing_history_lookup.get(aggregate_theme.get("theme_id"))
                if (
                    existing_aggregate
                    and not args.rewrite_aggregate_histories
                    and can_reuse_aggregate(existing_aggregate)
                ):
                    reused = merge_reused_aggregate_history(existing_aggregate, aggregate_theme)
                    reused["history_scope"] = "aggregate"
                    reused["aggregate_level"] = "aggregate"
                    reused["aggregate_source"] = "direct_leaf_histories"
                    reused["aggregation_strategy"] = "direct_leaf_plus_child_aggregates"
                    reused["coverage_policy"] = "cover_direct_leaf_items"
                    reused["num_direct_leaf_themes"] = len(reused.get("direct_leaf_theme_ids", []))
                    reused["num_child_aggregate_themes"] = len(reused.get("child_aggregate_theme_ids", []))
                    reused["num_leaf_themes"] = len(reused.get("descendant_leaf_theme_ids", []))
                    aggregate_history_lookup[reused["theme_id"]] = reused
                    results_by_theme_id[reused["theme_id"]] = reused
                    reused_aggregate_count += 1
                    reused_at_depth += 1
                else:
                    missing_aggregate_themes.append(aggregate_theme)

            print(
                f"[INFO] Aggregate depth {depth}: "
                f"reused={reused_at_depth}, requiring LLM generation={len(missing_aggregate_themes)}"
            )

            if missing_aggregate_themes:
                child_lookup_snapshot = dict(aggregate_history_lookup)
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = [
                        executor.submit(aggregate_worker, theme, child_lookup_snapshot)
                        for theme in missing_aggregate_themes
                    ]
                    for _ in as_completed(futures):
                        pass

        aggregate_results = list(results_by_theme_id.values())
        aggregate_results.sort(key=lambda x: x["theme_id"])

    combined_path = None
    if results:
        combined_path = save_library(results, aggregate_results)

    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"  Leaf themes requested:       {len(themes)}")
    print(f"  Leaf histories reused:       {reused_leaf_count}")
    print(f"  Leaf histories available:    {len(results)}")
    print(f"  Leaf failures:               {len(failures)}")
    print(f"  Aggregate histories reused:  {reused_aggregate_count}")
    print(f"  Aggregate histories written: {len(aggregate_results)}")
    print(f"  Aggregate failures:          {len(aggregate_failures)}")
    print(f"  Output dir:                  {FINAL_MEMORY_OUTPUT_DIR}")
    if combined_path:
        print(f"  Combined library:            {combined_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
