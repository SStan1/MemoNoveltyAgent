"""
refine_cluster_memory.py
------------------------
Recursively refine coarse KMeans clusters into an LLM-generated tree of semantic categories.

Design:
1. Start from each coarse KMeans cluster
2. First write a title/description for the coarse cluster itself
3. If the cluster has more than 15 points, ask the LLM to split it into >= 2 semantic child groups
4. For every child group, write its own title/description and continue recursively
5. Stop recursion when a node has <= 15 points
6. Preserve the full tree structure, not just the leaves

Outputs:
  <project-root>/memory_database/refined_memory_output/
    refined_cluster_<cluster_id>.json
    refined_memory_library.json
    refined_memory_index.json
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONSTRUCT_MEMORY_DIR = str(PROJECT_ROOT / "Construct_memory")
CLUSTER_OUTPUT_DIR = str(PROJECT_ROOT / "cluster_output")
REFINED_OUTPUT_DIR = str(PROJECT_ROOT / "memory_database" / "refined_memory_output")
LOG_DIR = str(PROJECT_ROOT / "logs")
MAX_LEAF_SIZE = 15
VALIDATION_RETRY_LIMIT = 3


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
                reasoning={"effort": "low", "summary": "auto"},
                temperature=temperature,
                stream=False,
            )
            content = strip_json_fence(extract_response_text(response))
            if not content:
                raise ValueError("Empty LLM response")
            return json.loads(content)
        except Exception as e:
            last_error = e
            print(f"[WARN] LLM JSON call failed attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    raise RuntimeError(f"LLM JSON call failed after {max_retries} attempts: {last_error}")


def load_cluster_assignments(cluster_assignment_path=None):
    if cluster_assignment_path is None:
        cluster_assignment_path = os.path.join(CLUSTER_OUTPUT_DIR, "cluster_kmeans.json")

    with open(cluster_assignment_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[INFO] Loaded cluster assignments: {cluster_assignment_path}")
    print(f"[INFO] Total innovation points with source mapping: {len(data)}")
    return data


def group_assignments_by_cluster(assignments):
    grouped = defaultdict(list)
    for item in assignments:
        grouped[int(item["cluster"])].append(item)
    return grouped


def build_points_payload(items):
    lines = []
    for idx, item in enumerate(items, 1):
        point_text = re.sub(r"\s+", " ", item["point_text"]).strip()
        lines.append(
            f"Point {idx}\n"
            f"Source file: {item['source_filename']}\n"
            f"Paper title: {item['title']}\n"
            f"Year: {item.get('year')}\n"
            f"Original point index in paper: {item['point_index']}\n"
            f"Innovation text: {point_text}"
        )
    return "\n\n".join(lines)


def get_node_title_description_prompts(node_id, items):
    system_prompt = (
        "You are an expert at naming and describing coherent groups of academic innovation points. "
        "You should write a concise, human-meaningful title and a short description summarizing the common theme."
    )

    payload = build_points_payload(items)
    user_prompt = f"""
You are given one semantic group of innovation points.

Your task is to summarize this group by writing:
1. A short title
2. A short 1-2 sentence description

Return ONLY a JSON object with this schema:
{{
  "title": "Short theme title",
  "description": "Short 1-2 sentence description"
}}

The title should be concise and descriptive.
The description should summarize the common topic of the group.

Group ID: {node_id}

Innovation points:

{payload}
""".strip()

    return system_prompt, user_prompt


def get_split_prompts(node_id, items):
    system_prompt = (
        "You are an expert at recursively organizing academic innovation points into a semantic taxonomy. "
        "You should split a broad group into multiple coherent child groups based on theme, content, and semantics."
    )

    payload = build_points_payload(items)
    user_prompt = f"""
You are given one broad group of innovation points.

Your task is to split this group into child groups based on innovation theme, topic, and semantic content.

Rules:
1. You MUST split the group into at least 2 child groups.
2. Do not keep the original group unchanged.
3. Group points based on semantic coherence only.
4. Each point must belong to exactly one child group.
5. For EACH child group, provide:
   - a short title
   - a short 1-2 sentence description
   - the list of point numbers in that child group

Return ONLY a JSON object with this schema:
{{
  "groups": [
    {{
      "group_id": "G1",
      "title": "Short child title",
      "description": "Short child description",
      "point_numbers": [1, 2, 5]
    }}
  ]
}}

Important validation rules:
- Every point number must appear exactly once across all child groups.
- Do not omit any point.
- Do not invent point numbers.
- You must produce at least 2 child groups.

Group ID: {node_id}

Innovation points:

{payload}
""".strip()

    return system_prompt, user_prompt


def materialize_members(items, point_numbers, seen=None, strict=True):
    point_lookup = {idx: item for idx, item in enumerate(items, 1)}
    members = []

    if seen is None:
        seen = set()

    for point_num in point_numbers:
        if point_num not in point_lookup:
            if strict:
                raise ValueError(f"Invalid point number {point_num}")
            continue
        if point_num in seen:
            if strict:
                raise ValueError(f"Duplicate point number {point_num}")
            continue
        seen.add(point_num)
        item = point_lookup[point_num]
        members.append({
            "point_number": point_num,
            "embedding_idx": item["embedding_idx"],
            "paper_id": item["paper_id"],
            "source_filename": item["source_filename"],
            "title": item["title"],
            "year": item["year"],
            "point_index": item["point_index"],
            "point_text": item["point_text"],
        })
    return members


def validate_split_result(node_id, items, llm_result):
    groups = llm_result.get("groups")
    if not isinstance(groups, list) or len(groups) < 2:
        raise ValueError(f"Node {node_id}: invalid LLM result, must contain at least 2 groups")

    seen = set()
    children = []
    for group in groups:
        point_numbers = group.get("point_numbers", [])
        if not isinstance(point_numbers, list) or not point_numbers:
            continue
        members = materialize_members(items, point_numbers, seen=seen, strict=True)
        children.append({
            "node_id": None,
            "title": group.get("title", "").strip(),
            "description": group.get("description", "").strip(),
            "num_points": len(members),
            "members": members,
            "children": [],
            "is_leaf": len(members) <= MAX_LEAF_SIZE,
            "refinement_method": "llm_split",
        })

    all_points = set(range(1, len(items) + 1))
    missing = sorted(all_points - seen)
    if missing:
        raise ValueError(f"Node {node_id}: missing point assignments {missing}")
    if len(children) < 2:
        raise ValueError(f"Node {node_id}: fewer than 2 non-empty groups after validation")
    return children


def repair_split_result(node_id, items, llm_result):
    groups = llm_result.get("groups")
    if not isinstance(groups, list):
        raise ValueError(f"Node {node_id}: cannot repair invalid split result without groups")

    seen = set()
    children = []
    point_lookup = {idx: item for idx, item in enumerate(items, 1)}

    for group in groups:
        raw_numbers = group.get("point_numbers", [])
        if not isinstance(raw_numbers, list):
            raw_numbers = []
        members = materialize_members(items, raw_numbers, seen=seen, strict=False)
        if not members:
            continue
        children.append({
            "node_id": None,
            "title": group.get("title", "").strip() or "Refined child group",
            "description": group.get("description", "").strip() or "LLM-refined child group.",
            "num_points": len(members),
            "members": members,
            "children": [],
            "is_leaf": len(members) <= MAX_LEAF_SIZE,
            "refinement_method": "llm_split_repaired",
        })

    missing = sorted(set(point_lookup.keys()) - seen)
    if missing:
        missing_members = []
        for point_num in missing:
            item = point_lookup[point_num]
            missing_members.append({
                "point_number": point_num,
                "embedding_idx": item["embedding_idx"],
                "paper_id": item["paper_id"],
                "source_filename": item["source_filename"],
                "title": item["title"],
                "year": item["year"],
                "point_index": item["point_index"],
                "point_text": item["point_text"],
            })

        if children:
            children[-1]["members"].extend(missing_members)
            children[-1]["num_points"] = len(children[-1]["members"])
            children[-1]["description"] = (
                children[-1]["description"]
                + " Some unassigned points were appended automatically during post-processing."
            )
        else:
            midpoint = max(1, len(missing_members) // 2)
            first = missing_members[:midpoint]
            second = missing_members[midpoint:]
            if not second:
                second = first[-1:]
                first = first[:-1]
            children = [
                {
                    "node_id": None,
                    "title": "Recovered child group A",
                    "description": "Fallback child group created during repair.",
                    "num_points": len(first),
                    "members": first,
                    "children": [],
                    "is_leaf": len(first) <= MAX_LEAF_SIZE,
                    "refinement_method": "llm_split_repaired",
                },
                {
                    "node_id": None,
                    "title": "Recovered child group B",
                    "description": "Fallback child group created during repair.",
                    "num_points": len(second),
                    "members": second,
                    "children": [],
                    "is_leaf": len(second) <= MAX_LEAF_SIZE,
                    "refinement_method": "llm_split_repaired",
                },
            ]

    if len(children) < 2:
        members = [
            {
                "point_number": idx,
                "embedding_idx": item["embedding_idx"],
                "paper_id": item["paper_id"],
                "source_filename": item["source_filename"],
                "title": item["title"],
                "year": item["year"],
                "point_index": item["point_index"],
                "point_text": item["point_text"],
            }
            for idx, item in enumerate(items, 1)
        ]
        midpoint = max(1, len(members) // 2)
        if midpoint >= len(members):
            midpoint = len(members) - 1
        children = [
            {
                "node_id": None,
                "title": "Recovered child group A",
                "description": "Fallback split created during repair.",
                "num_points": len(members[:midpoint]),
                "members": members[:midpoint],
                "children": [],
                "is_leaf": len(members[:midpoint]) <= MAX_LEAF_SIZE,
                "refinement_method": "llm_split_repaired",
            },
            {
                "node_id": None,
                "title": "Recovered child group B",
                "description": "Fallback split created during repair.",
                "num_points": len(members[midpoint:]),
                "members": members[midpoint:],
                "children": [],
                "is_leaf": len(members[midpoint:]) <= MAX_LEAF_SIZE,
                "refinement_method": "llm_split_repaired",
            },
        ]

    return children


def write_node_summary(node_id, items, client, config):
    system_prompt, user_prompt = get_node_title_description_prompts(node_id, items)
    result = call_llm_json(client, config, system_prompt, user_prompt)
    return {
        "title": result.get("title", "").strip() or f"Node {node_id}",
        "description": result.get("description", "").strip() or "Semantic group of innovation points.",
    }


def build_leaf_node(node_id, items, title, description, refinement_method):
    members = [
        {
            "point_number": idx,
            "embedding_idx": item["embedding_idx"],
            "paper_id": item["paper_id"],
            "source_filename": item["source_filename"],
            "title": item["title"],
            "year": item["year"],
            "point_index": item["point_index"],
            "point_text": item["point_text"],
        }
        for idx, item in enumerate(items, 1)
    ]
    return {
        "node_id": node_id,
        "title": title,
        "description": description,
        "num_points": len(members),
        "members": members,
        "children": [],
        "is_leaf": True,
        "refinement_method": refinement_method,
    }


def refine_node(node_id, items, client, config, stats):
    print(f"[INFO] Refining node {node_id} with {len(items)} points")
    summary = write_node_summary(node_id, items, client, config)
    stats["node_summaries"] += 1

    if len(items) <= MAX_LEAF_SIZE:
        stats["leaf_nodes"] += 1
        return build_leaf_node(
            node_id,
            items,
            summary["title"],
            summary["description"],
            "leaf_small_cluster",
        )

    system_prompt, user_prompt = get_split_prompts(node_id, items)
    last_llm_result = None
    for validation_attempt in range(VALIDATION_RETRY_LIMIT):
        llm_result = call_llm_json(client, config, system_prompt, user_prompt)
        last_llm_result = llm_result
        try:
            children = validate_split_result(node_id, items, llm_result)
            stats["llm_clean_splits"] += 1
            break
        except ValueError as e:
            print(
                f"[WARN] Node {node_id}: validation failed after LLM split "
                f"attempt {validation_attempt + 1}/{VALIDATION_RETRY_LIMIT}: {e}"
            )
    else:
        print(
            f"[WARN] Node {node_id}: validation still failing after "
            f"{VALIDATION_RETRY_LIMIT} attempts, applying automatic repair"
        )
        children = repair_split_result(node_id, items, last_llm_result or {"groups": []})
        stats["llm_repaired_splits"] += 1

    node = {
        "node_id": node_id,
        "title": summary["title"],
        "description": summary["description"],
        "num_points": len(items),
        "members": [],
        "children": [],
        "is_leaf": False,
        "refinement_method": "llm_recursive_split",
    }

    for child_idx, child in enumerate(children, 1):
        child_items = [
            {
                "embedding_idx": m["embedding_idx"],
                "paper_id": m["paper_id"],
                "source_filename": m["source_filename"],
                "title": m["title"],
                "year": m["year"],
                "point_index": m["point_index"],
                "point_text": m["point_text"],
            }
            for m in child["members"]
        ]
        child_node_id = f"{node_id}.{child_idx}"
        child_node = refine_node(child_node_id, child_items, client, config, stats)

        # Preserve the immediately-generated child title/description as a hint if recursion overwrote root semantics.
        if child.get("title") and not child_node.get("title"):
            child_node["title"] = child["title"]
        if child.get("description") and not child_node.get("description"):
            child_node["description"] = child["description"]

        node["children"].append(child_node)

    return node


def collect_leaf_nodes(node, source_cluster_id, leaves):
    if node.get("is_leaf"):
        leaves.append({
            "theme_id": node["node_id"],
            "source_cluster_id": source_cluster_id,
            "title": node.get("title"),
            "description": node.get("description"),
            "members": node.get("members", []),
        })
        return

    for child in node.get("children", []):
        collect_leaf_nodes(child, source_cluster_id, leaves)


def save_per_cluster_result(refined_cluster):
    os.makedirs(REFINED_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(REFINED_OUTPUT_DIR, f"refined_cluster_{refined_cluster['cluster_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(refined_cluster, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {path}")
    return path


def save_combined_outputs(refined_clusters):
    os.makedirs(REFINED_OUTPUT_DIR, exist_ok=True)

    combined_library = {
        "num_clusters": len(refined_clusters),
        "clusters": refined_clusters,
    }

    library_path = os.path.join(REFINED_OUTPUT_DIR, "refined_memory_library.json")
    with open(library_path, "w", encoding="utf-8") as f:
        json.dump(combined_library, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {library_path}")

    flat_index = []
    final_themes = []
    for cluster in refined_clusters:
        leaves = []
        collect_leaf_nodes(cluster["tree"], cluster["cluster_id"], leaves)
        final_themes.extend(leaves)
        for leaf in leaves:
            for member in leaf["members"]:
                flat_index.append({
                    "source_cluster_id": cluster["cluster_id"],
                    "theme_id": leaf["theme_id"],
                    "theme_title": leaf["title"],
                    "theme_description": leaf["description"],
                    **member,
                })

    index_path = os.path.join(REFINED_OUTPUT_DIR, "refined_memory_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(flat_index, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {index_path}")

    themes_path = os.path.join(REFINED_OUTPUT_DIR, "refined_memory_themes.json")
    with open(themes_path, "w", encoding="utf-8") as f:
        json.dump({"num_themes": len(final_themes), "themes": final_themes}, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {themes_path}")

    return library_path, index_path, themes_path, len(final_themes)


def process_single_cluster(cluster_id, items, client, config):
    print(f"\n{'=' * 80}")
    print(f"[INFO] Recursively refining coarse cluster {cluster_id} with {len(items)} points")
    print(f"{'=' * 80}")

    stats = {
        "node_summaries": 0,
        "llm_clean_splits": 0,
        "llm_repaired_splits": 0,
        "leaf_nodes": 0,
    }
    tree = refine_node(str(cluster_id), items, client, config, stats)

    refined = {
        "cluster_id": cluster_id,
        "num_points": len(items),
        "tree": tree,
        "stats": stats,
    }
    return refined


def main():
    parser = argparse.ArgumentParser(description="Recursively refine coarse KMeans clusters into a semantic tree")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--clusters", type=str, default="", help="Comma-separated cluster IDs to process, e.g. 0,1,2")
    parser.add_argument("--max-clusters", type=int, default=0, help="Only process the first N clusters (0 = all)")
    parser.add_argument("--workers", type=int, default=128, help="Number of parallel cluster workers (default: 128)")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"refine_cluster_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout
    print(f"[INFO] Log file: {log_path}")

    config = load_config(args.config)
    client = build_client(config)

    assignments = load_cluster_assignments()
    grouped = group_assignments_by_cluster(assignments)

    all_cluster_ids = sorted(grouped.keys())
    selected_cluster_ids = all_cluster_ids
    if args.clusters.strip():
        selected_cluster_ids = [int(x.strip()) for x in args.clusters.split(",") if x.strip()]
    elif args.max_clusters > 0:
        selected_cluster_ids = all_cluster_ids[:args.max_clusters]

    print(f"[INFO] Selected coarse clusters to refine: {len(selected_cluster_ids)}")
    print(f"[INFO] Parallel workers: {args.workers}")
    print(f"[INFO] Recursive split threshold (max leaf size): {MAX_LEAF_SIZE}")

    refined_clusters = []
    failures = []
    refined_lock = threading.Lock()
    failures_lock = threading.Lock()

    global_stats = {
        "node_summaries": 0,
        "llm_clean_splits": 0,
        "llm_repaired_splits": 0,
        "leaf_nodes": 0,
    }
    stats_lock = threading.Lock()

    def worker(cluster_id):
        items = grouped.get(cluster_id, [])
        if not items:
            print(f"[WARN] Cluster {cluster_id} is empty, skipping")
            return
        try:
            refined = process_single_cluster(cluster_id, items, client, config)
            save_per_cluster_result(refined)
            with refined_lock:
                refined_clusters.append(refined)
            with stats_lock:
                for key, value in refined["stats"].items():
                    global_stats[key] += value
        except Exception as e:
            print(f"[ERROR] Failed to refine cluster {cluster_id}: {e}")
            traceback.print_exc()
            with failures_lock:
                failures.append({"cluster_id": cluster_id, "error": str(e)})

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, cluster_id) for cluster_id in selected_cluster_ids]
        for _ in as_completed(futures):
            pass

    refined_clusters.sort(key=lambda x: x["cluster_id"])

    if refined_clusters:
        library_path, index_path, themes_path, num_themes = save_combined_outputs(refined_clusters)
    else:
        library_path, index_path, themes_path, num_themes = None, None, None, 0

    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"  Coarse clusters requested: {len(selected_cluster_ids)}")
    print(f"  Successfully refined:     {len(refined_clusters)}")
    print(f"  Failed:                   {len(failures)}")
    print(f"  Node summaries written:   {global_stats['node_summaries']}")
    print(f"  LLM clean splits:         {global_stats['llm_clean_splits']}")
    print(f"  Rule-repaired splits:     {global_stats['llm_repaired_splits']}")
    print(f"  Leaf themes:              {global_stats['leaf_nodes']}")
    print(f"  Flattened final themes:   {num_themes}")
    print(f"  Output dir:               {REFINED_OUTPUT_DIR}")
    if library_path:
        print(f"  Tree library:             {library_path}")
        print(f"  Flat index:               {index_path}")
        print(f"  Flat themes:              {themes_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
