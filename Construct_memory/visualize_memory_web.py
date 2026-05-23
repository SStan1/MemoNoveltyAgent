"""
visualize_memory_web.py
-----------------------
Streamlit app for exploring:
1. the recursive theme tree produced by refine_cluster_memory.py
2. the structured leaf and aggregate histories produced by write_cluster_development_history.py

Run:
  streamlit run <project-root>/Construct_memory/visualize_memory_web.py
"""

import json
import os
import re
from collections import defaultdict
from html import escape as html_escape
from pathlib import Path
from urllib.parse import quote, unquote

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFINED_MEMORY_OUTPUT_DIR = str(PROJECT_ROOT / "memory_database" / "refined_memory_output")
FINAL_MEMORY_OUTPUT_DIR = str(PROJECT_ROOT / "memory_database" / "final_memory_output")


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_year(value):
    """Return a comparable integer year when possible; otherwise None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None

    text = safe_text(value)
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    if not match:
        return None
    return int(match.group(0))


def preview_text(value, limit=160):
    text = safe_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


@st.cache_data(show_spinner=False)
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_final_themes(final_library):
    themes = final_library.get("themes", [])
    return themes if isinstance(themes, list) else []


def extract_aggregate_themes(final_library):
    themes = final_library.get("aggregate_themes", [])
    return themes if isinstance(themes, list) else []


def build_history_lookup(final_library):
    lookup = {}
    for theme in extract_final_themes(final_library) + extract_aggregate_themes(final_library):
        theme_id = safe_text(theme.get("theme_id"))
        if theme_id:
            lookup[theme_id] = theme
    return lookup


def convert_tree_node(node, source_cluster_id, history_lookup):
    node_id = safe_text(node.get("node_id")) or f"cluster_{source_cluster_id}"
    history_theme = history_lookup.get(node_id) or {}
    children = [
        convert_tree_node(child, source_cluster_id, history_lookup)
        for child in node.get("children", [])
        if isinstance(child, dict)
    ]
    return {
        "id": node_id,
        "title": safe_text(node.get("title")) or f"Theme {node_id}",
        "description": safe_text(node.get("description")),
        "num_points": int(node.get("num_points") or 0),
        "is_leaf": bool(node.get("is_leaf")),
        "source_cluster_id": source_cluster_id,
        "children": children,
        "has_history": node_id in history_lookup,
        "history_scope": safe_text(history_theme.get("history_scope")),
        "aggregate_level": safe_text(history_theme.get("aggregate_level")),
    }


def build_forest(refined_library, history_lookup):
    forest = []
    for cluster in refined_library.get("clusters", []):
        cluster_id = cluster.get("cluster_id")
        tree = cluster.get("tree")
        if not isinstance(tree, dict):
            continue
        forest.append(convert_tree_node(tree, cluster_id, history_lookup))
    return forest


def count_nodes(node):
    total = 1
    leaves = 1 if node.get("is_leaf") else 0
    for child in node.get("children", []):
        child_total, child_leaves = count_nodes(child)
        total += child_total
        leaves += child_leaves
    return total, leaves


def build_summary(forest, history_lookup):
    total_nodes = 0
    total_leaves = 0
    for root in forest:
        node_count, leaf_count = count_nodes(root)
        total_nodes += node_count
        total_leaves += leaf_count

    years = set()
    timeline_points = 0
    for theme in history_lookup.values():
        structured = theme.get("development_history_structured") or {}
        for point in structured.get("innovation_points", []):
            if not isinstance(point, dict):
                continue
            year = normalize_year(point.get("year"))
            if year is not None:
                years.add(year)
            timeline_points += 1

    year_range = "N/A"
    if years:
        year_range = f"{min(years)} - {max(years)}"

    return {
        "root_clusters": len(forest),
        "total_nodes": total_nodes,
        "leaf_themes": total_leaves,
        "history_themes": len(history_lookup),
        "timeline_points": timeline_points,
        "year_range": year_range,
    }


def build_node_index(forest):
    node_lookup = {}
    parent_lookup = {}

    def walk(node, parent_id=None):
        node_lookup[node["id"]] = node
        if parent_id is not None:
            parent_lookup[node["id"]] = parent_id
        for child in node.get("children", []):
            walk(child, node["id"])

    for root in forest:
        walk(root)

    return node_lookup, parent_lookup


def get_ancestor_chain(node_id, node_lookup, parent_lookup, levels):
    chain = []
    current = node_id
    for _ in range(max(levels, 0)):
        parent_id = parent_lookup.get(current)
        if not parent_id:
            break
        parent_node = node_lookup.get(parent_id)
        if parent_node is None:
            break
        chain.append(parent_node)
        current = parent_id
    chain.reverse()
    return chain


def is_descendant_or_same(ancestor_id, node_id, parent_lookup):
    current = node_id
    while current:
        if current == ancestor_id:
            return True
        current = parent_lookup.get(current)
    return False


def search_matching_node_ids(node_lookup, cluster_filter, query):
    normalized = safe_text(query).lower()
    if not normalized:
        return []

    matches = []
    for node in node_lookup.values():
        if cluster_filter != "all" and str(node.get("source_cluster_id")) != cluster_filter:
            continue
        if normalized in safe_text(node.get("title")).lower():
            matches.append(node)

    def sort_key(node):
        title = safe_text(node.get("title")).lower()
        return (
            0 if title.startswith(normalized) else 1,
            len(title),
            title,
            safe_text(node.get("id")),
        )

    matches.sort(key=sort_key)
    return [node["id"] for node in matches]


def history_type_label(node_or_theme, fallback_leaf=False):
    history_scope = safe_text(node_or_theme.get("history_scope"))
    if history_scope != "aggregate" and fallback_leaf:
        return "Leaf Node History"
    if history_scope == "aggregate":
        return "Aggregate History"
    return "Leaf Node History" if fallback_leaf else "Theme History"


def node_type_label(node):
    if node.get("has_history"):
        return history_type_label(node, fallback_leaf=node.get("is_leaf"))
    if node.get("is_leaf"):
        return "Leaf Node"
    return "Branch Node"


def format_candidate_label(node):
    node_type = node_type_label(node)
    return f"[Cluster {node.get('source_cluster_id')}] {node['title']} ({node_type})"


def rating_label(rating):
    rating = safe_text(rating)
    if not rating:
        return "unrated"
    return rating.replace("_", " ")


def rating_class(rating):
    rating = safe_text(rating)
    allowed = {
        "major_breakthrough",
        "important_advance",
        "strong_extension",
        "incremental_refinement",
        "limited_contribution",
        "mixed_or_uncertain",
    }
    return rating if rating in allowed else "rating-default"


def group_points_by_year(points):
    grouped = defaultdict(list)
    for point in points:
        grouped[normalize_year(point.get("year"))].append(point)

    def sort_key(year_value):
        if year_value is None:
            return (1, 999999)
        return (0, year_value)

    ordered_years = sorted(grouped.keys(), key=sort_key)
    return [(year, grouped[year]) for year in ordered_years]


def collect_display_timeline_points(history_theme):
    structured = history_theme.get("development_history_structured") or {}
    own_points = structured.get("innovation_points", [])
    if not isinstance(own_points, list):
        own_points = []

    display_points = []
    for point in own_points:
        if isinstance(point, dict):
            item = dict(point)
            item["_source_kind"] = "own_direct_leaf"
            item["_source_theme_title"] = safe_text(history_theme.get("title"))
            display_points.append(item)

    child_histories = history_theme.get("child_aggregate_histories_for_display", [])
    if not isinstance(child_histories, list):
        child_histories = []

    for child_history in child_histories:
        if not isinstance(child_history, dict):
            continue
        child_title = safe_text(child_history.get("title")) or "Child aggregate"
        child_points = child_history.get("innovation_points", [])
        if not isinstance(child_points, list):
            continue
        for point in child_points:
            if not isinstance(point, dict):
                continue
            evaluation = {
                "rating": point.get("rating"),
                "assessment": point.get("assessment"),
            }
            display_points.append({
                "item_number": point.get("item_number"),
                "year": point.get("year"),
                "paper_title": point.get("paper_title"),
                "innovation_content": point.get("innovation_content"),
                "evaluation": evaluation,
                "_source_kind": "child_aggregate",
                "_source_theme_title": child_title,
                "_source_theme_id": child_history.get("theme_id"),
            })

    return display_points


def get_query_param(name):
    try:
        value = st.query_params.get(name)
    except Exception:
        params = st.experimental_get_query_params()
        value = params.get(name)
    if isinstance(value, list):
        value = value[0] if value else ""
    return safe_text(value)


def clear_navigation_query_params():
    try:
        st.query_params.clear()
    except Exception:
        try:
            st.experimental_set_query_params()
        except Exception:
            pass


def apply_query_navigation(node_lookup):
    history_node_id = unquote(get_query_param("history_node_id"))
    selected_match_id = unquote(get_query_param("selected_match_id"))
    if not history_node_id or history_node_id not in node_lookup:
        return
    if selected_match_id not in node_lookup:
        selected_match_id = history_node_id
    st.session_state.selected_match_id = selected_match_id
    st.session_state.history_node_id = history_node_id
    st.session_state.page_mode = "history"


def history_href(node_id):
    selected_match_id = st.session_state.get("selected_match_id") or node_id
    return (
        f"?history_node_id={quote(safe_text(node_id))}"
        f"&selected_match_id={quote(safe_text(selected_match_id))}"
    )


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(199, 141, 61, 0.15), transparent 28%),
                radial-gradient(circle at top right, rgba(16, 120, 112, 0.10), transparent 25%),
                linear-gradient(180deg, #f7f3ec 0%, #ece4d7 100%);
        }
        .block-container {
            max-width: 1400px;
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(255,248,238,0.95));
            border: 1px solid rgba(115, 95, 69, 0.12);
            border-radius: 28px;
            padding: 1.65rem 1.8rem 1.25rem;
            box-shadow: 0 18px 40px rgba(95, 75, 48, 0.10);
            margin-bottom: 1rem;
        }
        .eyebrow {
            display: inline-block;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(115, 95, 69, 0.10);
            color: #8b5b1a;
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }
        .hero-copy {
            color: #5f6673;
            line-height: 1.8;
            margin-top: 0.5rem;
        }
        .panel-card {
            background: rgba(255, 252, 245, 0.86);
            border: 1px solid rgba(115, 95, 69, 0.10);
            border-radius: 24px;
            padding: 1rem 1rem 1.2rem;
            box-shadow: 0 14px 30px rgba(95, 75, 48, 0.08);
            margin-bottom: 1rem;
        }
        .detail-hero {
            background: linear-gradient(135deg, rgba(255,255,255,0.80), rgba(255,249,239,0.95));
            border: 1px solid rgba(115, 95, 69, 0.10);
            border-radius: 24px;
            padding: 1.2rem 1.25rem 1rem;
            margin-bottom: 1rem;
        }
        .detail-desc {
            color: #5f6673;
            line-height: 1.8;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.8rem;
        }
        .pill {
            display: inline-block;
            padding: 0.38rem 0.65rem;
            border-radius: 999px;
            font-size: 0.76rem;
            border: 1px solid transparent;
        }
        .pill.cluster {
            background: rgba(199, 141, 61, 0.14);
            color: #8a5d18;
            border-color: rgba(199, 141, 61, 0.18);
        }
        .pill.leaf {
            background: rgba(16, 120, 112, 0.12);
            color: #0d6d65;
            border-color: rgba(16, 120, 112, 0.18);
        }
        .pill.branch {
            background: rgba(63, 93, 115, 0.12);
            color: #3b586d;
            border-color: rgba(63, 93, 115, 0.18);
        }
        .theme-overview, .overall-summary {
            background: rgba(255,255,255,0.68);
            border: 1px solid rgba(115, 95, 69, 0.10);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin-bottom: 1rem;
        }
        .section-label {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #8b7556;
            margin-bottom: 0.4rem;
        }
        .tree-section-label {
            display: inline-block;
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            color: #6f5131;
            background: rgba(255,255,255,0.62);
            border: 1px solid rgba(115, 95, 69, 0.12);
            border-radius: 999px;
            padding: 0.34rem 0.72rem;
            margin: 1rem 0 0.75rem;
        }
        .level-guide {
            background: rgba(255,255,255,0.70);
            border: 1px solid rgba(115, 95, 69, 0.12);
            border-radius: 18px;
            padding: 0.85rem 0.95rem;
            color: #5f6673;
            line-height: 1.7;
            margin: 0.65rem 0 1rem;
        }
        .level-guide strong {
            color: #4f3822;
        }
        .breadcrumb-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.45rem;
            margin: 0.35rem 0 1rem;
        }
        .breadcrumb-chip {
            display: inline-block;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(115, 95, 69, 0.12);
            color: #5f6673;
            font-size: 0.85rem;
            line-height: 1.4;
        }
        .breadcrumb-chip.focused {
            background: rgba(16, 120, 112, 0.10);
            border-color: rgba(16, 120, 112, 0.20);
            color: #0d6d65;
            font-weight: 700;
        }
        .breadcrumb-arrow {
            color: #8b7556;
            font-size: 0.9rem;
        }
        .tree-node-wrap {
            position: relative;
            padding-left: 1.1rem;
            margin-bottom: 0.7rem;
        }
        .tree-node-wrap.child::before {
            content: "";
            position: absolute;
            left: 0.18rem;
            top: -0.8rem;
            bottom: -0.9rem;
            width: 2px;
            border-radius: 999px;
            background: linear-gradient(180deg, rgba(154, 134, 104, 0.08), rgba(16, 120, 112, 0.30), rgba(154, 134, 104, 0.08));
        }
        .tree-node-wrap.child::after {
            content: "";
            position: absolute;
            left: 0.18rem;
            top: 2.1rem;
            width: 1rem;
            height: 2px;
            border-radius: 999px;
            background: rgba(16, 120, 112, 0.34);
        }
        .tree-node-shell {
            background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(255,249,239,0.80));
            border: 1px solid rgba(115, 95, 69, 0.12);
            border-left: 5px solid rgba(115, 95, 69, 0.20);
            border-radius: 18px;
            padding: 0.95rem 1rem 1rem;
            margin: 0.15rem 0 0.45rem;
            box-shadow: 0 10px 24px rgba(95, 75, 48, 0.07);
        }
        .tree-node-shell.leaf-history {
            border-left-color: rgba(16, 120, 112, 0.58);
        }
        .tree-node-shell.aggregate-history {
            border-left-color: rgba(199, 141, 61, 0.70);
        }
        .tree-node-shell.parent-aggregate {
            border-left-color: rgba(199, 141, 61, 0.70);
        }
        .tree-node-shell.grandparent-aggregate {
            border-left-color: rgba(63, 93, 115, 0.72);
        }
        .tree-node-shell.selected {
            border-color: rgba(16, 120, 112, 0.35);
            box-shadow: 0 12px 28px rgba(16, 120, 112, 0.12);
        }
        .tree-node-kicker {
            display: inline-block;
            color: #7f643f;
            background: rgba(199, 141, 61, 0.12);
            border: 1px solid rgba(199, 141, 61, 0.16);
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.70rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .tree-node-title {
            font-size: 1.06rem;
            font-weight: 800;
            line-height: 1.45;
            color: #28323d;
            margin-bottom: 0.25rem;
        }
        .tree-node-meta {
            color: #8b7556;
            font-size: 0.76rem;
            line-height: 1.5;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.35rem;
        }
        .tree-node-preview {
            color: #5f6673;
            font-size: 0.92rem;
            line-height: 1.65;
        }
        .history-link {
            display: inline-block;
            margin-top: 0.75rem;
            padding: 0.45rem 0.72rem;
            border-radius: 999px;
            text-decoration: none !important;
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #0d6d65 !important;
            background: rgba(16, 120, 112, 0.10);
            border: 1px solid rgba(16, 120, 112, 0.18);
        }
        .history-link:hover {
            background: rgba(16, 120, 112, 0.16);
            border-color: rgba(16, 120, 112, 0.32);
        }
        .branch-group {
            background: rgba(255,255,255,0.58);
            border: 1px solid rgba(115, 95, 69, 0.08);
            border-radius: 20px;
            padding: 0.9rem 0.9rem 0.35rem;
            margin-bottom: 1rem;
        }
        .branch-title {
            color: #5f462b;
            font-size: 0.98rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            margin: 0.9rem 0 0.65rem;
            padding: 0.45rem 0.65rem;
            border-left: 4px solid rgba(199, 141, 61, 0.65);
            background: rgba(255,255,255,0.58);
            border-radius: 12px;
        }
        .tree-connector {
            color: #6f5131;
            margin: 0.2rem 0 0.35rem 0.35rem;
            font-size: 0.88rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }
        .tree-truncated {
            color: #8b7556;
            font-size: 0.86rem;
            margin: 0.2rem 0 0.7rem;
        }
        .candidate-note {
            color: #8b7556;
            font-size: 0.88rem;
            margin-top: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .empty-state {
            background: rgba(255,255,255,0.72);
            border: 1px dashed rgba(115, 95, 69, 0.22);
            border-radius: 24px;
            padding: 2rem 1.8rem;
            color: #5f6673;
            line-height: 1.8;
        }
        .year-shell {
            background: rgba(255,255,255,0.76);
            border: 1px solid rgba(115, 95, 69, 0.10);
            border-radius: 22px;
            padding: 1rem 1rem 0.5rem;
            box-shadow: 0 10px 24px rgba(95, 75, 48, 0.06);
            margin-bottom: 1rem;
        }
        .year-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .year-count {
            color: #8b7556;
            font-size: 0.82rem;
            margin-bottom: 0.9rem;
        }
        .point-shell {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(115, 95, 69, 0.10);
            border-radius: 18px;
            padding: 0.85rem 0.95rem 0.2rem;
            margin-bottom: 0.75rem;
        }
        .point-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.75rem;
        }
        .point-title {
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.5;
        }
        .rating-badge {
            display: inline-block;
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        .major_breakthrough {
            background: rgba(158, 72, 33, 0.16);
            color: #9a421d;
        }
        .important_advance {
            background: rgba(199, 141, 61, 0.16);
            color: #8a5d18;
        }
        .strong_extension {
            background: rgba(63, 93, 115, 0.16);
            color: #3b586d;
        }
        .incremental_refinement {
            background: rgba(16, 120, 112, 0.14);
            color: #0d6d65;
        }
        .limited_contribution {
            background: rgba(107, 114, 128, 0.14);
            color: #606977;
        }
        .mixed_or_uncertain, .rating-default {
            background: rgba(115, 95, 69, 0.12);
            color: #6b5a45;
        }
        .point-meta {
            color: #5f6673;
            font-size: 0.9rem;
            line-height: 1.75;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(115, 95, 69, 0.10);
            border-radius: 16px;
            background: rgba(255,255,255,0.78);
        }
        .stButton button {
            border-radius: 12px;
            border: 1px solid rgba(115, 95, 69, 0.14);
            background: rgba(255,255,255,0.80);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    defaults = {
        "search_query": "",
        "cluster_filter": "all",
        "ancestor_levels": 1,
        "descendant_levels": 3,
        "selected_match_id": None,
        "history_node_id": None,
        "page_mode": "empty",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar(summary, forest, node_lookup):
    st.sidebar.markdown("## Explore Controls")
    st.sidebar.caption("Search only by theme title, then inspect one focused local tree.")

    st.session_state.search_query = st.sidebar.text_input(
        "Search theme title",
        value=st.session_state.search_query,
        placeholder="e.g. machine learning",
    )

    cluster_options = ["all"] + [str(root.get("source_cluster_id")) for root in forest]
    current_cluster = st.session_state.cluster_filter
    if current_cluster not in cluster_options:
        current_cluster = "all"
    st.session_state.cluster_filter = st.sidebar.selectbox(
        "Coarse cluster filter",
        options=cluster_options,
        index=cluster_options.index(current_cluster),
        format_func=lambda x: "All coarse clusters" if x == "all" else f"Cluster {x}",
    )

    st.session_state.ancestor_levels = st.sidebar.slider(
        "Parent levels to show",
        min_value=0,
        max_value=4,
        value=int(st.session_state.ancestor_levels),
        step=1,
    )
    st.session_state.descendant_levels = st.sidebar.slider(
        "Child levels to show",
        min_value=0,
        max_value=6,
        value=int(st.session_state.descendant_levels),
        step=1,
    )

    match_ids = search_matching_node_ids(
        node_lookup,
        st.session_state.cluster_filter,
        st.session_state.search_query,
    )

    if match_ids and st.session_state.selected_match_id not in match_ids:
        st.session_state.selected_match_id = match_ids[0]
        st.session_state.page_mode = "tree"
        st.session_state.history_node_id = None

    if match_ids:
        candidate_map = {node_id: format_candidate_label(node_lookup[node_id]) for node_id in match_ids}
        selected_candidate = st.sidebar.selectbox(
            "Matching themes",
            options=match_ids,
            index=match_ids.index(st.session_state.selected_match_id),
            format_func=lambda node_id: candidate_map.get(node_id, node_id),
        )
        if selected_candidate != st.session_state.selected_match_id:
            clear_navigation_query_params()
            st.session_state.selected_match_id = selected_candidate
            st.session_state.page_mode = "tree"
            st.session_state.history_node_id = None
            st.rerun()
        st.sidebar.caption(f"{len(match_ids)} title match(es). The first match is shown by default.")
    elif safe_text(st.session_state.search_query):
        st.sidebar.warning("No theme title matches the current query.")

    if st.session_state.page_mode == "history" and st.session_state.history_node_id:
        if st.sidebar.button("Back To Focused Tree", use_container_width=True):
            clear_navigation_query_params()
            st.session_state.page_mode = "tree"
            st.session_state.history_node_id = None
            st.rerun()

    if st.sidebar.button("Reset View", use_container_width=True):
        clear_navigation_query_params()
        st.session_state.search_query = ""
        st.session_state.cluster_filter = "all"
        st.session_state.ancestor_levels = 1
        st.session_state.descendant_levels = 3
        st.session_state.selected_match_id = None
        st.session_state.history_node_id = None
        st.session_state.page_mode = "empty"
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Snapshot")
    st.sidebar.write(f"Root clusters: {summary['root_clusters']}")
    st.sidebar.write(f"Tree nodes: {summary['total_nodes']}")
    st.sidebar.write(f"Leaf themes: {summary['leaf_themes']}")
    st.sidebar.write(f"Timeline points: {summary['timeline_points']}")
    st.sidebar.write(f"Year range: {summary['year_range']}")

    return match_ids


def sync_view_state(match_ids, node_lookup, parent_lookup):
    if not match_ids:
        if (
            st.session_state.page_mode == "history"
            and st.session_state.history_node_id in node_lookup
        ):
            if st.session_state.selected_match_id not in node_lookup:
                st.session_state.selected_match_id = st.session_state.history_node_id
            return
        st.session_state.selected_match_id = None
        st.session_state.history_node_id = None
        st.session_state.page_mode = "empty"
        return

    if st.session_state.selected_match_id not in match_ids:
        st.session_state.selected_match_id = match_ids[0]
        st.session_state.page_mode = "tree"
        st.session_state.history_node_id = None

    history_node_id = st.session_state.history_node_id
    if st.session_state.page_mode == "history":
        if history_node_id not in node_lookup:
            st.session_state.page_mode = "tree"
            st.session_state.history_node_id = None
        elif not is_descendant_or_same(st.session_state.selected_match_id, history_node_id, parent_lookup):
            st.session_state.page_mode = "tree"
            st.session_state.history_node_id = None


def render_empty_state():
    st.markdown(
        """
        <div class="panel-card">
          <div class="empty-state">
            <h2 style="margin-top:0;">Start From A Theme Search</h2>
            <p>
              This page does not expand the full tree by default.
              Use the left sidebar to search by theme title and choose one matching theme.
            </p>
            <p>
              The main area will then show only a focused local tree:
              a small amount of parent context and a limited number of child levels.
              Click any theme with available history to jump into its development history page.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_theme_card(node, history_lookup, scope_key, margin_left=0, selected=False):
    type_label = node_type_label(node)
    if node.get("is_leaf") and node.get("has_history"):
        card_class = "leaf-history"
    elif safe_text(node.get("history_scope")) == "aggregate":
        card_class = "aggregate-history"
    else:
        card_class = "branch-node"
    wrap_class = "tree-node-wrap" if margin_left <= 0 else "tree-node-wrap child"
    st.markdown(
        f"""
        <div class="{wrap_class}" style="margin-left:{margin_left}px;">
          <div class="tree-node-shell {card_class}{' selected' if selected else ''}">
            <div class="tree-node-kicker">{type_label}</div>
            <div class="tree-node-title">{node['title']}</div>
            <div class="tree-node-meta">
              Cluster {safe_text(node.get("source_cluster_id"))}
              &nbsp;|&nbsp;
              {'Leaf Theme' if node.get('is_leaf') else 'Branch Node'}
              &nbsp;|&nbsp;
              {safe_text(node.get("num_points"))} points
            </div>
            <div class="tree-node-preview">{preview_text(node.get("description")) or "No theme description available."}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_breadcrumbs(ancestors, selected_node):
    crumb_html = []
    for ancestor in ancestors:
        crumb_html.append(f"<span class='breadcrumb-chip'>{ancestor['title']}</span>")
        crumb_html.append("<span class='breadcrumb-arrow'>→</span>")
    crumb_html.append(f"<span class='breadcrumb-chip focused'>{selected_node['title']}</span>")
    st.markdown(
        "<div class='breadcrumb-row'>" + "".join(crumb_html) + "</div>",
        unsafe_allow_html=True,
    )


def render_branch_subtree(node, history_lookup, max_depth, depth=1, scope_prefix="branch"):
    if depth > max_depth:
        return

    for index, child in enumerate(node.get("children", [])):
        margin_left = depth * 34
        st.markdown(
            f"<div class='tree-connector' style='margin-left:{margin_left}px;'>└─</div>",
            unsafe_allow_html=True,
        )
        render_theme_card(
            child,
            history_lookup,
            scope_key=f"{scope_prefix}_{depth}_{index}",
            margin_left=margin_left + 18,
        )
        if child.get("children"):
            if depth < max_depth:
                render_branch_subtree(
                    child,
                    history_lookup,
                    max_depth=max_depth,
                    depth=depth + 1,
                    scope_prefix=f"{scope_prefix}_{index}",
                )
            else:
                st.markdown(
                    f"<div class='tree-truncated' style='margin-left:{margin_left + 18}px;'>+ {len(child.get('children', []))} deeper subthemes hidden</div>",
                    unsafe_allow_html=True,
                )

def render_breadcrumbs(ancestors, selected_node):
    crumb_html = []
    for ancestor in ancestors:
        crumb_html.append(f"<span class='breadcrumb-chip'>{ancestor['title']}</span>")
        crumb_html.append("<span class='breadcrumb-arrow'>&rarr;</span>")
    crumb_html.append(f"<span class='breadcrumb-chip focused'>{selected_node['title']}</span>")
    st.markdown(
        "<div class='breadcrumb-row'>" + "".join(crumb_html) + "</div>",
        unsafe_allow_html=True,
    )


def render_branch_subtree(node, history_lookup, max_depth, depth=1, scope_prefix="branch"):
    if depth > max_depth:
        return

    for index, child in enumerate(node.get("children", [])):
        margin_left = depth * 34
        st.markdown(
            f"<div class='tree-connector' style='margin-left:{margin_left}px;'>Level {depth} Branch {index + 1}</div>",
            unsafe_allow_html=True,
        )
        render_theme_card(
            child,
            history_lookup,
            scope_key=f"{scope_prefix}_{depth}_{index}",
            margin_left=margin_left + 18,
        )
        if child.get("children"):
            if depth < max_depth:
                render_branch_subtree(
                    child,
                    history_lookup,
                    max_depth=max_depth,
                    depth=depth + 1,
                    scope_prefix=f"{scope_prefix}_{index}",
                )
            else:
                st.markdown(
                    f"<div class='tree-truncated' style='margin-left:{margin_left + 18}px;'>+ {len(child.get('children', []))} deeper subthemes hidden</div>",
                    unsafe_allow_html=True,
                )

def render_theme_card(node, history_lookup, scope_key, margin_left=0, selected=False):
    type_label = node_type_label(node)
    aggregate_level = safe_text(node.get("aggregate_level"))
    if node.get("is_leaf") and node.get("has_history"):
        card_class = "leaf-history"
    elif aggregate_level == "leaf_parent":
        card_class = "parent-aggregate"
    elif aggregate_level == "leaf_grandparent":
        card_class = "grandparent-aggregate"
    else:
        card_class = "branch-node"

    wrap_class = "tree-node-wrap" if margin_left <= 0 else "tree-node-wrap child"
    history_action = ""
    if node.get("has_history"):
        action_label = html_escape(f"Open {history_type_label(node, fallback_leaf=node.get('is_leaf'))}")
        history_action = f"<a class='history-link' href='{history_href(node['id'])}'>{action_label}</a>"

    st.markdown(
        f"""
        <div class="{wrap_class}" style="margin-left:{margin_left}px;">
          <div class="tree-node-shell {card_class}{' selected' if selected else ''}">
            <div class="tree-node-kicker">{html_escape(type_label)}</div>
            <div class="tree-node-title">{html_escape(node['title'])}</div>
            <div class="tree-node-meta">
              Cluster {html_escape(safe_text(node.get("source_cluster_id")))}
              &nbsp;|&nbsp;
              {html_escape("Leaf Node" if node.get("is_leaf") else "Branch Node")}
              &nbsp;|&nbsp;
              {html_escape(safe_text(node.get("num_points")))} points
            </div>
            <div class="tree-node-preview">{html_escape(preview_text(node.get("description")) or "No theme description available.")}</div>
            {history_action}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_breadcrumbs(ancestors, selected_node):
    crumb_html = []
    for ancestor in ancestors:
        crumb_html.append(f"<span class='breadcrumb-chip'>{html_escape(ancestor['title'])}</span>")
        crumb_html.append("<span class='breadcrumb-arrow'>&rarr;</span>")
    crumb_html.append(f"<span class='breadcrumb-chip focused'>{html_escape(selected_node['title'])}</span>")
    st.markdown(
        "<div class='breadcrumb-row'>" + "".join(crumb_html) + "</div>",
        unsafe_allow_html=True,
    )


def render_branch_subtree(node, history_lookup, max_depth, depth=1, scope_prefix="branch", branch_path=""):
    if depth > max_depth:
        return

    for index, child in enumerate(node.get("children", []), start=1):
        current_path = f"{branch_path}.{index}" if branch_path else str(index)
        margin_left = depth * 34
        st.markdown(
            f"<div class='tree-connector' style='margin-left:{margin_left}px;'>Branch {html_escape(current_path)}</div>",
            unsafe_allow_html=True,
        )
        render_theme_card(
            child,
            history_lookup,
            scope_key=f"{scope_prefix}_{depth}_{index}",
            margin_left=margin_left + 18,
        )
        if child.get("children"):
            if depth < max_depth:
                render_branch_subtree(
                    child,
                    history_lookup,
                    max_depth=max_depth,
                    depth=depth + 1,
                    scope_prefix=f"{scope_prefix}_{index}",
                    branch_path=current_path,
                )
            else:
                st.markdown(
                    f"<div class='tree-truncated' style='margin-left:{margin_left + 18}px;'>+ {len(child.get('children', []))} deeper subthemes hidden</div>",
                    unsafe_allow_html=True,
                )


def render_focused_tree_panel(selected_node, node_lookup, parent_lookup, history_lookup, ancestor_levels, descendant_levels, match_count):
    st.markdown("### Focused Theme Tree")
    st.markdown(
        f"<div class='candidate-note'>Currently showing 1 focused local tree out of {match_count} title match(es).</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="level-guide">
          <strong>Level guide:</strong>
          Leaf Node = the smallest LLM-split theme with detailed point history.
          Aggregate Node = a non-leaf theme that summarizes its direct leaf children; if it also has child aggregate branches,
          its overall summary additionally integrates those child aggregate histories.
          Branch labels identify sibling paths in the focused local tree; depth is visible from indentation and connectors.
        </div>
        """,
        unsafe_allow_html=True,
    )

    ancestors = get_ancestor_chain(selected_node["id"], node_lookup, parent_lookup, ancestor_levels)
    if ancestors:
        st.markdown("<div class='tree-section-label'>Parent Context</div>", unsafe_allow_html=True)
        render_breadcrumbs(ancestors, selected_node)

    st.markdown("<div class='tree-section-label'>Focused Theme</div>", unsafe_allow_html=True)
    render_theme_card(selected_node, history_lookup, scope_key="focused", margin_left=0, selected=True)

    if descendant_levels > 0 and selected_node.get("children"):
        st.markdown("<div class='tree-section-label'>Descendant Branches</div>", unsafe_allow_html=True)
        for branch_index, child in enumerate(selected_node.get("children", []), start=1):
            st.markdown(
                f"<div class='branch-title'>Branch {branch_index}</div>",
                unsafe_allow_html=True,
            )
            render_theme_card(
                child,
                history_lookup,
                scope_key=f"topbranch_{branch_index}",
                margin_left=0,
            )
            if child.get("children"):
                if descendant_levels > 1:
                    render_branch_subtree(
                        child,
                        history_lookup,
                        max_depth=descendant_levels,
                        depth=2,
                        scope_prefix=f"topbranch_{branch_index}",
                        branch_path=str(branch_index),
                    )
                else:
                    st.markdown(
                        f"<div class='tree-truncated'>+ {len(child.get('children', []))} child theme(s) hidden at current depth setting</div>",
                        unsafe_allow_html=True,
                    )
    elif selected_node.get("children"):
        st.info("Child levels are set to 0, so descendant themes are hidden.")
    else:
        st.info("This focused theme has no child themes to preview.")


def render_theme_history(node, history_theme):
    structured = history_theme.get("development_history_structured") or {}
    theme_overview = safe_text(structured.get("theme_overview"))
    overall_summary = safe_text(structured.get("overall_evolution_summary"))
    points = collect_display_timeline_points(history_theme)
    grouped = group_points_by_year(points)
    history_scope = safe_text(history_theme.get("history_scope"))
    is_leaf_history = history_scope != "aggregate" and node.get("is_leaf")
    history_label = history_type_label(history_theme, fallback_leaf=is_leaf_history)

    st.markdown(
        f"""
        <div class="detail-hero">
          <h2 style="margin:0;">{html_escape(node['title'])}</h2>
          <p class="detail-desc">{html_escape(safe_text(node.get("description")) or "No theme description available.")}</p>
          <div class="pill-row">
            <span class="pill cluster">Cluster {html_escape(safe_text(node.get("source_cluster_id")))}</span>
            <span class="pill leaf">{html_escape(history_label)}</span>
            <span class="pill branch">{html_escape(safe_text(node.get("num_points")))} points</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Back To Focused Tree", key="history_back_top", use_container_width=False):
        clear_navigation_query_params()
        st.session_state.page_mode = "tree"
        st.session_state.history_node_id = None
        st.rerun()

    if not grouped:
        st.warning("No structured development-history timeline is available for this theme.")
    else:
        st.markdown("#### Development Timeline")
        for year, year_points in grouped:
            year_label = safe_text(year) or "Unknown"
            st.markdown(f"### {year_label}")
            st.caption(f"{len(year_points)} item(s)")
            for point in year_points:
                evaluation = point.get("evaluation") or {}
                rating = safe_text(evaluation.get("rating"))
                assessment = safe_text(evaluation.get("assessment"))
                paper_title = safe_text(point.get("paper_title")) or "Unknown paper"
                item_number = point.get("item_number")
                source_kind = safe_text(point.get("_source_kind"))
                source_theme_title = safe_text(point.get("_source_theme_title"))
                source_label = "Child Aggregate" if source_kind == "child_aggregate" else "Own Direct Leaf"
                expander_title = paper_title
                if item_number is not None:
                    expander_title += f" | Item {item_number}"

                st.markdown(
                    f"""
                    <div class="point-head">
                      <div class="point-title">{html_escape(paper_title)}</div>
                      <span class="rating-badge {rating_class(rating)}">{html_escape(rating_label(rating))}</span>
                    </div>
                    <div class="point-meta">Source: {html_escape(source_label)}{html_escape(' - ' + source_theme_title if source_theme_title else '')}</div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander(f"Open details for {expander_title}", expanded=False):
                    st.markdown("**Innovation Content**")
                    st.write(safe_text(point.get("innovation_content")) or "No innovation content available.")
                    st.markdown("**Assessment**")
                    st.write(assessment or "No assessment available.")

    if theme_overview:
        st.markdown(
            f"""
            <div class="theme-overview">
              <div class="section-label">Theme Overview</div>
              <div class="point-meta">{html_escape(theme_overview)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if overall_summary:
        st.markdown(
            f"""
            <div class="overall-summary">
              <div class="section-label">Overall Evolution Summary</div>
              <div class="point-meta">{html_escape(overall_summary)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    raw_history = safe_text(history_theme.get("development_history"))
    if raw_history:
        with st.expander("Compatibility Text View", expanded=False):
            st.text(raw_history)


def main():
    st.set_page_config(
        page_title="Construct Memory Theme Atlas",
        page_icon="A",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    initialize_session_state()

    refined_library_path = os.path.join(REFINED_MEMORY_OUTPUT_DIR, "refined_memory_library.json")
    final_library_path = os.path.join(FINAL_MEMORY_OUTPUT_DIR, "final_memory_library.json")

    st.markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">Construct Memory Theme Atlas</div>
          <h1 style="margin:0;">Focused Theme Tree And Development-History Explorer</h1>
          <p class="hero-copy">
            Search by theme title in the left sidebar, inspect one focused local tree in the main view,
            and open leaf or aggregate development history when a theme has one.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing_paths = [path for path in [refined_library_path, final_library_path] if not Path(path).exists()]
    if missing_paths:
        st.error("Missing visualization input files:")
        for path in missing_paths:
            st.code(path)
        st.stop()

    refined_library = load_json(refined_library_path)
    final_library = load_json(final_library_path)
    history_lookup = build_history_lookup(final_library)
    forest = build_forest(refined_library, history_lookup)
    summary = build_summary(forest, history_lookup)
    node_lookup, parent_lookup = build_node_index(forest)

    match_ids = render_sidebar(summary, forest, node_lookup)
    apply_query_navigation(node_lookup)
    sync_view_state(match_ids, node_lookup, parent_lookup)

    if st.session_state.page_mode == "empty" or not st.session_state.selected_match_id:
        render_empty_state()
        return

    if st.session_state.page_mode == "history" and st.session_state.history_node_id:
        history_node = node_lookup.get(st.session_state.history_node_id)
        history_theme = history_lookup.get(st.session_state.history_node_id)
        if history_node is not None and history_theme is not None:
            render_theme_history(history_node, history_theme)
            return
        st.session_state.page_mode = "tree"
        st.session_state.history_node_id = None

    selected_node = node_lookup.get(st.session_state.selected_match_id)
    if selected_node is None:
        render_empty_state()
        return

    render_focused_tree_panel(
        selected_node=selected_node,
        node_lookup=node_lookup,
        parent_lookup=parent_lookup,
        history_lookup=history_lookup,
        ancestor_levels=int(st.session_state.ancestor_levels),
        descendant_levels=int(st.session_state.descendant_levels),
        match_count=len(match_ids),
    )


if __name__ == "__main__":
    main()
