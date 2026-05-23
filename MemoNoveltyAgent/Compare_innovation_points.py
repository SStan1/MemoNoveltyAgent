import os
import re
import importlib.util
import openai
from ragflow_sdk import RAGFlow
import time
import traceback
import fitz
from pathlib import Path

class _SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def format_prompt(template: str, **kwargs) -> str:
    return template.format_map(_SafeDict(**kwargs))


_MEMORY_RETRIEVER_MODULE = None
_MEMORY_RETRIEVAL_RESOURCES = {}


def get_construct_memory_dir(config):
    expert_cfg = config.get('expert_memory', {})
    configured_dir = expert_cfg.get('construct_memory_dir')
    base_dir = Path(__file__).resolve().parent

    if configured_dir:
        configured_path = Path(configured_dir)
        if not configured_path.is_absolute():
            configured_path = (base_dir / configured_path).resolve()
        return str(configured_path)

    return str((base_dir.parent / "Construct_memory").resolve())


def load_memory_retriever_module(config):
    global _MEMORY_RETRIEVER_MODULE
    if _MEMORY_RETRIEVER_MODULE is not None:
        return _MEMORY_RETRIEVER_MODULE

    construct_memory_dir = get_construct_memory_dir(config)
    module_path = os.path.join(construct_memory_dir, "retrieve_query_cluster.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Construct_memory retriever not found: {module_path}")

    spec = importlib.util.spec_from_file_location(
        "construct_memory_retrieve_query_cluster",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load retrieve_query_cluster.py from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MEMORY_RETRIEVER_MODULE = module
    return module


def get_memory_retrieval_resources(config):
    module = load_memory_retriever_module(config)
    expert_cfg = config.get('expert_memory', {})
    gpu_id = int(expert_cfg.get('gpu', 0))
    model_path = expert_cfg.get('embedding_model_path') or getattr(module, 'MODEL_PATH', 'Qwen/Qwen3-Embedding-8B')
    cache_key = (get_construct_memory_dir(config), gpu_id, model_path)

    if cache_key in _MEMORY_RETRIEVAL_RESOURCES:
        return _MEMORY_RETRIEVAL_RESOURCES[cache_key]

    embeddings, index = module.load_embeddings_and_index()
    final_library = module.load_final_memory_library()
    embedding_to_theme, theme_lookup = module.build_theme_lookup(final_library)
    leaf_parent_aggregate_lookup = build_leaf_parent_aggregate_lookup(final_library)
    tokenizer, model, device = module.load_embedding_model(model_path, gpu_id)

    resources = {
        "module": module,
        "embeddings": embeddings,
        "index": index,
        "embedding_to_theme": embedding_to_theme,
        "theme_lookup": theme_lookup,
        "leaf_parent_aggregate_lookup": leaf_parent_aggregate_lookup,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }
    _MEMORY_RETRIEVAL_RESOURCES[cache_key] = resources
    return resources


def build_memory_selection_config(config):
    llm_config = config['llm_config']
    expert_cfg = config.get('expert_memory', {})
    return {
        "llm_config": {
            "model": expert_cfg.get('selection_model') or llm_config['model'],
            "temperature": expert_cfg.get('selection_temperature', llm_config['temperature']),
            "max_retries": expert_cfg.get('selection_max_retries', llm_config['max_retries']),
            "retry_delay": expert_cfg.get('selection_retry_delay', llm_config['retry_delay']),
        }
    }


def extract_aggregate_themes_from_library(final_library):
    if isinstance(final_library, dict):
        themes = final_library.get("aggregate_themes", [])
        return themes if isinstance(themes, list) else []
    return []


def build_leaf_parent_aggregate_lookup(final_library):
    """
    Map each leaf theme to the closest aggregate node that directly owns it.
    If older/partial outputs lack direct_leaf_theme_ids, fall back to the
    deepest aggregate whose descendant_leaf_theme_ids contains that leaf.
    """
    selected_by_leaf = {}

    for aggregate in extract_aggregate_themes_from_library(final_library):
        if not isinstance(aggregate, dict):
            continue
        try:
            depth = int(aggregate.get("tree_depth", 0) or 0)
        except Exception:
            depth = 0

        direct_ids = aggregate.get("direct_leaf_theme_ids", [])
        descendant_ids = aggregate.get("descendant_leaf_theme_ids", [])
        if not isinstance(direct_ids, list):
            direct_ids = []
        if not isinstance(descendant_ids, list):
            descendant_ids = []

        for leaf_id in descendant_ids:
            leaf_key = str(leaf_id).strip()
            if not leaf_key:
                continue
            current = selected_by_leaf.get(leaf_key)
            rank = (0, depth)
            if current is None or rank > current[0]:
                selected_by_leaf[leaf_key] = (rank, aggregate)

        for leaf_id in direct_ids:
            leaf_key = str(leaf_id).strip()
            if not leaf_key:
                continue
            current = selected_by_leaf.get(leaf_key)
            rank = (1, depth)
            if current is None or rank > current[0]:
                selected_by_leaf[leaf_key] = (rank, aggregate)

    return {leaf_id: aggregate for leaf_id, (_, aggregate) in selected_by_leaf.items()}


def retrieve_expert_knowledge_for_innovation_point(client, innovation_point, config):
    expert_cfg = config.get('expert_memory', {})
    if not expert_cfg.get('enabled', True):
        return None

    resources = get_memory_retrieval_resources(config)
    module = resources["module"]

    query_embedding = module.embed_single_text(
        innovation_point,
        resources["tokenizer"],
        resources["model"],
        resources["device"],
        max_length=int(expert_cfg.get('max_length', 512)),
    )
    retrieved_points = module.retrieve_top_k(
        query_embedding,
        resources["embeddings"],
        resources["index"],
        resources["embedding_to_theme"],
        top_k=int(expert_cfg.get('top_k', 10)),
    )
    candidate_themes = module.summarize_candidate_themes(retrieved_points, resources["theme_lookup"])

    if not candidate_themes:
        return {
            "query_text": innovation_point,
            "retrieved_points": retrieved_points,
            "candidate_leaf_themes": [],
            "llm_decision": {"selection_mode": "no_candidate_themes"},
            "selected_themes": [],
            "selected_theme_ids": [],
        }

    if len(candidate_themes) == 1:
        selected_theme_ids, llm_decision = module.build_auto_selection_result(candidate_themes)
    else:
        selected_theme_ids, llm_decision = module.select_themes_with_retry(
            client,
            build_memory_selection_config(config),
            innovation_point,
            candidate_themes,
            retrieved_points,
            max_attempts=int(expert_cfg.get('selection_attempts', 3)),
        )

    selected_themes = [
        resources["theme_lookup"][theme_id]
        for theme_id in selected_theme_ids
        if theme_id in resources["theme_lookup"]
    ]
    parent_lookup = resources.get("leaf_parent_aggregate_lookup", {})
    selected_parent_aggregates_by_theme_id = {
        theme.get("theme_id"): parent_lookup.get(str(theme.get("theme_id", "")).strip())
        for theme in selected_themes
        if parent_lookup.get(str(theme.get("theme_id", "")).strip())
    }

    return {
        "query_text": innovation_point,
        "retrieved_points": retrieved_points,
        "candidate_leaf_themes": candidate_themes,
        "llm_decision": llm_decision,
        "selected_themes": selected_themes,
        "selected_theme_ids": selected_theme_ids,
        "selected_parent_aggregates_by_theme_id": selected_parent_aggregates_by_theme_id,
    }


def safe_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def get_structured_history(theme):
    structured = theme.get("development_history_structured", {})
    return structured if isinstance(structured, dict) else {}


def get_structured_points(theme):
    points = get_structured_history(theme).get("innovation_points", [])
    return points if isinstance(points, list) else []


def make_historical_citation_marker(paper_title):
    title = safe_text(paper_title)
    if not title or title == "Unknown paper":
        return ""
    title = title.replace("$", "").replace("#", "").strip()
    return f"##document_name: {title}$$"


def format_retrieved_hit_lines(candidate):
    hit_lines = []
    for hit in candidate.get("hit_points", []):
        hit_lines.append(
            f"- rank {hit.get('rank')} | score {hit.get('score', 0.0):.4f} | "
            f"{hit.get('paper_title')} ({hit.get('year')})\n"
            f"  Retrieved innovation point: {hit.get('innovation_point')}"
        )
    return hit_lines


def format_theme_innovation_content_lines(theme):
    lines = []
    for point in get_structured_points(theme):
        paper_title = safe_text(point.get("paper_title")) or "Unknown paper"
        year = safe_text(point.get("year")) or "Unknown year"
        innovation_content = safe_text(point.get("innovation_content"))
        citation_marker = make_historical_citation_marker(paper_title)
        if not innovation_content:
            continue
        lines.append(
            f"- [{year}] {paper_title}\n"
            f"  Historical citation marker: {citation_marker or 'Not available'}\n"
            f"  Historical innovation content: {innovation_content}"
        )

    if lines:
        return lines

    fallback_lines = []
    for member in theme.get("members", []):
        fallback_lines.append(
            f"- [{member.get('year')}] {member.get('title')} | "
            f"Theme innovation point: {member.get('point_text')}"
        )
    return fallback_lines


def format_theme_evaluation_lines(theme):
    lines = []
    for point in get_structured_points(theme):
        paper_title = safe_text(point.get("paper_title")) or "Unknown paper"
        year = safe_text(point.get("year")) or "Unknown year"
        citation_marker = make_historical_citation_marker(paper_title)
        evaluation = point.get("evaluation", {})
        if not isinstance(evaluation, dict):
            evaluation = {}
        rating = safe_text(evaluation.get("rating")) or "unrated"
        assessment = safe_text(evaluation.get("assessment"))
        point_content = safe_text(point.get("innovation_content"))
        lines.append(
            f"- [{year}] {paper_title}\n"
            f"  Historical citation marker: {citation_marker or 'Not available'}\n"
            f"  Historical innovation content: {point_content or 'Not available'}\n"
            f"  Historical evaluation rating: {rating}\n"
            f"  Historical assessment: {assessment or 'Not available'}"
        )
    return lines


def format_parent_aggregate_overall_context(parent_aggregate):
    if not isinstance(parent_aggregate, dict) or not parent_aggregate:
        return "No broader aggregate history was available for this selected leaf theme."

    structured = get_structured_history(parent_aggregate)
    overall = safe_text(structured.get("overall_evolution_summary"))
    overview = safe_text(structured.get("theme_overview"))
    return (
        f"Broader aggregate theme ID: {parent_aggregate.get('theme_id')}\n"
        f"Broader aggregate title: {parent_aggregate.get('title')}\n"
        f"Broader aggregate description: {parent_aggregate.get('description')}\n"
        f"Broader aggregate overview: {overview or 'Not available'}\n"
        f"Broader aggregate overall evolution summary:\n{overall or 'Not available'}\n"
        "Use this broader summary only as higher-level field-development context. "
        "For concrete paper-level citation, cite the historical paper markers listed in the selected leaf theme above whenever possible."
    )


def format_expert_knowledge_for_comparison_prompt(expert_result):
    if not expert_result:
        return "Expert knowledge retrieval was disabled or unavailable."

    selected_themes = expert_result.get("selected_themes", [])
    if not selected_themes:
        return "No expert-memory themes were selected for this innovation point."

    candidate_map = {
        theme["theme_id"]: theme
        for theme in expert_result.get("candidate_leaf_themes", [])
    }
    reason_map = {
        item.get("theme_id"): item.get("reason", "")
        for item in expert_result.get("llm_decision", {}).get("selected_themes", [])
        if item.get("theme_id")
    }
    parent_map = expert_result.get("selected_parent_aggregates_by_theme_id", {}) or {}

    blocks = []
    selection_mode = expert_result.get("llm_decision", {}).get("selection_mode", "llm_selected")
    blocks.append(f"Expert-memory selection mode: {selection_mode}")
    blocks.append(
        "Use the historical innovation-point contents below as the PRIMARY expert-memory evidence for similarity and difference analysis. "
        "When using a concrete historical paper from this memory, cite its Historical citation marker. "
        "Use the broader aggregate overall summary only as higher-level field-development context."
    )

    for idx, theme in enumerate(selected_themes, 1):
        candidate = candidate_map.get(theme["theme_id"], {})
        hit_lines = format_retrieved_hit_lines(candidate)
        history_structured = get_structured_history(theme)
        innovation_lines = format_theme_innovation_content_lines(theme)
        theme_overview = safe_text(history_structured.get("theme_overview"))
        overall_evolution = safe_text(history_structured.get("overall_evolution_summary"))

        blocks.append(
            f"=== Expert Theme {idx} ===\n"
            f"Theme ID: {theme.get('theme_id')}\n"
            f"Selection reason: {reason_map.get(theme.get('theme_id'), 'No explicit reason returned.')}\n"
            f"Theme title: {theme.get('title')}\n"
            f"Theme description: {theme.get('description')}\n"
            f"Theme overview: {theme_overview or 'Not available'}\n"
            f"Most directly matched retrieved innovation points:\n"
            f"{chr(10).join(hit_lines) if hit_lines else '- No retrieved hit points recorded.'}\n\n"
            f"Historical innovation-point contents inside this theme (primary comparison evidence):\n"
            f"{chr(10).join(innovation_lines) if innovation_lines else '- No theme innovation contents recorded.'}\n\n"
            f"Leaf-theme overall evolution summary:\n{overall_evolution or 'Not available'}\n\n"
            f"Broader field-development context from the direct parent aggregate node:\n"
            f"{format_parent_aggregate_overall_context(parent_map.get(theme.get('theme_id')))}"
        )

    return "\n\n".join(blocks)


def format_expert_knowledge_for_summary_prompt(expert_result):
    if not expert_result:
        return "Expert knowledge retrieval was disabled or unavailable."

    selected_themes = expert_result.get("selected_themes", [])
    if not selected_themes:
        return "No expert-memory themes were selected for this innovation point."

    candidate_map = {
        theme["theme_id"]: theme
        for theme in expert_result.get("candidate_leaf_themes", [])
    }
    reason_map = {
        item.get("theme_id"): item.get("reason", "")
        for item in expert_result.get("llm_decision", {}).get("selected_themes", [])
        if item.get("theme_id")
    }
    parent_map = expert_result.get("selected_parent_aggregates_by_theme_id", {}) or {}

    blocks = []
    selection_mode = expert_result.get("llm_decision", {}).get("selection_mode", "llm_selected")
    blocks.append(f"Expert-memory selection mode: {selection_mode}")
    blocks.append(
        "Use the historical evaluations, ratings, and historical-position judgments below as the PRIMARY expert-memory evidence for novelty-level judgment. "
        "When using a concrete historical paper from this memory, cite its Historical citation marker if citations are allowed in the current section. "
        "Use the broader aggregate overall summary to calibrate the larger field trajectory."
    )

    for idx, theme in enumerate(selected_themes, 1):
        candidate = candidate_map.get(theme["theme_id"], {})
        hit_lines = format_retrieved_hit_lines(candidate)
        history_structured = get_structured_history(theme)
        evaluation_lines = format_theme_evaluation_lines(theme)
        theme_overview = safe_text(history_structured.get("theme_overview"))
        overall_evolution = safe_text(history_structured.get("overall_evolution_summary"))

        blocks.append(
            f"=== Expert Theme {idx} ===\n"
            f"Theme ID: {theme.get('theme_id')}\n"
            f"Selection reason: {reason_map.get(theme.get('theme_id'), 'No explicit reason returned.')}\n"
            f"Theme title: {theme.get('title')}\n"
            f"Theme description: {theme.get('description')}\n"
            f"Theme overview: {theme_overview or 'Not available'}\n"
            f"Most directly matched retrieved innovation points:\n"
            f"{chr(10).join(hit_lines) if hit_lines else '- No retrieved hit points recorded.'}\n\n"
            f"Historical innovation-point evaluations inside this theme:\n"
            f"{chr(10).join(evaluation_lines) if evaluation_lines else '- No structured historical evaluations recorded.'}\n\n"
            f"Leaf-theme overall evolution summary:\n{overall_evolution or 'Not available'}\n\n"
            f"Broader field-development context from the direct parent aggregate node:\n"
            f"{format_parent_aggregate_overall_context(parent_map.get(theme.get('theme_id')))}"
        )

    return "\n\n".join(blocks)

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text")
        doc.close()
        return full_text
    except Exception as e:
        print(f"[ERROR] Error extracting text from PDF {pdf_path}: {e}")
        return None

def generate_queries_from_innovation_point(client, innovation_point, config, paper_name, point_num):
    system_prompt = config['prompts']['query_generation']['system_prompt']
    user_prompt = config['prompts']['query_generation']['user_prompt'].format(
        paper_name=paper_name,
        point_num=point_num,
        innovation_point=innovation_point.strip()
    )

    max_retries = config['llm_config']['max_retries']
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries} to generate queries...")
            response = client.chat.completions.create(
                model=config['llm_config']['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config['llm_config']['temperature']
            )
            response_content = response.choices[0].message.content
            lines = response_content.strip().split('\n')
            queries = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines if re.match(r'^\d+\.\s*', line.strip())]

            if len(queries) >= 6:
                return queries[:6]
            elif len(queries) >= 3:
                while len(queries) < 6:
                    queries.append(innovation_point.strip())
                return queries
            else:
                continue
        except Exception as e:
            if attempt >= max_retries - 1:
                break
    return [innovation_point.strip()] * 6

def get_knowledge_from_ragflow_multiple_queries(rag_object, queries, dataset_name, config):
    return get_knowledge_trace_from_ragflow_multiple_queries(
        rag_object, queries, dataset_name, config
    )["knowledge_text"]


def get_knowledge_trace_from_ragflow_multiple_queries(rag_object, queries, dataset_name, config):
    all_chunks = []
    query_records = []
    for i, query in enumerate(queries, 1):
        chunks = get_knowledge_from_ragflow(rag_object, query, dataset_name, config)
        query_records.append({
            "query_index": i,
            "query": query,
            "num_chunks": len(chunks) if chunks else 0,
            "chunks": chunks or [],
        })
        if chunks:
            all_chunks.extend(chunks)

    unique_chunks = []
    seen_chunks = set()
    for chunk in all_chunks:
        if chunk in seen_chunks:
            continue
        seen_chunks.add(chunk)
        unique_chunks.append(chunk)

    all_knowledge = ""
    for i, chunk in enumerate(unique_chunks, 1):
        all_knowledge += f"=== Chunk {i} ===\n{chunk}\n\n"

    return {
        "queries": query_records,
        "unique_chunks": unique_chunks,
        "knowledge_text": all_knowledge,
    }

def get_knowledge_from_ragflow(rag_object, query, dataset_name, config):
    try:
        datasets = rag_object.list_datasets(name=dataset_name)
        if not datasets:
            return []
        dataset = datasets[0]
        rag_config = config['rag']
        results = list(rag_object.retrieve(
            question=query,
            dataset_ids=[dataset.id],
            page=rag_config['page'],
            page_size=rag_config['page_size'],
            similarity_threshold=rag_config['similarity_threshold'],
            vector_similarity_weight=rag_config['vector_similarity_weight'],
            top_k=rag_config['top_k'],
            rerank_id=rag_config['rerank_id'],
            keyword=rag_config['keyword']
        ))
        chunks = []
        for c in results:
            try:
                doc = dataset.list_documents(id=c.document_id)
                document_name = doc[0].name if doc else "Unknown Document"
            except Exception:
                document_name = "Unknown Document"
            chunk_text = f"Source Document: {document_name}\nContent: {c.content}"
            chunks.append(chunk_text)
        return chunks
    except Exception as e:
        print(f"[ERROR] An error occurred in get_knowledge_from_ragflow: {e}")
        return []

def truncate_paper_text(paper_text, max_chars=50000):
    if len(paper_text) <= max_chars:
        return paper_text
    sections_to_keep = []
    abstract_match = re.search(r'(?i)(abstract\s*\n.*?)(?=\n\s*\d+\.?\s*introduction|\n\s*1\.?\s+)', paper_text, re.DOTALL)
    if abstract_match:
        sections_to_keep.append(abstract_match.group(1)[:5000])
    intro_match = re.search(r'(?i)(\d+\.?\s*introduction\s*\n.*?)(?=\n\s*\d+\.?\s*\w)', paper_text, re.DOTALL)
    if intro_match:
        sections_to_keep.append(intro_match.group(1)[:10000])
    method_match = re.search(r'(?i)(\d+\.?\s*(?:method|methodology|approach|proposed)\s*\n.*?)(?=\n\s*\d+\.?\s*\w)', paper_text, re.DOTALL)
    if method_match:
        sections_to_keep.append(method_match.group(1)[:15000])
    
    if sections_to_keep:
        truncated = "\n\n[...sections extracted...]\n\n".join(sections_to_keep)
        if len(truncated) < max_chars:
            remaining = max_chars - len(truncated)
            truncated = paper_text[:remaining//2] + "\n\n[...truncated...]\n\n" + truncated
        return truncated[:max_chars]
    else:
        return paper_text[:max_chars] + "\n\n[...content truncated due to length...]"

def parse_innovation_points(content):
    points = []
    matches = re.finditer(r'^\s*(\d+)\.\s*(.*?)(?=(?:\n\s*\d+\.\s)|$)', content, re.DOTALL | re.MULTILINE)
    for match in matches:
        points.append(match.group(2).strip())
    if not points and content:
        lines = content.strip().split('\n')
        current_point = ""
        for line in lines:
            if re.match(r'^\d+\.\s*', line):
                if current_point:
                    points.append(current_point.strip())
                current_point = line
            elif current_point:
                current_point += "\n" + line
        if current_point:
            points.append(current_point.strip())
    return [re.sub(r'^\d+\.\s*', '', p).strip() for p in points]

def limit_innovation_points(innovation_points, max_points=5):
    if len(innovation_points) <= max_points:
        return innovation_points
    return innovation_points[:max_points]


def build_comparison_system_prompt(config, knowledge, expert_knowledge_text):
    return format_prompt(
        config['prompts']['innovation_comparison']['system_prompt'],
        knowledge=knowledge,
        expert_knowledge=expert_knowledge_text or "Expert knowledge retrieval was disabled or unavailable."
    )


def build_comparison_user_prompt(config, user_prompt_key, paper_name, point_num, innovation_point, original_paper_text, expert_knowledge_text):
    return format_prompt(
        config['prompts']['innovation_comparison'][user_prompt_key],
        paper_name=paper_name,
        point_num=point_num,
        innovation_point=innovation_point.strip(),
        original_paper_text=original_paper_text,
        expert_knowledge=expert_knowledge_text or "Expert knowledge retrieval was disabled or unavailable."
    )

def compare_paper_innovations(config, paper_name, dataset_name, innovation_content, main_pdf_path=None):
    try:
        rag_object = RAGFlow(
            api_key=config['api']['api_key'],
            base_url=config['api']['base_url']
        )
        client = openai.OpenAI(
            api_key=config['api']['openai_api_key'],
            base_url=config['api']['openai_base_url'],
            timeout=config['api']['openai_timeout']
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize clients: {e}")
        return None

    print("[INFO] Starting innovation point comparison...")

    # Read full-text analysis mode toggle
    use_full_text = config.get('use_full_text_in_comparison', True)

    original_paper_text = ""
    if use_full_text:
        print("[INFO] Full text analysis mode: ON (using original paper text in comparison prompts)")
        if main_pdf_path and os.path.exists(main_pdf_path):
            original_paper_text = extract_text_from_pdf(main_pdf_path)
            if original_paper_text:
                original_paper_text = truncate_paper_text(original_paper_text, max_chars=50000)
                print(f"   Extracted and truncated original paper text: {len(original_paper_text)} characters")
            else:
                print("   [WARN] Failed to extract text from main PDF, proceeding without full text.")
        else:
            print("   [WARN] Main PDF path not available, proceeding without full text.")
    else:
        print("[INFO] Full text analysis mode: OFF (saving tokens, using only innovation descriptions and RAG knowledge)")

    innovation_points = parse_innovation_points(innovation_content)
    if not innovation_points:
        return None

    innovation_points = limit_innovation_points(innovation_points, max_points=5)
    comparison_data = []

    # Select prompt key based on mode
    if use_full_text:
        user_prompt_key = 'user_prompt'
    else:
        user_prompt_key = 'user_prompt_no_fulltext'

    for i, point in enumerate(innovation_points, 1):
        print(f"\n{'='*60}\n[INFO] Processing innovation point {i}/{len(innovation_points)}\n{'='*60}")
        try:
            queries = generate_queries_from_innovation_point(client, point, config, paper_name, i)
            rag_retrieval_trace = get_knowledge_trace_from_ragflow_multiple_queries(
                rag_object, queries, dataset_name, config
            )
            knowledge = rag_retrieval_trace.get("knowledge_text", "")
            try:
                expert_result = retrieve_expert_knowledge_for_innovation_point(client, point.strip(), config)
                expert_knowledge_text = format_expert_knowledge_for_comparison_prompt(expert_result)
                expert_knowledge_summary_text = format_expert_knowledge_for_summary_prompt(expert_result)
            except Exception as expert_error:
                print(f"[WARN] Expert knowledge retrieval failed for point {i}: {expert_error}")
                expert_result = None
                expert_knowledge_text = "Expert knowledge retrieval failed for this innovation point."
                expert_knowledge_summary_text = expert_knowledge_text

            if not knowledge:
                print(f"[WARN] No knowledge retrieved for point {i}.")
                continue

            system_prompt = build_comparison_system_prompt(
                config,
                knowledge,
                expert_knowledge_text
            )
            user_prompt = build_comparison_user_prompt(
                config,
                user_prompt_key,
                paper_name,
                i,
                point.strip(),
                original_paper_text,
                expert_knowledge_text
            )

            response = client.chat.completions.create(
                model=config['llm_config']['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config['llm_config']['temperature']
            )

            full_response = response.choices[0].message.content
            comparison_data.append({
                'point_number': i,
                'innovation_point': point.strip(),
                'content': full_response,
                'generated_queries': queries,
                'rag_retrieval_trace': rag_retrieval_trace,
                'rag_knowledge': knowledge,
                'expert_knowledge': expert_knowledge_text,
                'expert_knowledge_summary': expert_knowledge_summary_text,
                'expert_knowledge_result': expert_result,
                'comparison_prompt_inputs': {
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'used_full_text': use_full_text,
                },
            })
            print(f"[OK] Comparison for point {i} completed.")
            time.sleep(1)
        except Exception as e:
            print(f"[ERROR] Error processing innovation point {i}: {e}")
            continue

    return comparison_data
