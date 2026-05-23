"""
retrieve_query_cluster.py
-------------------------
Static retrieval-and-assignment script for a single hard-coded innovation query.

This version uses the STATIC final memory library produced by
write_cluster_development_history.py.

Pipeline:
1. Embed the query innovation point using the same embedding model as the memory bank
2. Retrieve the top-K most similar innovation points by cosine similarity
3. Identify which FINAL leaf themes those points belong to using final_memory_library.json
4. Ask the LLM to choose the highly relevant final leaf themes for the query
5. Return the retrieved top-K points, candidate leaf themes, and the selected themes with full information

Outputs:
  <project-root>/retrieval_output/
    retrieval_result.json
"""

import os
import json
import time
import sys
import argparse
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import openai
import torch
from transformers import AutoTokenizer, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONSTRUCT_MEMORY_DIR = str(PROJECT_ROOT / "Construct_memory")
EMBEDDING_OUTPUT_DIR = str(PROJECT_ROOT / "memory_database" / "embedding_output")
FINAL_MEMORY_OUTPUT_DIR = str(PROJECT_ROOT / "memory_database" / "final_memory_output")
RETRIEVAL_OUTPUT_DIR = str(PROJECT_ROOT / "retrieval_output")
LOG_DIR = str(PROJECT_ROOT / "logs")
MODEL_PATH = os.environ.get("MEMORY_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")



EXPERIMENTAL_QUERY = (
    "(Classification: Dataset/Benchmark) A comprehensive suite of simulated clinical case datasets spanning "
    "diverse medical specialties, languages, and real-world electronic health records; HOW: This benchmark "
    "fills the gap of limited diversity, realism, and multimodal support in existing conversational medical "
    "evaluations by structuring cases from real electronic health records, multimodal medical journal challenges, "
    "nine distinct medical specialties, and seven different languages. This enables new research questions "
    "regarding how well language models generalize across linguistic barriers, specialized medical domains, "
    "and complex multimodal diagnostic tasks where image readings and physical examination findings must be "
    "requested and interpreted; Evidence: The benchmark's utility is validated through extensive evaluations "
    "of models across hundreds of cases, alongside a clinical reader study involving human medical doctors who "
    "rated the generated patient-doctor dialogues for realism, measurement accuracy, and empathy."
)


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
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
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


def load_embedding_model(model_path, gpu_id):
    print(f"[INFO] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    device = f"cuda:{gpu_id}" if (gpu_id is not None and int(gpu_id) >= 0 and torch.cuda.is_available()) else "cpu"
    print(f"[INFO] Loading embedding model on {device}")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def embed_single_text(text, tokenizer, model, device, max_length=512):
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    token_embeddings = outputs.last_hidden_state
    attention_mask = encoded["attention_mask"].unsqueeze(-1)
    sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
    count = attention_mask.sum(dim=1).clamp(min=1e-9)
    embedding = sum_embeddings / count
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding[0].cpu().float().numpy()


def load_embeddings_and_index():
    embeddings_path = os.path.join(EMBEDDING_OUTPUT_DIR, "embeddings.npy")
    index_path = os.path.join(EMBEDDING_OUTPUT_DIR, "index.json")

    embeddings = np.load(embeddings_path)
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    print(f"[INFO] Loaded embeddings: {embeddings.shape}")
    print(f"[INFO] Loaded index entries: {len(index)}")
    return embeddings, index


def extract_final_themes(final_library):
    if isinstance(final_library, dict):
        themes = final_library.get("themes", [])
        if isinstance(themes, list):
            return themes
    if isinstance(final_library, list):
        return final_library
    raise ValueError("Unsupported final memory library format: expected {'themes': [...]} or a list of themes")


def load_final_memory_library():
    path = os.path.join(FINAL_MEMORY_OUTPUT_DIR, "final_memory_library.json")
    with open(path, "r", encoding="utf-8") as f:
        library = json.load(f)
    themes = extract_final_themes(library)
    print(f"[INFO] Loaded final memory library: {path}")
    print(f"[INFO] Final themes: {len(themes)}")
    return library


def build_theme_lookup(final_library):
    embedding_to_theme = {}
    theme_lookup = {}

    for theme in extract_final_themes(final_library):
        theme_id = theme["theme_id"]
        theme_lookup[theme_id] = theme

        for member in theme.get("members", []):
            embedding_idx = member.get("embedding_idx")
            if embedding_idx is None:
                continue
            embedding_to_theme[int(embedding_idx)] = {
                "theme_id": theme_id,
                "theme_title": theme.get("title"),
                "theme_description": theme.get("description"),
            }

    return embedding_to_theme, theme_lookup


def retrieve_top_k(query_embedding, embeddings, index, embedding_to_theme, top_k=10):
    scores = embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        item = dict(index[int(idx)])
        item["embedding_idx"] = int(idx)
        item["rank"] = rank
        item["score"] = float(scores[int(idx)])
        item.update(embedding_to_theme.get(int(idx), {}))
        results.append(item)
    return results


def summarize_candidate_themes(retrieved_points, theme_lookup):
    candidate_themes = {}

    for item in retrieved_points:
        theme_id = item.get("theme_id")
        if theme_id is None:
            continue
        if theme_id not in candidate_themes:
            theme = theme_lookup[theme_id]
            candidate_themes[theme_id] = {
                "theme_id": theme_id,
                "source_cluster_id": theme.get("source_cluster_id"),
                "title": theme.get("title"),
                "description": theme.get("description"),
                "num_hits": 0,
                "hit_ranks": [],
                "hit_points": [],
                "num_members": len(theme.get("members", [])),
            }

        candidate_themes[theme_id]["num_hits"] += 1
        candidate_themes[theme_id]["hit_ranks"].append(item["rank"])
        candidate_themes[theme_id]["hit_points"].append({
            "rank": item["rank"],
            "score": item["score"],
            "paper_title": item.get("title"),
            "year": item.get("year"),
            "innovation_point": item.get("point_text"),
        })

    ordered = sorted(candidate_themes.values(), key=lambda x: (-x["num_hits"], min(x["hit_ranks"])))
    return ordered


def build_theme_selection_prompt(query_text, candidate_themes):
    system_prompt = (
        "You are an expert at assigning a new innovation claim to the most appropriate existing leaf-level memory themes. "
        "Use semantic similarity, the query innovation content, and the grouped retrieved innovation points under each candidate theme "
        "to choose only the candidate themes that are highly relevant to the query. Focus on leaf-level fit only."
    )

    theme_blocks = []
    for theme in candidate_themes:
        hit_examples = "\n".join(
            f"  - rank {p['rank']} | score {p['score']:.4f} | paper: {p['paper_title']} ({p['year']})\n"
            f"    innovation point: {p['innovation_point']}"
            for p in theme.get("hit_points", [])
        )
        theme_blocks.append(
            f"Theme {theme['theme_id']}\n"
            f"Theme title: {theme.get('title')}\n"
            f"Theme description: {theme.get('description')}\n"
            f"Theme size: {theme.get('num_members')} innovation points\n"
            f"Retrieved hits in top-k: {theme['num_hits']}\n"
            f"Retrieved innovation points grouped under this theme:\n{hit_examples}"
        )

    max_selection_count = min(2, len(candidate_themes))
    user_prompt = f"""
You need to identify the existing leaf themes that are highly relevant to the query innovation point.

Decision principles:
1. Prioritize semantic and methodological fit.
2. Use the grouped retrieved innovation points under each theme as the main evidence.
3. Use the candidate theme title and description as supporting context, not as the sole basis.
4. Prefer themes whose retrieved innovation points are strongly aligned with the query in specific content, not just in broad wording.
5. Return at least 1 theme and at most {max_selection_count} themes.
6. Only include a theme if its relevance is clearly high. Do not include weakly related themes just to reach the maximum.
7. Rank the returned themes from strongest match to weaker match.

Return ONLY a JSON object with this schema:
{{
  "selected_themes": [
    {{
      "theme_id": "12_C1",
      "rank": 1,
      "reason": "short explanation of why this theme is highly relevant"
    }},
    {{
      "theme_id": "7_C2",
      "rank": 2,
      "reason": "short explanation of why this theme is also highly relevant"
    }}
  ],
  "overall_reason": "short explanation of why these themes are the highly relevant ones"
}}

Query Innovation Point:
{query_text}

Candidate leaf themes with their grouped retrieved innovation points:

{"\n\n".join(theme_blocks)}
""".strip()

    return system_prompt, user_prompt


def resolve_selected_theme_ids(llm_decision, candidate_themes):
    candidate_ids = [theme["theme_id"] for theme in candidate_themes]
    candidate_id_set = set(candidate_ids)
    max_count = min(2, len(candidate_ids))

    selected_ids = []
    for item in llm_decision.get("selected_themes", []):
        theme_id = item.get("theme_id")
        if theme_id in candidate_id_set and theme_id not in selected_ids:
            selected_ids.append(theme_id)

    for theme_id in llm_decision.get("selected_theme_ids", []):
        if theme_id in candidate_id_set and theme_id not in selected_ids:
            selected_ids.append(theme_id)

    single_theme_id = llm_decision.get("selected_theme_id")
    if single_theme_id in candidate_id_set and single_theme_id not in selected_ids:
        selected_ids.append(single_theme_id)

    return selected_ids[:max_count]


def build_auto_selection_result(candidate_themes):
    selected_ids = [candidate_themes[0]["theme_id"]]
    llm_decision = {
        "selected_themes": [
            {
                "theme_id": candidate_themes[0]["theme_id"],
                "rank": 1,
                "reason": (
                    "Automatically selected because the retrieved top-k innovation points map to exactly one "
                    "candidate leaf theme."
                ),
            }
        ],
        "overall_reason": (
            "LLM selection skipped because the retrieved innovation points map to exactly one candidate leaf theme."
        ),
        "selection_mode": "automatic_single_candidate",
    }
    return selected_ids, llm_decision


def build_rank1_fallback_result(fallback_theme_id):
    llm_decision = {
        "selected_themes": [
            {
                "theme_id": fallback_theme_id,
                "rank": 1,
                "reason": (
                    "Fallback selection because the LLM failed 3 times to return at least one valid theme. "
                    "Using the theme of the rank-1 retrieved innovation point."
                ),
            }
        ],
        "overall_reason": (
            "LLM selection failed after 3 retries, so the result falls back to the theme attached to the "
            "highest-similarity retrieved innovation point."
        ),
        "selection_mode": "fallback_rank1_retrieved_point_theme",
    }
    return [fallback_theme_id], llm_decision


def get_rank1_theme_id(retrieved_points, candidate_themes):
    if retrieved_points:
        rank1_theme_id = retrieved_points[0].get("theme_id")
        if rank1_theme_id:
            return rank1_theme_id
    return candidate_themes[0]["theme_id"]


def select_themes_with_retry(client, config, query_text, candidate_themes, retrieved_points, max_attempts=3):
    fallback_theme_id = get_rank1_theme_id(retrieved_points, candidate_themes)

    for attempt in range(1, max_attempts + 1):
        system_prompt, user_prompt = build_theme_selection_prompt(query_text, candidate_themes)
        llm_decision = call_llm_json(client, config, system_prompt, user_prompt)
        selected_theme_ids = resolve_selected_theme_ids(llm_decision, candidate_themes)

        if selected_theme_ids:
            if attempt > 1:
                llm_decision["selection_mode"] = f"llm_retry_success_attempt_{attempt}"
            return selected_theme_ids, llm_decision

        print(
            f"[WARN] LLM selection attempt {attempt}/{max_attempts} returned no valid theme IDs"
        )

    print(
        "[WARN] LLM selection failed after 3 attempts; falling back to the theme of the rank-1 retrieved innovation point"
    )
    return build_rank1_fallback_result(fallback_theme_id)


def main():
    parser = argparse.ArgumentParser(description="Retrieve nearest innovations and assign a query to 1-2 highly relevant final leaf themes")
    parser.add_argument("--top-k", type=int, default=10, help="Number of nearest innovation points to retrieve (default: 10)")
    parser.add_argument("--gpu", type=int, default=1, help="GPU id for query embedding (default: 1)")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length for query embedding (default: 512)")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        f"retrieve_query_cluster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    sys.stdout = Logger(log_path)
    sys.stderr = sys.stdout
    print(f"[INFO] Log file: {log_path}")

    os.makedirs(RETRIEVAL_OUTPUT_DIR, exist_ok=True)

    config = load_config()
    client = build_client(config)

    embeddings, index = load_embeddings_and_index()
    final_library = load_final_memory_library()
    embedding_to_theme, theme_lookup = build_theme_lookup(final_library)

    print(f"[INFO] Embedding query using the same model as the memory bank")
    tokenizer, model, device = load_embedding_model(MODEL_PATH, args.gpu)
    query_embedding = embed_single_text(EXPERIMENTAL_QUERY, tokenizer, model, device, max_length=args.max_length)
    print(f"[INFO] Query embedding shape: {query_embedding.shape}")

    retrieved_points = retrieve_top_k(query_embedding, embeddings, index, embedding_to_theme, top_k=args.top_k)
    print(f"\n[INFO] Top-{args.top_k} retrieved innovation points:")
    for item in retrieved_points:
        print(
            f"  rank={item['rank']:>2} score={item['score']:.4f} theme={item.get('theme_id')} "
            f"paper={item.get('title')}"
        )

    candidate_themes = summarize_candidate_themes(retrieved_points, theme_lookup)
    if not candidate_themes:
        raise RuntimeError("No candidate leaf themes found from the retrieved innovation points.")

    print(f"\n[INFO] Candidate leaf themes from retrieved points:")
    for theme in candidate_themes:
        print(
            f"  theme={theme['theme_id']} hits={theme['num_hits']} "
            f"title={theme.get('title')}"
        )

    if len(candidate_themes) == 1:
        selected_theme_ids, llm_decision = build_auto_selection_result(candidate_themes)
        print(f"\n[INFO] Skipping LLM selection because there is only 1 candidate leaf theme")
    else:
        selected_theme_ids, llm_decision = select_themes_with_retry(
            client,
            config,
            EXPERIMENTAL_QUERY,
            candidate_themes,
            retrieved_points,
            max_attempts=3,
        )

    selected_themes = [theme_lookup[theme_id] for theme_id in selected_theme_ids]

    print(f"\n[INFO] LLM selected themes: {selected_theme_ids}")
    print(f"[INFO] LLM decision: {json.dumps(llm_decision, ensure_ascii=False, indent=2)}")
    for theme in selected_themes:
        print(f"[INFO] Selected theme preview: {theme['theme_id']} | {theme.get('title')}")
        print(theme.get("development_history", "")[:1000])

    result = {
        "query": {
            "text": EXPERIMENTAL_QUERY,
            "embedding_method": "Qwen3-Embedding-8B mean pooling + L2 normalization",
            "retrieval_method": "cosine similarity over normalized innovation embeddings",
            "query_needs_embedding": True,
        },
        "retrieved_points": retrieved_points,
        "candidate_leaf_themes": candidate_themes,
        "llm_decision": llm_decision,
        "selected_themes": selected_themes,
        "final_return": {
            "selected_theme_ids": selected_theme_ids,
            "selected_themes": selected_themes,
        },
    }

    output_path = os.path.join(RETRIEVAL_OUTPUT_DIR, "retrieval_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] Retrieval result: {output_path}")
    print(f"[DONE] Experimental retrieval finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        traceback.print_exc()
        raise
