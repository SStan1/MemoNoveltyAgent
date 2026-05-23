"""
embed_innovations.py
--------------------
Reads initial memory JSONs, splits innovation_points into individual points,
embeds each point using Qwen3-Embedding-8B, and saves:
  1. embeddings.npy  — (N, dim) float32 numpy array
  2. index.json      — list of N records, each mapping an embedding back to
                        its source paper and innovation point text

Usage:
    python embed_innovations.py                          # default: 1000 points
    python embed_innovations.py --max-points 5000        # embed more
    python embed_innovations.py --batch-size 16          # adjust batch size
    python embed_innovations.py --gpus 2,3               # specify GPU IDs
"""

import os
from pathlib import Path
import re
import json
import time
import argparse
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =========================================================================
# Config / paths
# =========================================================================

MEMORY_OUTPUT_DIR = str(PROJECT_ROOT / "memory_output")
EMBEDDING_OUTPUT_DIR = str(PROJECT_ROOT / "memory_database" / "embedding_output")
MODEL_PATH = os.environ.get("MEMORY_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")


# =========================================================================
# Innovation point parsing
# =========================================================================

def split_innovation_points(text):
    """
    Split the LLM-generated innovation_points text into individual points.

    Expected format:
        1. (Classification: ...) Some text...
        2. (Classification: ...) Some text...
        ...

    Returns a list of strings, one per innovation point.
    """
    if not text or not text.strip():
        return []

    # Split by numbered list pattern: "1." "2." etc. at the start of a line
    # The pattern looks for a number followed by a period at line start
    # (possibly with leading whitespace)
    parts = re.split(r"(?m)(?=^\s*\d+\.\s)", text.strip())

    points = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        # Verify it starts with a number pattern (skip any preamble text)
        if re.match(r"^\d+\.\s", cleaned):
            points.append(cleaned)

    return points


# =========================================================================
# Collect points from memory JSONs
# =========================================================================

def collect_innovation_points(memory_dir, max_points=None):
    """
    Read all individual memory JSONs from memory_dir,
    split their innovation_points, and return a list of records.

    If max_points is None, collect all points (no limit).

    Each record: {
        "paper_id": str,
        "source_filename": str,
        "title": str,
        "year": int or None,
        "point_index": int,          # 0-based index within this paper
        "point_text": str,           # the individual innovation point text
    }
    """
    # Find all individual paper JSONs (exclude all_memories.json and debug/)
    json_files = []
    for fname in sorted(os.listdir(memory_dir)):
        if fname.endswith(".json") and fname != "all_memories.json":
            json_files.append(os.path.join(memory_dir, fname))

    print(f"[INFO] Found {len(json_files)} memory JSON files in {memory_dir}")
    if max_points:
        print(f"[INFO] Max points limit: {max_points}")
    else:
        print(f"[INFO] No limit, embedding all points")

    records = []
    papers_used = 0
    papers_skipped = 0

    for jpath in json_files:
        if max_points and len(records) >= max_points:
            break

        try:
            with open(jpath, "r", encoding="utf-8") as f:
                mem = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {jpath}: {e}")
            continue

        innovation_text = mem.get("innovation_points")
        if not innovation_text:
            papers_skipped += 1
            continue

        points = split_innovation_points(innovation_text)
        if not points:
            papers_skipped += 1
            continue

        papers_used += 1
        paper_id = mem.get("paper_id", "unknown")
        title = mem.get("title", "unknown")
        metadata = mem.get("metadata", {})
        source_filename = metadata.get("source_filename", os.path.basename(jpath))
        year = metadata.get("year")

        for idx, pt in enumerate(points):
            if max_points and len(records) >= max_points:
                break
            records.append({
                "paper_id": paper_id,
                "source_filename": source_filename,
                "title": title,
                "year": year,
                "point_index": idx,
                "point_text": pt,
            })

    print(f"[INFO] Collected {len(records)} innovation points "
          f"from {papers_used} papers ({papers_skipped} papers skipped)")

    # Print year distribution
    year_dist = {}
    for r in records:
        y = r["year"] or "unknown"
        year_dist[y] = year_dist.get(y, 0) + 1
    print(f"[INFO] Points per year:")
    for y in sorted(year_dist.keys(), key=lambda x: (isinstance(x, str), x)):
        print(f"  {y}: {year_dist[y]}")

    return records


# =========================================================================
# Embedding model
# =========================================================================

def load_embedding_model(model_path, gpu_ids):
    """
    Load Qwen3-Embedding-8B in fp16 on specified GPUs.
    Returns (tokenizer, model).
    """
    print(f"[INFO] Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[INFO] Loading model in fp16 on GPU(s) {gpu_ids}...")

    if len(gpu_ids) == 1:
        device = f"cuda:{gpu_ids[0]}"
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
    else:
        # Multi-GPU: use device_map with max_memory to restrict to specified GPUs
        max_memory = {i: "22GiB" for i in gpu_ids}
        max_memory["cpu"] = "1GiB"
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
        )

    model.eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[INFO] Model loaded: {param_count:.1f}B parameters")

    return tokenizer, model


def embed_texts(texts, tokenizer, model, batch_size=8, max_length=512):
    """
    Embed a list of texts using mean pooling on the last hidden state.
    Returns numpy array of shape (len(texts), hidden_dim).
    """
    all_embeddings = []

    # Determine device (for single-GPU case)
    if hasattr(model, "device"):
        device = model.device
    else:
        # Multi-GPU device_map: input goes to the first module's device
        device = next(model.parameters()).device

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        if batch_num % 10 == 1 or batch_num == total_batches:
            print(f"  [EMBED] Batch {batch_num}/{total_batches} "
                  f"({i}/{len(texts)} texts)")

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        # Mean pooling: average over non-padding tokens
        token_embeddings = outputs.last_hidden_state  # (B, seq_len, hidden)
        attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, seq_len, 1)
        sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
        count = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_embeddings = sum_embeddings / count  # (B, hidden)

        # Normalize to unit vectors (for cosine similarity)
        mean_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)

        all_embeddings.append(mean_embeddings.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Embed innovation points using Qwen3-Embedding-8B")
    parser.add_argument("--max-points", type=int, default=0,
                        help="Max number of innovation points to embed (0 = all, default: all)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for embedding (default: 8)")
    parser.add_argument("--gpus", type=str, default="1,2",
                        help="Comma-separated GPU IDs to use (default: 1,2)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length per text (default: 512)")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    # After setting CUDA_VISIBLE_DEVICES, remap to 0-indexed
    remapped_ids = list(range(len(gpu_ids)))

    print(f"[INFO] Using GPU(s): {gpu_ids} (remapped to {remapped_ids})")

    # --- Output directory ---
    os.makedirs(EMBEDDING_OUTPUT_DIR, exist_ok=True)
    embeddings_path = os.path.join(EMBEDDING_OUTPUT_DIR, "embeddings.npy")
    index_path = os.path.join(EMBEDDING_OUTPUT_DIR, "index.json")

    # --- Step 1: Collect innovation points ---
    print(f"\n{'='*80}")
    print("Step 1: Collecting innovation points from memory JSONs")
    print(f"{'='*80}")
    max_pts = args.max_points if args.max_points > 0 else None
    records = collect_innovation_points(MEMORY_OUTPUT_DIR, max_points=max_pts)

    if not records:
        print("[ERROR] No innovation points found. Run build_initial_memory.py first.")
        return

    texts = [r["point_text"] for r in records]
    print(f"[INFO] Total texts to embed: {len(texts)}")

    # Show a sample
    print(f"\n[SAMPLE] First innovation point:")
    print(f"  Paper: {records[0]['title']}")
    print(f"  Text:  {texts[0][:200]}...")

    # --- Step 2: Load model ---
    print(f"\n{'='*80}")
    print("Step 2: Loading embedding model")
    print(f"{'='*80}")
    start_load = time.time()
    tokenizer, model = load_embedding_model(MODEL_PATH, remapped_ids)
    print(f"[INFO] Model loaded in {time.time() - start_load:.1f}s")

    # --- Step 3: Embed ---
    print(f"\n{'='*80}")
    print("Step 3: Embedding innovation points")
    print(f"{'='*80}")
    start_embed = time.time()
    embeddings = embed_texts(
        texts, tokenizer, model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    embed_time = time.time() - start_embed
    print(f"[INFO] Embedding completed in {embed_time:.1f}s "
          f"({embed_time/len(texts):.2f}s per point)")
    print(f"[INFO] Embedding shape: {embeddings.shape}")

    # --- Step 4: Save ---
    print(f"\n{'='*80}")
    print("Step 4: Saving results")
    print(f"{'='*80}")

    # Save embeddings
    np.save(embeddings_path, embeddings)
    size_mb = os.path.getsize(embeddings_path) / 1024 / 1024
    print(f"[SAVED] Embeddings: {embeddings_path} ({size_mb:.1f} MB)")

    # Save index (without point_text to keep it lean, but include it for traceability)
    index_records = []
    for i, rec in enumerate(records):
        index_records.append({
            "embedding_idx": i,
            "paper_id": rec["paper_id"],
            "source_filename": rec["source_filename"],
            "title": rec["title"],
            "year": rec["year"],
            "point_index": rec["point_index"],
            "point_text": rec["point_text"],
        })

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_records, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] Index: {index_path} ({len(index_records)} entries)")

    # --- Summary ---
    print(f"\n{'='*80}")
    print("DONE")
    print(f"  Points embedded:    {len(texts)}")
    print(f"  Embedding dim:      {embeddings.shape[1]}")
    print(f"  Embedding time:     {embed_time:.1f}s")
    print(f"  Embeddings file:    {embeddings_path}")
    print(f"  Index file:         {index_path}")
    print(f"  Output dir:         {EMBEDDING_OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
