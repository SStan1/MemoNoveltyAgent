"""
cluster_innovations.py
----------------------
Loads embeddings.npy + index.json, performs KMeans clustering (default k=400),
saves t-SNE visualization and cluster assignment results.

Usage:
    python cluster_innovations.py                  # default k=400
    python cluster_innovations.py --k 200          # override k
    python cluster_innovations.py --perplexity 50  # adjust t-SNE
"""

import os
from pathlib import Path
import json
import argparse
import numpy as np
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =========================================================================
# Paths
# =========================================================================

EMBEDDING_DIR = str(PROJECT_ROOT / "memory_database" / "embedding_output")
CLUSTER_OUTPUT_DIR = str(PROJECT_ROOT / "cluster_output")


# =========================================================================
# Loading
# =========================================================================

def load_data(embedding_dir):
    embeddings_path = os.path.join(embedding_dir, "embeddings.npy")
    index_path = os.path.join(embedding_dir, "index.json")

    print(f"[INFO] Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    print(f"[INFO] Loading index from {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    print(f"[INFO] Index entries: {len(index)}")

    assert len(index) == embeddings.shape[0], \
        f"Mismatch: {len(index)} index entries vs {embeddings.shape[0]} embeddings"

    return embeddings, index


# =========================================================================
# KMeans
# =========================================================================

def run_kmeans(embeddings, k=400, random_state=42):
    print(f"\n[KMEANS] Running KMeans with k={k} (n_init=10)...")
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10, verbose=1)
    labels = km.fit_predict(embeddings)

    print(f"[KMEANS] Computing silhouette score (may take a moment)...")
    sil = silhouette_score(embeddings, labels, sample_size=min(10000, len(labels)))
    print(f"[KMEANS] Silhouette score: {sil:.4f}")
    print(f"[KMEANS] Inertia: {km.inertia_:.2f}")

    # Cluster size statistics
    unique, counts = np.unique(labels, return_counts=True)
    print(f"[KMEANS] Cluster size stats:")
    print(f"  Total clusters: {len(unique)}")
    print(f"  Min size:  {counts.min()}")
    print(f"  Max size:  {counts.max()}")
    print(f"  Mean size: {counts.mean():.1f}")
    print(f"  Median:    {np.median(counts):.1f}")

    sorted_idx = np.argsort(-counts)
    print(f"  Top-10 largest:  {list(zip(unique[sorted_idx[:10]], counts[sorted_idx[:10]]))}")
    print(f"  Top-10 smallest: {list(zip(unique[sorted_idx[-10:]], counts[sorted_idx[-10:]]))}")

    return labels, km, {"silhouette": sil, "inertia": float(km.inertia_), "k": k}


# =========================================================================
# Visualization
# =========================================================================

def plot_clusters(coords_2d, labels, k, sil_score, save_path):
    """Plot t-SNE scatter colored by cluster assignment."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    norm = plt.Normalize(vmin=labels.min(), vmax=labels.max())
    cmap = cm.get_cmap("nipy_spectral")

    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=labels, cmap=cmap, norm=norm,
        s=6, alpha=0.6, edgecolors="none",
    )

    ax.set_title(f"KMeans (k={k}, silhouette={sil_score:.3f}, n={len(labels)} points)",
                 fontsize=14)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Cluster ID")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Plot: {save_path}")


# =========================================================================
# Save cluster assignments
# =========================================================================

def save_cluster_results(index, labels, metrics, output_dir):
    """
    Save:
      - cluster_kmeans.json: every point with its cluster label + source info
      - cluster_kmeans_summary.json: per-cluster member list + stats
    """
    assignments = []
    for i, rec in enumerate(index):
        assignments.append({
            "embedding_idx": i,
            "cluster": int(labels[i]),
            "paper_id": rec["paper_id"],
            "source_filename": rec["source_filename"],
            "title": rec["title"],
            "year": rec["year"],
            "point_index": rec["point_index"],
            "point_text": rec["point_text"],
        })

    assignment_path = os.path.join(output_dir, "cluster_kmeans.json")
    with open(assignment_path, "w", encoding="utf-8") as f:
        json.dump(assignments, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] Assignments: {assignment_path}")

    # --- Per-cluster summary ---
    clusters = defaultdict(list)
    for a in assignments:
        clusters[a["cluster"]].append({
            "title": a["title"],
            "source_filename": a["source_filename"],
            "year": a["year"],
            "point_index": a["point_index"],
            "point_text_preview": (a["point_text"][:150] + "...")
                                  if len(a["point_text"]) > 150 else a["point_text"],
        })

    summary = {
        "method": "kmeans",
        "metrics": metrics,
        "num_points": len(assignments),
        "num_clusters": metrics["k"],
        "clusters": {},
    }

    for cid in sorted(clusters.keys()):
        members = clusters[cid]
        unique_papers = set(m["source_filename"] for m in members)
        year_dist = defaultdict(int)
        for m in members:
            year_dist[m["year"] or "unknown"] += 1

        summary["clusters"][str(cid)] = {
            "size": len(members),
            "unique_papers": len(unique_papers),
            "year_distribution": dict(sorted(year_dist.items(),
                                             key=lambda x: (isinstance(x[0], str), x[0]))),
            "members": members,
        }

    summary_path = os.path.join(output_dir, "cluster_kmeans_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] Summary: {summary_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="KMeans clustering of innovation embeddings")
    parser.add_argument("--k", type=int, default=400,
                        help="Number of KMeans clusters (default: 400)")
    parser.add_argument("--perplexity", type=float, default=50,
                        help="t-SNE perplexity (default: 50)")
    args = parser.parse_args()

    os.makedirs(CLUSTER_OUTPUT_DIR, exist_ok=True)

    # --- Load data ---
    embeddings, index = load_data(EMBEDDING_DIR)

    # --- KMeans ---
    labels, km_model, metrics = run_kmeans(embeddings, k=args.k)

    # --- t-SNE ---
    print(f"\n[INFO] Computing t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42,
                init="pca", learning_rate="auto")
    coords_2d = tsne.fit_transform(embeddings)

    tsne_path = os.path.join(CLUSTER_OUTPUT_DIR, "tsne_coords.npy")
    np.save(tsne_path, coords_2d)
    print(f"[SAVED] t-SNE coords: {tsne_path}")

    # --- Plot ---
    plot_path = os.path.join(CLUSTER_OUTPUT_DIR, "kmeans_tsne.png")
    plot_clusters(coords_2d, labels, args.k, metrics["silhouette"], plot_path)

    # --- Save results ---
    save_cluster_results(index, labels, metrics, CLUSTER_OUTPUT_DIR)

    # --- Save centroids ---
    centroids_path = os.path.join(CLUSTER_OUTPUT_DIR, "kmeans_centroids.npy")
    np.save(centroids_path, km_model.cluster_centers_)
    print(f"[SAVED] Centroids: {centroids_path} (shape: {km_model.cluster_centers_.shape})")

    # --- Done ---
    print(f"\n{'='*80}")
    print("DONE")
    print(f"  Points:          {len(labels)}")
    print(f"  Clusters (k):    {args.k}")
    print(f"  Silhouette:      {metrics['silhouette']:.4f}")
    print(f"  Output dir:      {CLUSTER_OUTPUT_DIR}")
    print(f"{'='*80}")
    print(f"\nFiles:")
    for fname in sorted(os.listdir(CLUSTER_OUTPUT_DIR)):
        fpath = os.path.join(CLUSTER_OUTPUT_DIR, fname)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            size_str = f"{size/1024/1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {fname:45s} {size_str}")


if __name__ == "__main__":
    main()
