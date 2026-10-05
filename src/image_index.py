"""
Builds and queries a FAISS index over RAM-H1200 hand X-ray embeddings.
Each indexed case carries its SvdH BE/JSN scores, so retrieval returns
real, scored cases -- not just visually similar images.

Run this once (via scripts/build_image_index.py) to build the index,
then use `search_similar_cases()` at query time.
"""
import json
import faiss
import numpy as np
from pathlib import Path
from config import (
    RAM_H1200_BE_SCORES,
    RAM_H1200_JSN_SCORES,
    RAM_H1200_ROOT,
    INDICES_DIR,
    TOP_K_IMAGE_CASES,
)
from src.embeddings import embed_images_batch, embed_image

IMAGE_INDEX_PATH = INDICES_DIR / "image_index.faiss"
IMAGE_METADATA_PATH = INDICES_DIR / "image_metadata.json"


def _load_scores() -> dict:
    """Merge BE and JSN score dicts, keyed by image filename."""
    with open(RAM_H1200_BE_SCORES) as f:
        be_scores = json.load(f)
    with open(RAM_H1200_JSN_SCORES) as f:
        jsn_scores = json.load(f)
    merged = {}
    for fname, scores in be_scores.items():
        merged.setdefault(fname, {})["BE"] = scores
    for fname, scores in jsn_scores.items():
        merged.setdefault(fname, {})["JSN"] = scores
    return merged


def build_image_index(segmentation_split: str = "train") -> None:
    """
    Build the FAISS index. Adjust the glob below to match how you've laid
    out the downloaded dataset locally.
    """
    scores_by_file = _load_scores()
    image_dir = RAM_H1200_ROOT / "Segmentation" / segmentation_split
    image_paths = sorted(image_dir.glob("*.bmp"))

    print(f"Found {len(image_paths)} images. Embedding in batches...")
    embeddings = embed_images_batch([str(p) for p in image_paths])
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
    index.add(embeddings)

    metadata = []
    for p in image_paths:
        metadata.append({
            "filename": p.name,
            "path": str(p),
            "scores": scores_by_file.get(p.name, {}),
        })

    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(IMAGE_INDEX_PATH))
    with open(IMAGE_METADATA_PATH, "w") as f:
        json.dump(metadata, f)

    print(f"Index built with {index.ntotal} vectors -> {IMAGE_INDEX_PATH}")


def search_similar_cases(query_image_path: str, top_k: int = TOP_K_IMAGE_CASES) -> "list[dict]":
    """Given a newly uploaded X-ray, return the top-k most similar scored cases."""
    index = faiss.read_index(str(IMAGE_INDEX_PATH))
    with open(IMAGE_METADATA_PATH) as f:
        metadata = json.load(f)

    query_vec = np.array([embed_image(query_image_path)], dtype="float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        case = metadata[idx].copy()
        case["similarity"] = float(score)
        results.append(case)
    return results
