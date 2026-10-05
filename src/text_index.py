"""
Hybrid retrieval over your RA text knowledge base (guidelines, literature,
patient education material). Combines dense retrieval (FAISS over sentence
embeddings) with sparse retrieval (BM25 keyword matching), then fuses the
two ranked lists.

Put your source text files (.txt) as one-chunk-per-file OR pre-chunked
text in data/guidelines/ before running build_text_index().
"""
import json
import re
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from config import GUIDELINES_DIR, INDICES_DIR, TEXT_EMBED_MODEL_NAME, TOP_K_TEXT_PASSAGES, HYBRID_ALPHA

TEXT_INDEX_PATH = INDICES_DIR / "text_index.faiss"
TEXT_METADATA_PATH = INDICES_DIR / "text_metadata.json"
BM25_CORPUS_PATH = INDICES_DIR / "bm25_corpus.json"

_embed_model = SentenceTransformer(TEXT_EMBED_MODEL_NAME)


def _chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> "list[str]":
    """Simple word-count based chunking. Swap for a smarter splitter if needed."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def _tokenize(text: str) -> "list[str]":
    return re.findall(r"\w+", text.lower())


def build_text_index() -> None:
    """Read every .txt file in data/guidelines/, chunk it, and build both indices."""
    all_chunks = []
    all_sources = []

    for fpath in sorted(GUIDELINES_DIR.glob("*.txt")):
        text = fpath.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        all_chunks.extend(chunks)
        all_sources.extend([fpath.name] * len(chunks))

    if not all_chunks:
        raise ValueError(
            f"No .txt files found in {GUIDELINES_DIR}. Add your RA guideline "
            "sources there first (see README)."
        )

    print(f"Chunked {len(all_chunks)} passages from {GUIDELINES_DIR}")

    # Dense index
    embeddings = _embed_model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(TEXT_INDEX_PATH))

    metadata = [{"text": c, "source": s} for c, s in zip(all_chunks, all_sources)]
    with open(TEXT_METADATA_PATH, "w") as f:
        json.dump(metadata, f)

    # Sparse (BM25) corpus -- store tokenized chunks for reuse at query time
    tokenized = [_tokenize(c) for c in all_chunks]
    with open(BM25_CORPUS_PATH, "w") as f:
        json.dump(tokenized, f)

    print(f"Text index built -> {TEXT_INDEX_PATH}")


def hybrid_search(query: str, top_k: int = TOP_K_TEXT_PASSAGES) -> "list[dict]":
    """
    Retrieve top-k passages using a weighted fusion of FAISS (dense) and
    BM25 (sparse) rankings. HYBRID_ALPHA in config controls the weighting.
    """
    if not TEXT_INDEX_PATH.exists() or not TEXT_METADATA_PATH.exists():
        # No guideline text has been indexed yet (empty data/guidelines/ when
        # build_indices.py ran). Fail soft so the rest of the pipeline
        # (image retrieval, ACR/EULAR scoring, Gemini generation) still works.
        return []

    with open(TEXT_METADATA_PATH) as f:
        metadata = json.load(f)
    with open(BM25_CORPUS_PATH) as f:
        tokenized_corpus = json.load(f)

    # Dense scores
    index = faiss.read_index(str(TEXT_INDEX_PATH))
    query_vec = np.array([_embed_model.encode(query)], dtype="float32")
    faiss.normalize_L2(query_vec)
    dense_scores, dense_indices = index.search(query_vec, len(metadata))
    dense_score_map = {int(i): float(s) for i, s in zip(dense_indices[0], dense_scores[0])}

    # Sparse scores
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    bm25_score_map = {i: s / max_bm25 for i, s in enumerate(bm25_scores)}  # normalize to 0-1

    # Fuse
    fused = []
    for i in range(len(metadata)):
        dense = dense_score_map.get(i, 0.0)
        sparse = bm25_score_map.get(i, 0.0)
        fused_score = HYBRID_ALPHA * dense + (1 - HYBRID_ALPHA) * sparse
        fused.append((fused_score, i))

    fused.sort(reverse=True)
    top = fused[:top_k]

    return [
        {"text": metadata[i]["text"], "source": metadata[i]["source"], "score": score}
        for score, i in top
    ]
