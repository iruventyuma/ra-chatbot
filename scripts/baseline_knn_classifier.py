"""
Baseline experiment for your report's comparison section: how much does
fine-tuning actually buy you over the base pretrained ViT?

Baseline: embed every test image with the BASE (not fine-tuned) ViT
(the same model src/embeddings.py uses for retrieval), classify each
test image by a k-NN vote over its nearest neighbors in the TRAIN set
(using the isRA label), and report accuracy.

This uses the same backbone architecture as your fine-tuned model, so
the comparison isolates the effect of fine-tuning specifically, not just
"transformer vs not".

Run this AFTER scripts/finetune_vit.py, then compare its printed test
accuracy to this script's output.

Run:
    python scripts/baseline_knn_classifier.py
"""

import sys
import re
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from config import (
    RAM_H1200_ROOT,
    METADATA_XLSX,
)

from src.embeddings import embed_images_batch


K_NEIGHBORS = 5


def _normalize_stem(value) -> str:
    """
    Normalize image identifiers so metadata names match image filenames.

    Handles:
    - whitespace
    - .bmp extension
    - left/right suffixes such as _L and _R
    - case differences
    """

    s = str(value).strip()

    # Remove .bmp extension if present
    if s.lower().endswith(".bmp"):
        s = s[: -len(".bmp")]

    # Remove left/right suffix such as _L or _R
    s = re.sub(r"_[LR]$", "", s, flags=re.IGNORECASE)

    return s.lower()


def _load_split(
    split_name: str,
    metadata_df: pd.DataFrame
):
    image_dir = (
        RAM_H1200_ROOT
        / "Segmentation"
        / split_name
    )

    stem_to_label = {
        _normalize_stem(stem): label
        for stem, label in zip(
            metadata_df["Mapped Image Stem"],
            metadata_df["isRA"]
        )
    }

    paths = []
    labels = []

    for p in sorted(image_dir.glob("*.bmp")):

        stem = _normalize_stem(p.stem)

        if stem in stem_to_label:

            paths.append(str(p))

            labels.append(
                int(bool(stem_to_label[stem]))
            )

    return paths, labels


def knn_predict(
    train_embs: np.ndarray,
    train_labels: list[int],
    query_embs: np.ndarray,
    k: int
) -> list[int]:

    # Cosine similarity via normalized dot product

    train_norm = (
        train_embs
        / np.linalg.norm(
            train_embs,
            axis=1,
            keepdims=True
        )
    )

    query_norm = (
        query_embs
        / np.linalg.norm(
            query_embs,
            axis=1,
            keepdims=True
        )
    )

    sims = query_norm @ train_norm.T

    preds = []

    for row in sims:

        top_k_idx = np.argsort(row)[-k:]

        top_k_labels = [
            train_labels[i]
            for i in top_k_idx
        ]

        preds.append(
            Counter(
                top_k_labels
            ).most_common(1)[0][0]
        )

    return preds


def main():

    metadata_df = pd.read_excel(
        METADATA_XLSX
    )

    train_paths, train_labels = _load_split(
        "train",
        metadata_df
    )

    test_paths, test_labels = _load_split(
        "test",
        metadata_df
    )

    if not train_paths or not test_paths:

        print(
            "0 images matched in train or test split -- "
            "check the dataset mapping first "
            "(see scripts/finetune_vit.py's diagnostics)."
        )

        return

    print(
        f"Train: {len(train_paths)} images | "
        f"Test: {len(test_paths)} images"
    )

    print(
        f"Train labels: "
        f"RA={sum(train_labels)}, "
        f"non_RA={len(train_labels) - sum(train_labels)}"
    )

    print(
        f"Test labels: "
        f"RA={sum(test_labels)}, "
        f"non_RA={len(test_labels) - sum(test_labels)}"
    )

    print(
        "\nEmbedding train set with "
        "base (non-fine-tuned) ViT..."
    )

    train_embs = np.array(
        embed_images_batch(train_paths),
        dtype="float32"
    )

    print(
        "Embedding test set..."
    )

    test_embs = np.array(
        embed_images_batch(test_paths),
        dtype="float32"
    )

    preds = knn_predict(
        train_embs,
        train_labels,
        test_embs,
        K_NEIGHBORS
    )

    acc = accuracy_score(
        test_labels,
        preds
    )

    print(
        f"\nBaseline (base ViT embeddings + "
        f"{K_NEIGHBORS}-NN): "
        f"test accuracy = {acc:.4f}"
    )

    print(
        "\nFull classification report:"
    )

    print(
        classification_report(
            test_labels,
            preds,
            labels=[0, 1],
            target_names=[
                "non_RA",
                "RA"
            ],
            zero_division=0
        )
    )

    print(
        "Compare this to "
        "scripts/evaluate_classifier.py's "
        "fine-tuned model accuracy."
    )

    print(
        "The difference between the two is the "
        "measured effect of fine-tuning, but remember "
        "that the current dataset has patient overlap "
        "between splits and severe class imbalance."
    )


if __name__ == "__main__":
    main()