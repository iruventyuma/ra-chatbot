"""
Research-grade evaluation of the fine-tuned ViT classifier, for your
report's evaluation section. Goes beyond finetune_vit.py's train-time
accuracy printout: adds a majority-class baseline (so accuracy has
something meaningful to be compared against), ROC-AUC, and a confusion
matrix.

Requires a checkpoint at config.FINETUNED_VIT_DIR (run
scripts/finetune_vit.py first).

Run:
    python scripts/evaluate_classifier.py
"""

import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
)

from PIL import Image

from config import (
    RAM_H1200_ROOT,
    METADATA_XLSX,
    FINETUNED_VIT_DIR,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def _load_test_split(metadata_df: pd.DataFrame):
    image_dir = RAM_H1200_ROOT / "Segmentation" / "test"

    stem_to_label = {
        _normalize_stem(stem): label
        for stem, label in zip(
            metadata_df["Mapped Image Stem"],
            metadata_df["isRA"]
        )
    }

    paths, labels = [], []

    for p in sorted(image_dir.glob("*.bmp")):

        stem = _normalize_stem(p.stem)

        if stem in stem_to_label:
            paths.append(p)
            labels.append(
                int(bool(stem_to_label[stem]))
            )

    return paths, labels


def majority_class_baseline(labels: list[int]) -> dict:
    """
    The floor any real model must beat:
    always predict the majority class.
    """

    majority = 1 if sum(labels) >= len(labels) / 2 else 0

    preds = [majority] * len(labels)

    acc = accuracy_score(labels, preds)

    return {
        "strategy": (
            f"always predict "
            f"'{ 'RA' if majority else 'non_RA' }'"
        ),
        "accuracy": acc,
    }


def main():

    if not FINETUNED_VIT_DIR.exists():

        print(
            f"No fine-tuned checkpoint found at "
            f"{FINETUNED_VIT_DIR}. "
            "Run scripts/finetune_vit.py first."
        )

        return

    metadata_df = pd.read_excel(METADATA_XLSX)

    paths, labels = _load_test_split(metadata_df)

    if not paths:

        print(
            "0 test images matched metadata -- check the mapping "
            "(see scripts/finetune_vit.py's diagnostics) before "
            "trusting any metric below."
        )

        return

    print(
        f"Test set: {len(paths)} images "
        f"(RA: {sum(labels)}, "
        f"non-RA: {len(labels) - sum(labels)})"
    )

    # ---------------------------------------------------------
    # Majority-class baseline
    # ---------------------------------------------------------

    baseline = majority_class_baseline(labels)

    print(
        f"\nBaseline -- {baseline['strategy']}: "
        f"accuracy = {baseline['accuracy']:.4f}"
    )

    print(
        "(Your fine-tuned model's accuracy is only meaningful "
        "if it clears this baseline by a solid margin. "
        "With class imbalance, matching the baseline means "
        "the model may have learned very little.)"
    )

    # ---------------------------------------------------------
    # Load fine-tuned ViT
    # ---------------------------------------------------------

    processor = ViTImageProcessor.from_pretrained(
        str(FINETUNED_VIT_DIR)
    )

    model = ViTForImageClassification.from_pretrained(
        str(FINETUNED_VIT_DIR)
    ).to(DEVICE).eval()

    # ---------------------------------------------------------
    # Model predictions
    # ---------------------------------------------------------

    all_preds = []
    all_probs_ra = []
    all_labels = []

    with torch.no_grad():

        for path, label in zip(paths, labels):

            image = Image.open(path).convert("RGB")

            inputs = processor(
                images=image,
                return_tensors="pt"
            ).to(DEVICE)

            logits = model(**inputs).logits

            probs = (
                torch.softmax(logits, dim=-1)
                .squeeze()
                .cpu()
                .numpy()
            )

            all_preds.append(
                int(probs.argmax())
            )

            all_probs_ra.append(
                float(probs[1])
            )

            all_labels.append(label)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_preds,
            average="binary",
            zero_division=0
        )
    )

    try:

        auc = roc_auc_score(
            all_labels,
            all_probs_ra
        )

    except ValueError:

        auc = float("nan")

    print(
        f"\nFine-tuned model -- "
        f"accuracy: {acc:.4f}, "
        f"precision: {precision:.4f}, "
        f"recall: {recall:.4f}, "
        f"F1: {f1:.4f}, "
        f"ROC-AUC: {auc:.4f}"
    )

    print(
        f"\nImprovement over baseline: "
        f"{acc - baseline['accuracy']:+.4f} "
        f"accuracy points"
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    print(
        "\nConfusion matrix "
        "(rows=true, cols=predicted, "
        "order=[non_RA, RA]):"
    )

    print(
        confusion_matrix(
            all_labels,
            all_preds,
            labels=[0, 1]
        )
    )

    # ---------------------------------------------------------
    # Full classification report
    # ---------------------------------------------------------

    print("\nFull classification report:")

    print(
        classification_report(
            all_labels,
            all_preds,
            labels=[0, 1],
            target_names=[
                "non_RA",
                "RA"
            ],
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()