"""
Fine-tune ViT as a binary classifier: RA vs non-RA.

Ground truth:
    Metadata.xlsx -> isRA

Dataset:
    RAM-H1200 -> Segmentation/{train,val,test}

Run:
    python scripts/finetune_vit.py --epochs 8 --batch_size 16

The model is evaluated on the held-out TEST split.
"""

import sys
import re
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from tqdm import tqdm

from config import (
    RAM_H1200_ROOT,
    METADATA_XLSX,
    VIT_MODEL_NAME,
    FINETUNED_VIT_DIR,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DATASET
# ============================================================

class RAXrayDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        processor
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(
            self.image_paths[idx]
        ).convert("RGB")

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        )["pixel_values"].squeeze(0)

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return pixel_values, label


# ============================================================
# NORMALIZE IMAGE / METADATA STEM
# ============================================================

def _normalize_stem(value) -> str:
    """
    Normalize filenames so Metadata.xlsx can be matched
    with the actual RAM-H1200 image filenames.

    IMPORTANT:
    Metadata.xlsx contains:

        JP_HMCRD_P0001_20210203_6791

    Actual image files contain:

        JP_HMCRD_P0001_20210203_6791_L.bmp
        JP_HMCRD_P0001_20210203_6791_R.bmp

    One metadata row represents both left and right hand images.

    Therefore the final _L / _R suffix is removed.
    """

    s = str(value).strip()

    # Remove .bmp if present
    if s.lower().endswith(".bmp"):
        s = s[:-4]

    # Remove final _L or _R
    s = re.sub(
        r"_[LR]$",
        "",
        s,
        flags=re.IGNORECASE
    )

    # Lowercase for reliable matching
    return s.lower()


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(
    split_name: str,
    metadata_df: pd.DataFrame
):

    """
    Match Segmentation/{split}/ images with Metadata.xlsx.

    Returns:
        image_paths
        labels
    """

    image_dir = (
        RAM_H1200_ROOT
        / "Segmentation"
        / split_name
    )

    print()
    print("=" * 60)
    print(f"PROCESSING SPLIT: {split_name.upper()}")
    print("=" * 60)

    if not image_dir.exists():

        raise FileNotFoundError(
            f"Image directory does not exist:\n{image_dir}"
        )

    # --------------------------------------------------------
    # Check required metadata columns
    # --------------------------------------------------------

    required_columns = [
        "Mapped Image Stem",
        "isRA",
        "Normalized PatientID",
    ]

    for column in required_columns:

        if column not in metadata_df.columns:

            raise ValueError(
                f"Metadata.xlsx is missing required column: "
                f"'{column}'"
            )

    # --------------------------------------------------------
    # Build metadata lookup
    # --------------------------------------------------------

    stem_to_label = {}

    for stem, label in zip(
        metadata_df["Mapped Image Stem"],
        metadata_df["isRA"]
    ):

        normalized_stem = _normalize_stem(stem)

        if normalized_stem:

            stem_to_label[
                normalized_stem
            ] = label

    print(
        f"Metadata entries available: "
        f"{len(stem_to_label)}"
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_files = sorted(
        image_dir.glob("*.bmp")
    )

    print(
        f"BMP images found: "
        f"{len(image_files)}"
    )

    if len(image_files) == 0:

        raise ValueError(
            f"No BMP images found in:\n{image_dir}"
        )

    # --------------------------------------------------------
    # Match images
    # --------------------------------------------------------

    image_paths = []
    labels = []
    unmatched = []

    for img_path in image_files:

        normalized_stem = _normalize_stem(
            img_path.stem
        )

        if normalized_stem in stem_to_label:

            label = stem_to_label[
                normalized_stem
            ]

            image_paths.append(
                str(img_path)
            )

            labels.append(
                int(bool(label))
            )

        else:

            unmatched.append(
                img_path.name
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    matched_count = len(image_paths)
    total_count = len(image_files)

    print(
        f"Matched: "
        f"{matched_count}/{total_count}"
    )

    print(
        f"Unmatched: "
        f"{len(unmatched)}/{total_count}"
    )

    # --------------------------------------------------------
    # Fail loudly if ZERO matches
    # --------------------------------------------------------

    if matched_count == 0:

        example_file = (
            _normalize_stem(
                image_files[0].stem
            )
        )

        metadata_examples = list(
            stem_to_label.keys()
        )

        example_metadata = (
            metadata_examples[0]
            if metadata_examples
            else "NO METADATA"
        )

        raise ValueError(
            "\nZERO IMAGES MATCHED.\n\n"
            f"Example normalized image stem:\n"
            f"  {example_file}\n\n"
            f"Example normalized metadata stem:\n"
            f"  {example_metadata}\n\n"
            "Check RAM_H1200_ROOT and Metadata.xlsx."
        )

    # --------------------------------------------------------
    # Show unmatched examples
    # --------------------------------------------------------

    if unmatched:

        print(
            "\nWARNING: Some images could not "
            "be matched."
        )

        print(
            "Examples:"
        )

        for name in unmatched[:5]:

            print(
                f"  {name}"
            )

    # --------------------------------------------------------
    # Class balance
    # --------------------------------------------------------

    ra_count = sum(labels)

    non_ra_count = (
        len(labels) - ra_count
    )

    print(
        f"\nClass balance:"
    )

    print(
        f"  RA:     {ra_count}"
    )

    print(
        f"  non-RA: {non_ra_count}"
    )

    print(
        f"  Total:  {len(labels)}"
    )

    return image_paths, labels


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    dataloader
):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for (
            pixel_values,
            labels
        ) in dataloader:

            pixel_values = (
                pixel_values.to(DEVICE)
            )

            logits = model(
                pixel_values=pixel_values
            ).logits

            preds = (
                logits
                .argmax(dim=-1)
                .cpu()
                .numpy()
            )

            all_preds.extend(
                preds
            )

            all_labels.extend(
                labels.numpy()
            )

    acc = accuracy_score(
        all_labels,
        all_preds
    )

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[
            "non_RA",
            "RA"
        ],
        zero_division=0
    )

    return acc, report


# ============================================================
# MAIN
# ============================================================

def main(
    epochs: int,
    batch_size: int,
    lr: float
):

    print()
    print("=" * 60)
    print("       RAM-H1200 RA ViT CLASSIFIER")
    print("=" * 60)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Dataset root:\n"
        f"{RAM_H1200_ROOT}"
    )

    print(
        f"\nMetadata:\n"
        f"{METADATA_XLSX}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    print(
        "\nLoading Metadata.xlsx..."
    )

    metadata_df = pd.read_excel(
        METADATA_XLSX
    )

    print(
        f"Metadata rows: "
        f"{len(metadata_df)}"
    )

    print(
        "Metadata columns:"
    )

    print(
        metadata_df.columns.tolist()
    )

    # --------------------------------------------------------
    # Overall metadata class balance
    # --------------------------------------------------------

    print(
        "\nOverall metadata class balance:"
    )

    print(
        metadata_df["isRA"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Load train/val/test
    # --------------------------------------------------------

    train_paths, train_labels = (
        load_split(
            "train",
            metadata_df
        )
    )

    val_paths, val_labels = (
        load_split(
            "val",
            metadata_df
        )
    )

    test_paths, test_labels = (
        load_split(
            "test",
            metadata_df
        )
    )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Train:      {len(train_paths)}"
    )

    print(
        f"Validation: {len(val_paths)}"
    )

    print(
        f"Test:       {len(test_paths)}"
    )

    # --------------------------------------------------------
    # Load processor
    # --------------------------------------------------------

    print()
    print(
        "Loading pretrained ViT processor..."
    )

    processor = (
        ViTImageProcessor.from_pretrained(
            VIT_MODEL_NAME
        )
    )

    # --------------------------------------------------------
    # Load ViT
    # --------------------------------------------------------

    print(
        "Loading pretrained ViT model..."
    )

    model = (
        ViTForImageClassification
        .from_pretrained(
            VIT_MODEL_NAME,
            num_labels=2,
            id2label={
                0: "non_RA",
                1: "RA"
            },
            label2id={
                "non_RA": 0,
                "RA": 1
            },
            ignore_mismatched_sizes=True
        )
        .to(DEVICE)
    )

    print(
        "ViT model loaded successfully!"
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_ds = RAXrayDataset(
        train_paths,
        train_labels,
        processor
    )

    val_ds = RAXrayDataset(
        val_paths,
        val_labels,
        processor
    )

    test_ds = RAXrayDataset(
        test_paths,
        test_labels,
        processor
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=lr
    )

    total_steps = (
        len(train_loader) * epochs
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
    )

    loss_fn = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_acc = 0.0

    FINETUNED_VIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    print(
        f"Epochs: {epochs}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Learning rate: {lr}"
    )

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{epochs}"
            )
        )

        for (
            pixel_values,
            labels
        ) in progress:

            pixel_values = (
                pixel_values.to(DEVICE)
            )

            labels = (
                labels.to(DEVICE)
            )

            optimizer.zero_grad()

            outputs = model(
                pixel_values=pixel_values
            )

            logits = outputs.logits

            loss = loss_fn(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            scheduler.step()

            running_loss += (
                loss.item()
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_acc, val_report = evaluate(
            model,
            val_loader
        )

        average_loss = (
            running_loss
            / len(train_loader)
        )

        print()
        print(
            f"Epoch {epoch + 1}:"
        )

        print(
            f"  Train loss: "
            f"{average_loss:.4f}"
        )

        print(
            f"  Validation accuracy: "
            f"{val_acc:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            model.save_pretrained(
                FINETUNED_VIT_DIR
            )

            processor.save_pretrained(
                FINETUNED_VIT_DIR
            )

            print(
                "  -> Best model saved!"
            )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    test_acc, test_report = evaluate(
        model,
        test_loader
    )

    print()
    print(
        f"Test accuracy: "
        f"{test_acc:.4f}"
    )

    print()
    print(
        "Classification report:"
    )

    print(
        test_report
    )

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.4f}"
    )

    print(
        f"Final test accuracy: "
        f"{test_acc:.4f}"
    )

    print(
        f"\nCheckpoint saved at:"
    )

    print(
        FINETUNED_VIT_DIR
    )

    print("=" * 60)


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=8
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5
    )

    args = parser.parse_args()

    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )