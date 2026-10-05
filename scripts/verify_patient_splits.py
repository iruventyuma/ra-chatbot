"""
Verifies that RAM-H1200's train/val/test splits are patient-disjoint --
i.e. no single patient's studies leak across more than one split. This is
a standard data-leakage check that should run before reporting any
test-set metric: if the same patient appears in both train and test, the
model can partly "memorize" that patient's anatomy, inflating accuracy
in a way that won't hold up on truly unseen patients.

Run:
    python scripts/verify_patient_splits.py
"""

import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from config import RAM_H1200_ROOT, METADATA_XLSX


SPLIT_DIRS = {
    "train": RAM_H1200_ROOT / "Segmentation" / "train",
    "val": RAM_H1200_ROOT / "Segmentation" / "val",
    "test": RAM_H1200_ROOT / "Segmentation" / "test",
}


def _normalize_stem(value) -> str:
    s = str(value).strip()

    # Remove .bmp extension if present
    if s.lower().endswith(".bmp"):
        s = s[: -len(".bmp")]

    # Remove left/right suffix such as _L or _R
    s = re.sub(r"_[LR]$", "", s, flags=re.IGNORECASE)

    return s.lower()


def main():
    metadata_df = pd.read_excel(METADATA_XLSX)

    required_cols = {"Mapped Image Stem", "Normalized PatientID"}
    missing = required_cols - set(metadata_df.columns)

    if missing:
        raise ValueError(
            f"Metadata.xlsx is missing expected column(s): {missing}. "
            f"Columns found: {list(metadata_df.columns)}"
        )

    stem_to_patient = {
        _normalize_stem(stem): patient
        for stem, patient in zip(
            metadata_df["Mapped Image Stem"],
            metadata_df["Normalized PatientID"]
        )
    }

    split_patients = {}

    for split_name, split_dir in SPLIT_DIRS.items():

        if not split_dir.exists():
            print(
                f"WARNING: {split_dir} does not exist "
                f"-- skipping '{split_name}'"
            )
            continue

        stems = [
            _normalize_stem(p.stem)
            for p in split_dir.glob("*.bmp")
        ]

        matched = [
            s for s in stems
            if s in stem_to_patient
        ]

        patients = {
            stem_to_patient[s]
            for s in matched
        }

        split_patients[split_name] = patients

        print(
            f"{split_name}: {len(stems)} images, "
            f"{len(matched)} matched to metadata, "
            f"{len(patients)} unique patients"
        )

        if len(matched) < len(stems):
            print(
                f"  WARNING: {len(stems) - len(matched)} image(s) "
                f"had no metadata match -- "
                f"re-run scripts/finetune_vit.py's diagnostics "
                f"if this number is large."
            )

    names = list(split_patients.keys())
    leak_found = False

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            a, b = names[i], names[j]

            overlap = split_patients[a] & split_patients[b]

            if overlap:
                leak_found = True

                sample = list(overlap)[:5]

                print(
                    f"\nLEAK: {len(overlap)} patient(s) appear "
                    f"in both '{a}' and '{b}': {sample}"
                    f"{'...' if len(overlap) > 5 else ''}"
                )

    print()

    if not leak_found:
        print(
            "No patient overlap found between splits. "
            "The provided train/val/test splits are patient-disjoint "
            "-- safe to report test-set metrics as a genuine "
            "generalization estimate in your write-up."
        )
    else:
        print(
            "Patient overlap detected. Do NOT report test-set accuracy "
            "as an unqualified generalization estimate -- either "
            "re-split by patient ID yourself, or explicitly disclose "
            "this leakage as a limitation."
        )


if __name__ == "__main__":
    main()