"""
Central configuration for the RA chatbot project.
Edit the paths below to match where you put the RAM-H1200 dataset.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# API
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Gemini model used for translation and RAG answer generation
GEMINI_MODEL = "gemini-3.5-flash-lite"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
GUIDELINES_DIR = DATA_DIR / "guidelines"
INDICES_DIR = DATA_DIR / "indices"


# ============================================================
# RAM-H1200 DATASET
# ============================================================

RAM_H1200_ROOT = Path(
    r"D:\rheumatoid arthrisis\dataset"
)

RAM_H1200_BE_SCORES = (
    RAM_H1200_ROOT
    / "SvdH_Scoring"
    / "SvdH_BE_Scoring"
    / "train"
    / "_annotation_be_scores.json"
)

RAM_H1200_JSN_SCORES = (
    RAM_H1200_ROOT
    / "SvdH_Scoring"
    / "SvdH_JSN_Scoring"
    / "train"
    / "_annotation_jsn_scores.json"
)


# ============================================================
# MODELS
# ============================================================

VIT_MODEL_NAME = (
    "google/vit-base-patch16-224-in21k"
)

TEXT_EMBED_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Path to the fine-tuned ViT checkpoint.
# This is created by scripts/finetune_vit.py.

FINETUNED_VIT_DIR = (
    INDICES_DIR.parent / "finetuned_vit"
)

METADATA_XLSX = (
    RAM_H1200_ROOT / "Metadata.xlsx"
)


# ============================================================
# RETRIEVAL
# ============================================================

TOP_K_IMAGE_CASES = 3
TOP_K_TEXT_PASSAGES = 5
HYBRID_ALPHA = 0.5


# ============================================================
# LANGUAGES
# ============================================================

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
}