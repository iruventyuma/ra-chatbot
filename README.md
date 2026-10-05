# 🩺 Rheumatoid Arthritis RAG Assistant

A multilingual, retrieval-augmented chatbot that helps walk a patient through an
ACR/EULAR-style classification suggestion for rheumatoid arthritis (RA) — combining
X-ray similarity retrieval, a fine-tuned image classifier, and grounded LLM answers
sourced from a medical knowledge base.

**This is a classification suggestion tool for a capstone project, not a diagnostic
device.** It is not a substitute for a rheumatologist.

## Features

- 💬 **Chat-first interface** (Streamlit) with a guided, plain-language symptom
  intake — no sidebar forms, no medical jargon assumed. Tap-to-answer quick-reply
  buttons are offered alongside free text, for accessibility.
- 🌐 **Multilingual**: English, Hindi, and Telugu, including full UI localization
  (not just the chat answers).
- 🩻 **X-ray similarity retrieval**: upload a hand/wrist X-ray, retrieve similar
  cases from the RAM-H1200 dataset via a ViT embedding + FAISS index.
- 🧠 **Optional fine-tuned ViT classifier** with Grad-CAM visualization, once
  `scripts/finetune_vit.py` has been run.
- 📊 **ACR/EULAR scoring**: a rules-based implementation of the 2010 ACR/EULAR
  classification criteria, with explicit handling of "unknown/not tested" answers
  (never silently coded as a false negative).
- 📚 **Hybrid RAG**: FAISS (dense) + BM25 (sparse) retrieval over a medical
  guideline knowledge base, fused and passed to Gemini for grounded answer
  generation with source citations.

## Tech stack

Streamlit · Gemini API (`google-genai`) · FAISS · rank-bm25 · sentence-transformers
· HuggingFace Transformers (ViT) · PyTorch · pytorch-grad-cam

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Gemini API key:
