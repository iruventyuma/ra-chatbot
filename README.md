# RA Chatbot -- RAG-based Rheumatoid Arthritis QA System

Multimodal, multilingual RAG assistant for rheumatoid arthritis. Combines
hand/wrist X-ray retrieval (ViT + FAISS over RAM-H1200) with hybrid
text retrieval (FAISS + BM25 over RA guidelines) and ACR/EULAR-based
symptom questioning, grounded through Gemini. Supports English, Hindi,
and Telugu.

## 1. Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Gemini API key.

## 2. Point the config at your dataset

Open `config.py` and set `RAM_H1200_ROOT` to wherever you downloaded
`RAM-H1200-v1` (the folder that contains `Segmentation/`, `SvdH_Scoring/`,
and `Metadata.xlsx`).

## 3. Add your knowledge base

Drop plain-text (.txt) files into `data/guidelines/` -- ACR/EULAR
guideline text, RA literature excerpts, patient education material, etc.
One source per file is fine; they get automatically chunked.

## 4. Build the indices

```bash
python scripts/build_indices.py
```

This embeds and indexes both the RAM-H1200 images and your guideline
text. Only needs to be re-run when the dataset or knowledge base changes.

## 5. (Optional but recommended) Fine-tune ViT + enable Grad-CAM

Fine-tunes ViT as an RA vs non-RA classifier using the `isRA` column in
`Metadata.xlsx`, and gives you a real accuracy number for your evaluation
section:

```bash
python scripts/finetune_vit.py --epochs 8 --batch_size 16
```

- Reports accuracy on train/val each epoch, and on the held-out **test**
  split at the end -- report the test number in your paper.
- Saves the best checkpoint to `finetuned_vit/`. Once that folder exists,
  the Streamlit app automatically shows the classifier's prediction and a
  **Grad-CAM heatmap** (via `src/gradcam.py`) after each image upload, so
  you can see which regions of the X-ray drove the prediction.
- If accuracy plateaus below 85%: check the class balance printed at the
  start of training (imbalance is the most common cause), try a lower
  learning rate, more epochs, or light augmentation (horizontal flip is
  safe for hand X-rays).
- `Metadata.xlsx` is joined to images via the `Mapped Image Stem` column --
  if your copy of the sheet differs, adjust the join key in
  `scripts/finetune_vit.py::load_split()`.

## 6. Run the app

```bash
streamlit run app.py
```

## Project structure

```
ra_chatbot/
├── app.py                  # Streamlit UI: upload + chat
├── config.py                # paths, model names, retrieval settings
├── requirements.txt
├── .env.example
├── data/
│   ├── guidelines/          # <- put your .txt knowledge base files here
│   └── indices/             # built FAISS/BM25 indices (auto-generated)
├── scripts/
│   └── build_indices.py     # run once to build both indices
└── src/
    ├── embeddings.py         # ViT image embeddings
    ├── image_index.py        # FAISS index over RAM-H1200 + SvdH scores
    ├── text_index.py          # hybrid FAISS + BM25 over guideline text
    ├── acr_eular.py            # 2010 ACR/EULAR scoring calculator
    ├── gemini_client.py         # generation + translation via Gemini
    ├── gradcam.py                # Grad-CAM for the fine-tuned classifier
    └── chatbot.py                  # orchestrates the full pipeline
```

## Changelog (fixes applied to close gaps in the review table)

- **Fixed the actual blocker**: `build_indices.py` used to crash immediately
  because `data/guidelines/` was empty (text index tried to build before the
  image index, and `hybrid_search()` would then also crash on missing index
  files). Added 3 starter guideline files, made the text-index step skip
  gracefully with a warning if the folder is ever empty again, and made
  `hybrid_search()` fail soft (returns `[]`) instead of crashing.
- **Fixed a real scoring bug** in `src/acr_eular.py`: the joint-involvement
  sub-score didn't correctly implement the 2010 ACR/EULAR thresholds
  (e.g. 4-10 small joints should score 3, but the old logic only matched
  `small in (4, 9)`, and the ">10 joints" tier wasn't based on total joint
  count at all). Rewritten to match the published criteria and verified
  against reference cases.
- **Hardened the ViT fine-tuning dataset join** (`scripts/finetune_vit.py`):
  the `Mapped Image Stem` <-> filename join used to fail silently to 0
  matched images on any whitespace/case/extension mismatch between the
  Excel sheet and the actual files. Now normalizes both sides and raises a
  clear error with example values if the join still fails, instead of
  training on an empty dataset.
- **Closed the multimodal integration gap**: the fine-tuned classifier's
  prediction was only ever shown once in the upload UI and never reached
  the RAG/Gemini step. `RAChatSession` now carries it through and includes
  it in the grounded-answer context.
- Updated the default Gemini model in `config.py` to `gemini-2.5-flash`
  to match your confirmed-working setup.

## Notes

## Evaluation scripts (added to close the "research/evaluation" gaps)

Run these after you've built the indices and (optionally) fine-tuned the
classifier -- they're what your report's evaluation section should cite:

- `scripts/verify_patient_splits.py` -- confirms no patient appears in more
  than one of train/val/test (a real leakage check, using
  `Normalized PatientID` from Metadata.xlsx).
- `scripts/evaluate_classifier.py` -- full metrics (accuracy, precision,
  recall, F1, ROC-AUC, confusion matrix) for the fine-tuned classifier,
  plus a majority-class baseline so the accuracy number has something to
  be compared against.
- `scripts/baseline_knn_classifier.py` -- a from-scratch baseline (base,
  non-fine-tuned ViT embeddings + k-NN) so you can report exactly how much
  fine-tuning helped, not just the fine-tuned number in isolation.
- `scripts/evaluate_grounding.py` -- heuristic grounding/hallucination
  check: flags generated-answer sentences with no supporting retrieved
  passage above a similarity threshold.
- `scripts/evaluate_multilingual_consistency.py` -- checks whether the
  same question answered in English/Hindi/Telugu produces semantically
  consistent answers once translated back to English.

All five are heuristic/proxy evaluations appropriate for a capstone
report, not certified clinical-grade metrics -- say so explicitly when you
write them up.

## Not yet integrated (needs your files, not just code)

`src/be_roi.py` is a placeholder only. It raises `NotImplementedError` on
purpose. Your BE ROI model, `BE_labels.csv`, and `BE_Masks` aren't in this
project because I don't have them -- upload them and describe their
format/schema and this becomes real integration code.

- The image retrieval is grounded in real SvdH BE/JSN scores, not just
  visual similarity -- see `src/image_index.py`.
- The ACR/EULAR calculator (`src/acr_eular.py`) is a straightforward
  implementation of the 2010 ACR/EULAR classification criteria (score
  >= 6/10 classifies as RA). It is a *classification suggestion*, not a
  diagnosis -- keep this framing in the UI and in your write-up.
- Multilingual support is a translate-in / translate-out wrapper around
  the English pipeline (see `src/gemini_client.py`). This was the
  fastest path to working multilingual support; a stronger version would
  use multilingual embeddings directly in retrieval -- worth mentioning
  as future work in your paper.
- `RAM-H1200-v1` is licensed CC BY-NC-SA 4.0 -- cite it, keep use
  non-commercial, and check the dataset card for the ethics/consent
  terms before using it in your write-up.
