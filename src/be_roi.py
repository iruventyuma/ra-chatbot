"""
INTEGRATION STUB -- not functional yet.

This file is a placeholder for wiring in your own BE (bone erosion) ROI
model, BE_labels.csv, and BE_Masks work. None of that exists in this
project yet -- I don't have your model weights or your labeled data, so
I can't write real code against them.

To actually integrate this:
  1. Upload your BE ROI model file (e.g. a .pt/.pth checkpoint or
     whatever format you trained it in) and tell me its architecture
     (what framework, what input size, what it outputs per joint).
  2. Upload BE_labels.csv and describe its schema (columns, how it keys
     to image filenames or joint crops).
  3. Upload/describe the BE_Masks format (image masks? per-joint crops?
     same size as SvdH ROI crops?).

Once I have those, this file becomes a real module that:
  - loads your BE ROI model
  - runs it on the joint-level ROI crops (see RAM_H1200's
    SvdH_Scoring/SvdH_BE_Scoring/<case>/ folders for the existing 16
    ROI crops per case, which your model likely expects similar input to)
  - merges your model's predictions with (or in place of) the dataset's
    ground-truth SvdH BE scores already used in src/image_index.py
  - surfaces the result in src/chatbot.py's image_context, the same way
    the fine-tuned ViT classifier prediction was wired in

Until then, calling anything in this file will raise NotImplementedError
on purpose, rather than silently doing nothing.
"""


def load_be_roi_model():
    raise NotImplementedError(
        "No BE ROI model has been provided yet. Upload your model file and "
        "describe its architecture/inputs/outputs so this can be implemented."
    )


def predict_be_scores(roi_image_paths: "list[str]") -> dict:
    raise NotImplementedError(
        "No BE ROI model, BE_labels.csv, or BE_Masks have been provided yet. "
        "See the module docstring for what's needed to implement this."
    )
