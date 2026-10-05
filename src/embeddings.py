"""
Extracts a fixed-size embedding vector from an X-ray image using a
pretrained Vision Transformer (ViT). Used both to build the FAISS index
over RAM-H1200 and to embed a newly uploaded image at query time.
"""
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from config import VIT_MODEL_NAME

_device = "cuda" if torch.cuda.is_available() else "cpu"
_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
_model = ViTModel.from_pretrained(VIT_MODEL_NAME).to(_device).eval()


@torch.no_grad()
def embed_image(image_path: str) -> "list[float]":
    """Return a single embedding vector (CLS token) for one X-ray image."""
    image = Image.open(image_path).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt").to(_device)
    outputs = _model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]
    return cls_embedding.squeeze().cpu().numpy().tolist()


@torch.no_grad()
def embed_images_batch(image_paths: "list[str]", batch_size: int = 16) -> "list[list[float]]":
    """Batch version for building the index over many images at once."""
    all_embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = _processor(images=images, return_tensors="pt").to(_device)
        outputs = _model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.extend(cls_embeddings.cpu().numpy().tolist())
    return all_embeddings
