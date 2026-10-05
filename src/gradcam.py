"""
Grad-CAM for the fine-tuned ViT classifier.

This module:
1. Loads the fine-tuned Hugging Face ViT classifier.
2. Predicts RA / non-RA with confidence.
3. Generates a Grad-CAM heatmap showing where the ViT is focusing.

The Hugging Face model returns a ModelOutput object, while
pytorch-grad-cam expects the model forward() to return a plain tensor.
Therefore, a small wrapper is used specifically for Grad-CAM.
"""

import numpy as np
import torch
import cv2

from PIL import Image

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import FINETUNED_VIT_DIR


# ============================================================
# DEVICE
# ============================================================

_device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Grad-CAM device: {_device}")


# ============================================================
# LOAD PROCESSOR AND FINE-TUNED MODEL
# ============================================================

_processor = ViTImageProcessor.from_pretrained(
    str(FINETUNED_VIT_DIR)
)

_model = ViTForImageClassification.from_pretrained(
    str(FINETUNED_VIT_DIR)
).to(_device).eval()


print("Fine-tuned ViT model loaded successfully.")


# ============================================================
# GRAD-CAM MODEL WRAPPER
# ============================================================

class _LogitsOnlyWrapper(torch.nn.Module):
    """
    pytorch_grad_cam expects model.forward() to return
    a plain tensor.

    Hugging Face ViTForImageClassification normally returns
    a SequenceClassifierOutput / ModelOutput object.

    This wrapper extracts only .logits for Grad-CAM.
    """

    def __init__(self, hf_model):
        super().__init__()

        self.hf_model = hf_model
        self.config = hf_model.config

    def forward(self, pixel_values):
        return self.hf_model(
            pixel_values=pixel_values
        ).logits


# Wrapper is used ONLY for Grad-CAM.
# Normal prediction continues to use _model directly.

_gradcam_model = _LogitsOnlyWrapper(
    _model
).to(_device).eval()


# ============================================================
# FIND A SUITABLE ViT TARGET LAYER
# ============================================================

def _resolve_target_layer(vit_model):
    """
    Find a suitable LayerNorm / transformer layer for Grad-CAM.

    Different versions of Transformers can expose the ViT
    architecture using slightly different attribute names.
    """

    candidates = [
        (
            "encoder.layer[-1].layernorm_before",
            lambda m: m.encoder.layer[-1].layernorm_before,
        ),
        (
            "encoder.layers[-1].layernorm_before",
            lambda m: m.encoder.layers[-1].layernorm_before,
        ),
        (
            "layer[-1].layernorm_before",
            lambda m: m.layer[-1].layernorm_before,
        ),
        (
            "layers[-1].layernorm_before",
            lambda m: m.layers[-1].layernorm_before,
        ),
        (
            "encoder.layer[-1].layernorm1",
            lambda m: m.encoder.layer[-1].layernorm1,
        ),
        (
            "layers[-1].norm1",
            lambda m: m.layers[-1].norm1,
        ),
        (
            "blocks[-1].norm1",
            lambda m: m.blocks[-1].norm1,
        ),
    ]

    for label, get_layer in candidates:

        try:
            layer = get_layer(vit_model)

            print(
                f"Grad-CAM target layer resolved via: {label}"
            )

            return layer

        except AttributeError:
            continue

    # --------------------------------------------------------
    # Fallback: search for the last LayerNorm
    # --------------------------------------------------------

    layernorms = [
        module
        for module in vit_model.modules()
        if module.__class__.__name__ == "LayerNorm"
    ]

    if layernorms:

        print(
            "Grad-CAM target layer resolved via fallback: "
            "last LayerNorm found in model."
        )

        return layernorms[-1]

    raise AttributeError(
        "Could not find any LayerNorm module in the fine-tuned "
        "ViT to target for Grad-CAM."
    )


# The target layer comes from the ORIGINAL Hugging Face model.
# The Grad-CAM wrapper references this same model.

_target_layers = [
    _resolve_target_layer(_model.vit)
]


# ============================================================
# ViT TOKEN -> IMAGE GRID TRANSFORMATION
# ============================================================

def _reshape_transform(
    tensor,
    height=14,
    width=14,
):
    """
    Convert ViT patch tokens into a spatial feature map.

    ViT-base with:
        image size = 224 x 224
        patch size = 16 x 16

    gives:

        224 / 16 = 14

    Therefore there are:

        14 x 14 = 196

    image patch tokens.

    The first token is the CLS token, so it is removed.
    """

    # Remove CLS token
    result = tensor[:, 1:, :]

    # Convert tokens into 14 x 14 spatial grid
    result = result.reshape(
        tensor.size(0),
        height,
        width,
        tensor.size(2),
    )

    # Convert:
    # [batch, height, width, channels]
    #
    # into:
    # [batch, channels, height, width]

    result = result.permute(
        0,
        3,
        1,
        2,
    )

    return result


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    image_path: str,
    output_path: str,
    target_class: int = 1,
) -> str:
    """
    Generate a Grad-CAM visualization for an X-ray.

    Parameters
    ----------
    image_path:
        Path to the input X-ray.

    output_path:
        Path where the Grad-CAM image will be saved.

    target_class:
        Class index to explain.

        Default:
            1 = RA

    Returns
    -------
    str
        Path to the generated Grad-CAM image.
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Prepare image for visualization
    # --------------------------------------------------------

    rgb_img = np.array(
        image.resize((224, 224))
    ).astype(np.float32) / 255.0


    # --------------------------------------------------------
    # Prepare ViT input
    # --------------------------------------------------------

    inputs = _processor(
        images=image,
        return_tensors="pt",
    )

    input_tensor = inputs[
        "pixel_values"
    ].to(_device)


    # --------------------------------------------------------
    # Create Grad-CAM
    # --------------------------------------------------------

    cam = GradCAM(
        model=_gradcam_model,
        target_layers=_target_layers,
        reshape_transform=_reshape_transform,
    )


    # --------------------------------------------------------
    # Select target class
    # --------------------------------------------------------

    targets = [
        ClassifierOutputTarget(target_class)
    ]


    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
    )[0]


    # --------------------------------------------------------
    # Overlay heatmap on X-ray
    # --------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True,
    )


    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR,
        ),
    )


    print(
        f"Grad-CAM saved to: {output_path}"
    )

    return output_path


# ============================================================
# PREDICT WITH CONFIDENCE
# ============================================================

def predict_with_confidence(
    image_path: str,
) -> dict:
    """
    Predict the class of an X-ray and return probabilities.

    Returns
    -------
    dict
        {
            "predicted_label": "...",
            "confidence": 0.XX,
            "probabilities": {
                "non_RA": 0.XX,
                "RA": 0.XX
            }
        }
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    inputs = _processor(
        images=image,
        return_tensors="pt",
    )

    inputs = inputs.to(_device)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = _model(
            **inputs
        ).logits

        probs = torch.softmax(
            logits,
            dim=-1,
        ).squeeze().cpu().numpy()


    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    pred_idx = int(
        probs.argmax()
    )


    predicted_label = _model.config.id2label[
        pred_idx
    ]


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {
        "predicted_label": predicted_label,

        "confidence": float(
            probs[pred_idx]
        ),

        "probabilities": {
            _model.config.id2label[i]: float(probs[i])
            for i in range(len(probs))
        },
    }