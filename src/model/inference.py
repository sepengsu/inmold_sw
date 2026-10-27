"""Checkpoint loading and batched binary or three-class U-Net prediction."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .unet import UNet


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[UNet, dict]:
    """Load a supported vision_data U-Net checkpoint for inference."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    architecture = checkpoint.get("architecture")
    if architecture == "grayscale_unet":
        expected = {"label": "Exposure", "mask_rule": "alpha > 0",
                    "normalization": "uint8 / 255"}
        out_channels = 1
        positive_classes = ("Exposure",)
    elif architecture == "grayscale_unet32_3class":
        expected = {"class_names": ("Background", "Exposure", "Weldline"),
                    "base_channels": 32}
        out_channels = 3
        positive_classes = ("Exposure", "Weldline")
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture!r}")
    for key, value in expected.items():
        actual = checkpoint.get(key)
        if key == "class_names":
            actual = tuple(actual) if isinstance(actual, (tuple, list)) else actual
        if actual != value:
            raise ValueError(f"Unsupported checkpoint {key}: {checkpoint.get(key)!r}")
    threshold = checkpoint.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("Checkpoint threshold must be between 0 and 1")
    model = UNet(checkpoint["base_channels"], out_channels=out_channels)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metadata = {key: checkpoint[key] for key in ("epoch", "threshold", "base_channels")}
    metadata.update(architecture=architecture, positive_classes=positive_classes)
    model.to(device).eval()
    return model, metadata


@torch.inference_mode()
def predict_tiles(
    model: UNet, patches: Sequence[np.ndarray], *, batch_size: int = 8,
) -> list[np.ndarray]:
    """Return Exposure or Exposure∪Weldline probabilities per uint8 tile."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not patches:
        return []
    shape = patches[0].shape
    if len(shape) != 2 or not shape[0] or not shape[1]:
        raise ValueError("Tiles must be non-empty 2D arrays")
    for patch in patches:
        if not isinstance(patch, np.ndarray) or patch.dtype != np.uint8 or patch.shape != shape:
            raise ValueError("All tiles must be same-size uint8 grayscale arrays")
    device = next(model.parameters()).device
    model.eval()
    probabilities = []
    for start in range(0, len(patches), batch_size):
        batch = np.stack(patches[start:start + batch_size])[:, None].astype(np.float32) / 255.0
        logits = model(torch.from_numpy(batch).to(device))
        if logits.shape != (len(batch), model.head.out_channels, *shape):
            raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")
        if model.head.out_channels == 1:
            values = logits.float().sigmoid()[:, 0]
        elif model.head.out_channels == 3:
            values = torch.softmax(logits.float(), dim=1)[:, 1:].sum(dim=1)
        else:
            raise ValueError(f"Unsupported model output channels: {model.head.out_channels}")
        values = values.cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError("Model returned non-finite probabilities")
        probabilities.extend(values)
    return probabilities
