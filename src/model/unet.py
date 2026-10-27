"""Grayscale U-Net architecture compatible with vision_data checkpoints."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, outgoing), outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, outgoing), outgoing), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, base_channels: int = 32, activation_checkpointing: bool = False,
                 out_channels: int = 1):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of 8")
        if out_channels < 1:
            raise ValueError("out_channels must be positive")
        widths = [base_channels * 2**i for i in range(5)]
        self.activation_checkpointing = activation_checkpointing
        self.encoders = nn.ModuleList(
            DoubleConv(1 if i == 0 else widths[i - 1], width)
            for i, width in enumerate(widths)
        )
        self.decoders = nn.ModuleList(
            DoubleConv(widths[i + 1] + widths[i], widths[i])
            for i in range(3, -1, -1)
        )
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def apply(block, value):
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                return checkpoint(block, value, use_reentrant=False)
            return block(value)

        skips = []
        for i, encoder in enumerate(self.encoders):
            x = apply(encoder, x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = apply(decoder, torch.cat((skip, x), dim=1))
        return self.head(x)

    def predict_binary(self, x: torch.Tensor, threshold: float = 0.5):
        """Merge Exposure and Weldline from a three-class head into one defect mask."""
        if self.head.out_channels != 3:
            raise ValueError("predict_binary requires a 3-class Background/Exposure/Weldline head")
        logits = self(x)
        probability = torch.softmax(logits, dim=1)[:, 1:].sum(dim=1)
        return probability >= threshold, probability, logits
