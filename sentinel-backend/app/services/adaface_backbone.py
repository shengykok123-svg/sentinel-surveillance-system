"""
PyTorch neural network architectures for AdaFace face recognition models.

This module supports the legacy IR-SE-50 backbone and the released IR101
WebFace12M backbone. Both accept 112x112 RGB face images and produce
512-dimensional L2-normalised embeddings suitable for cosine similarity.

Architecture overview:
    Input (3x112x112)
    -> input_layer (Conv2d + BN + PReLU)
    -> 4 stages of IRSEBlocks (with increasing channel depth: 64->128->256->512)
    -> output_layer (BN + Dropout + Flatten + FC + BN)
    -> L2 normalisation
    -> 512-d unit embedding + norm scalar
"""

from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Linear, MaxPool2d, Module, PReLU, ReLU, Sequential, Sigmoid

# Flattens a multi-dimensional tensor into a 2-D tensor (batch_size, -1), used in the output layer to reshape the spatial feature map into a flat vector before the fully-connected layer.
class Flatten(Module):
    
# Forward pass reshapes the input tensor while preserving the batch dimension.
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor, preserving the batch dimension.

        Args:
            inputs: Tensor of shape (batch_size, C, H, W).

        Returns:
            Tensor of shape (batch_size, C * H * W).
        """
        return inputs.view(inputs.size(0), -1)

# Squeeze-and-Excitation (SE) block definition, applying channel-wise attention to recalibrate feature responses.
class SEModule(Module):

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        # Two-layer bottleneck: reduce channels by `reduction`, then expand back
        self.fc = nn.Sequential(
            Linear(channels, channels // reduction, bias=False),
            ReLU(inplace=True),
            Linear(channels // reduction, channels, bias=False),
            Sigmoid(),
        )

# Forward pass applies SE attention scaling to the input feature map.
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        
        # Global average pooling: (B, C, H, W) -> (B, C)
        scale = inputs.mean(dim=(2, 3))
        # Bottleneck FC to compute per-channel scale factors, then reshape for broadcasting
        scale = self.fc(scale).view(inputs.size(0), inputs.size(1), 1, 1)
        # Element-wise channel scaling
        return inputs * scale

# IR-SE block definition with optional downsampling and channel expansion in the skip connection.
class IRSEBlock(Module):

    def __init__(self, in_channels: int, depth: int, stride: int, use_se: bool = True) -> None:
        super().__init__()
        # Pre-activation batch normalisation on input
        self.bn1 = BatchNorm2d(in_channels)
        # First 3x3 convolution (no spatial downsampling here)
        self.conv1 = Conv2d(in_channels, depth, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(depth)
        # PReLU activation with learnable slope per channel
        self.prelu = PReLU(depth)
        # Second 3x3 convolution (applies stride for spatial downsampling)
        self.conv2 = Conv2d(depth, depth, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = BatchNorm2d(depth)
        # The official IR101 WebFace12M checkpoint uses plain IR blocks, while
        # the legacy IR-SE-50 model includes this channel-attention layer.
        self.se = SEModule(depth, reduction=16) if use_se else None

        # Skip connection: learned projection if dimensions change, identity otherwise
        if stride != 1 or in_channels != depth:
            self.down = Sequential(
                Conv2d(in_channels, depth, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(depth),
            )
        else:
            # Identity shortcut via 1x1 max-pool (effectively a no-op)
            self.down = MaxPool2d(1, stride)

# Forward pass with residual connection
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    
        # Compute the shortcut path (may downsample/project)
        shortcut = self.down(inputs)
        # Main path: BN -> Conv -> BN -> PReLU -> Conv -> BN -> SE
        outputs = self.bn1(inputs)
        outputs = self.conv1(outputs)
        outputs = self.bn2(outputs)
        outputs = self.prelu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn3(outputs)
        if self.se is not None:
            outputs = self.se(outputs)
        # Residual addition
        return outputs + shortcut


# Named tuple defining the specification for a single IRSEBlock.
# Fields: in_channels (int), depth (int), stride (int).
BlockSpec = namedtuple("BlockSpec", ["in_channels", "depth", "stride"])

# Helper function to generate a list of BlockSpec entries for a stage of the network.
def _make_stage(in_channels: int, depth: int, num_units: int, stride: int = 2) -> list[BlockSpec]:

    # First block handles channel expansion and spatial downsampling
    # Remaining blocks maintain depth and spatial dimensions (stride=1)
    return [BlockSpec(in_channels, depth, stride)] + [BlockSpec(depth, depth, 1) for _ in range(num_units - 1)]

# Full IR-SE-50 backbone definition, consisting of the input layer, four stages of IRSEBlocks, and the output layer that produces 512-d embeddings.
class AdaFaceBackbone(Module):

    def __init__(self, layers: tuple[int, int, int, int], *, use_se: bool = True) -> None:
        super().__init__()
        # Initial convolution: 3 input RGB channels -> 64 feature channels
        self.input_layer = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        # Four stages with progressively increasing channel depth and spatial downsampling
        self.layer1 = self._build_stage(_make_stage(64, 64, layers[0], stride=2), use_se=use_se)
        self.layer2 = self._build_stage(_make_stage(64, 128, layers[1], stride=2), use_se=use_se)
        self.layer3 = self._build_stage(_make_stage(128, 256, layers[2], stride=2), use_se=use_se)
        self.layer4 = self._build_stage(_make_stage(256, 512, layers[3], stride=2), use_se=use_se)

        # Output layer: BN -> Dropout -> Flatten -> FC to 512-d -> BN
        # The FC input dimension is 512*7*7 because after 4 stages of stride-2
        # downsampling, a 112x112 input becomes 7x7 spatially.
        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(0.4),
            Flatten(),
            Linear(512 * 7 * 7, 512),
            BatchNorm1d(512),
        )

    @staticmethod
    # Helper method to build a sequential stage of IRSEBlocks from a list of BlockSpec entries.
    def _build_stage(specs: list[BlockSpec], *, use_se: bool) -> Sequential:
        return Sequential(
            *[IRSEBlock(spec.in_channels, spec.depth, spec.stride, use_se=use_se) for spec in specs]
        )

# Forward pass through the backbone, producing L2-normalised embeddings and their pre-normalisation norms.
    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.input_layer(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.output_layer(outputs)
        # Compute L2 norm per sample, clamped to avoid division by zero
        norms = torch.norm(outputs, 2, 1, keepdim=True).clamp(min=1e-8)
        # Normalise to unit-length embeddings for cosine similarity comparisons
        embeddings = torch.div(outputs, norms)
        return embeddings, norms


class IRBlock(Module):
    """Plain IR residual block used by the official AdaFace IR101 model."""

    def __init__(self, in_channels: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channels == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channels, depth, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(depth),
            )
        # Keep this ordering and name aligned with the released checkpoint's
        # ``body.<n>.res_layer`` keys.
        self.res_layer = Sequential(
            BatchNorm2d(in_channels),
            Conv2d(in_channels, depth, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.res_layer(inputs) + self.shortcut_layer(inputs)


class AdaFaceIRBackbone(Module):
    """IR backbone whose module layout matches AdaFace's IR101 release."""

    def __init__(self, layers: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.input_layer = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        stages = (
            _make_stage(64, 64, layers[0], stride=2),
            _make_stage(64, 128, layers[1], stride=2),
            _make_stage(128, 256, layers[2], stride=2),
            _make_stage(256, 512, layers[3], stride=2),
        )
        self.body = Sequential(
            *[IRBlock(spec.in_channels, spec.depth, spec.stride) for stage in stages for spec in stage]
        )
        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(0.4),
            Flatten(),
            Linear(512 * 7 * 7, 512),
            # AdaFace's released IR101 checkpoint has an affine-free final
            # batch-normalisation layer.
            BatchNorm1d(512, affine=False),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.output_layer(self.body(self.input_layer(inputs)))
        norms = torch.norm(outputs, 2, 1, keepdim=True).clamp(min=1e-8)
        return torch.div(outputs, norms), norms

# Factory function to instantiate an AdaFaceBackbone by architecture name.
def build_adaface_backbone(architecture: str) -> Module:
    arch = architecture.lower()
    if arch in {"ir_se_50", "ir_50"}:
        return AdaFaceBackbone((3, 4, 14, 3))
    if arch in {"ir_101", "ir101"}:
        return AdaFaceIRBackbone((3, 13, 30, 3))
    raise ValueError(f"Unsupported AdaFace architecture '{architecture}'")
