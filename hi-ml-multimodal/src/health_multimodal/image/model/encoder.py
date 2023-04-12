#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from health_multimodal.common.device import get_module_device
from timm.models.layers import trunc_normal_

from .resnet import resnet18, resnet50
from .transformer import VisionTransformerPooler
from .types import ImageEncoderType

DEFAULT_DILATION_VALUES_FOR_RESNET = (False, False, True)
ImageEncoderOutputType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class ImageEncoder(nn.Module):
    """Image encoder trunk module for the ``ImageModel`` class.

    :param img_encoder_type : Type of image encoder model to use, either ``"resnet18_multi_image"`` or
                              ``"resnet50_multi_image"``.
    """

    def __init__(self, img_encoder_type: str):
        super().__init__()
        self.img_encoder_type = img_encoder_type
        self.encoder = self._create_encoder()

    def _create_encoder(self, **kwargs: Any) -> nn.Module:
        if self.img_encoder_type in [ImageEncoderType.RESNET18, ImageEncoderType.RESNET18_MULTI_IMAGE]:
            encoder_class = resnet18
        elif self.img_encoder_type in [ImageEncoderType.RESNET50, ImageEncoderType.RESNET50_MULTI_IMAGE]:
            encoder_class = resnet50
        else:
            supported = ImageEncoderType.get_members(multi_image_encoders_only=False)
            raise NotImplementedError(f"Image encoder type \"{self.img_encoder_type}\" must be in {supported}")

        encoder = encoder_class(pretrained=False, **kwargs)

        return encoder

    def forward(self, current_image: torch.Tensor, return_patch_embeddings: bool = False) -> ImageEncoderOutputType:
        """Get image global and patch embeddings"""

        patch_emb = self.encoder(current_image)
        avg_pooled_emb = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(patch_emb, (1, 1)), 1)
        if return_patch_embeddings:
            return patch_emb, avg_pooled_emb

        return avg_pooled_emb

    def reload_encoder_with_dilation(self, replace_stride_with_dilation: Optional[Sequence[bool]] = None) -> None:
        """Workaround for enabling dilated convolutions after model initialization.

        :param replace_stride_with_dilation: Replace the 2x2 standard convolution stride with a dilated convolution
                                             in each layer in the last three blocks of ResNet architecture.
        """
        if self.img_encoder_type == ImageEncoderType.RESNET18:
            # resnet18 uses BasicBlock implementation, which does not support dilated convolutions.
            raise NotImplementedError("resnet18 does not support dilated convolutions")

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = DEFAULT_DILATION_VALUES_FOR_RESNET

        device = next(self.encoder.parameters()).device
        new_encoder = self._create_encoder(replace_stride_with_dilation=replace_stride_with_dilation).to(device)

        if self.encoder.training:
            new_encoder.train()
        else:
            new_encoder.eval()

        new_encoder.load_state_dict(self.encoder.state_dict())
        self.encoder = new_encoder


class MultiImageEncoder(ImageEncoder):
    """Multi-image encoder trunk module for the ``ImageModel`` class.
    It can be used to encode multiple images into combined latent representation.
    Currently it only supports two input images but can be extended to support more in future.

    :param img_encoder_type: Type of image encoder model to use: either ``"resnet18"`` or ``"resnet50"``.
    """

    def __init__(self, img_encoder_type: str):
        super().__init__(img_encoder_type)

        output_dim = 256  # The aggregate feature dim of the encoder is `2 * output_dim` i.e. [f_static, f_diff]
        grid_shape = (14, 14)  # Spatial dimensions of patch grid.

        backbone_output_feature_dim = get_encoder_output_dim(self.encoder, device=get_module_device(self))

        self.backbone_to_vit = nn.Conv2d(
            in_channels=backbone_output_feature_dim,
            out_channels=output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.vit_pooler = VisionTransformerPooler(input_dim=output_dim, grid_shape=grid_shape)

        # Missing image embedding
        self.missing_previous_emb = nn.Parameter(torch.zeros(1, output_dim, 1, 1))
        trunc_normal_(self.missing_previous_emb, std=0.02)

    def forward(  # type: ignore[override]
        self,
        current_image: torch.Tensor,
        previous_image: Optional[torch.Tensor] = None,
        return_patch_embeddings: bool = False,
    ) -> ImageEncoderOutputType:
        batch_size = current_image.shape[0]

        if previous_image is not None:
            assert current_image.shape == previous_image.shape
            x = torch.cat([current_image, previous_image], dim=0)
            x = super().forward(x, return_patch_embeddings=True)[0]
            x = self.backbone_to_vit(x)
            patch_x, patch_x_previous = x[:batch_size], x[batch_size:]
            diff_x = self.vit_pooler(current_image=patch_x, previous_image=patch_x_previous)
        else:
            x = super().forward(current_image, return_patch_embeddings=True)[0]
            patch_x = self.backbone_to_vit(x)
            B, _, W, H = patch_x.shape
            diff_x = self.missing_previous_emb.repeat(B, 1, W, H)

        patch_fused = torch.cat([patch_x, diff_x], dim=1)
        avg_pooled_emb = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(patch_fused, (1, 1)), 1)

        if return_patch_embeddings:
            return patch_fused, avg_pooled_emb

        return avg_pooled_emb

    def reload_encoder_with_dilation(self, replace_stride_with_dilation: Optional[Sequence[bool]] = None) -> None:
        raise NotImplementedError


@torch.no_grad()
def get_encoder_output_dim(module: torch.nn.Module, device: torch.device) -> int:
    """Calculate the output dimension of an encoder by making a single forward pass.

    :param module: Encoder module.
    :param device: Compute device to use.
    """
    # Target device
    assert isinstance(device, torch.device)

    x = torch.rand((1, 3, 448, 448)).to(device)

    # Extract the number of output feature dimensions
    with restore_training_mode(module):
        module.eval()
        representations = module(x)
    return representations.shape[1]


@contextmanager
def restore_training_mode(module: nn.Module) -> Generator[None, None, None]:
    """Restore the training mode of a module after some operation.

    :param module: PyTorch module.
    """
    training_mode = module.training
    yield
    module.train(mode=training_mode)


def get_encoder_from_type(img_encoder_type: str) -> ImageEncoder:
    """Returns the encoder class for the given encoder type.

    :param img_encoder_type: Encoder type. {RESNET18, RESNET50, RESNET18_MULTI_IMAGE, RESNET50_MULTI_IMAGE}
    """
    if img_encoder_type in ImageEncoderType.get_members(multi_image_encoders_only=True):
        return MultiImageEncoder(img_encoder_type=img_encoder_type)
    else:
        return ImageEncoder(img_encoder_type=img_encoder_type)
