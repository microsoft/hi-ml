#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from torch import Tensor as T

from .modules import MLP, MultiTaskModel


@dataclass
class ImageModelOutput():
    img_embedding: torch.Tensor
    patch_embedding: torch.Tensor
    projected_global_embedding: torch.Tensor
    class_logits: torch.Tensor
    projected_patch_embeddings: torch.Tensor


class ImageModel(nn.Module):
    """Image encoder module"""

    def __init__(self,
                 img_model_type: str,
                 joint_feature_size: int,
                 freeze_encoder: bool = False,
                 pretrained_model_path: Optional[str] = None,
                 **downstream_classifier_kwargs: Any):
        super().__init__()

        # Initiate encoder, projector, and classifier
        self.encoder = ImageEncoder(img_model_type)
        self.feature_size = get_encoder_output_dim(self.encoder)
        self.projector = MLP(input_dim=self.feature_size, output_dim=joint_feature_size,
                             hidden_dim=joint_feature_size, use_1x1_convs=True)
        self.downstream_classifier_kwargs = downstream_classifier_kwargs
        self.classifier = self.create_downstream_classifier() if downstream_classifier_kwargs else None

        # Initialise the mode of modules
        self.freeze_encoder = freeze_encoder
        self.train()

        if pretrained_model_path is not None:
            assert isinstance(pretrained_model_path, str), f"Expected a string, got {type(pretrained_model_path)}"
            self.load_state_dict(torch.load(pretrained_model_path))

    def train(self, mode: bool = True) -> Any:
        """Switch the model between training and evaluation modes."""
        super().train(mode=mode)
        if self.freeze_encoder:
            self.encoder.train(mode=False)
            self.projector.train(mode=False)
        return self

    def forward(self, x: torch.Tensor, *args, **kwargs) -> ImageModelOutput:    # type: ignore
        """
        :param x: Input image tensor
        """
        with torch.set_grad_enabled(not self.freeze_encoder):
            patch_x, pooled_x = self.encoder(x, return_patch_embeddings=True)
            projected_patch_embeddings = self.projector(patch_x)
            projected_global_embedding = torch.mean(projected_patch_embeddings, dim=(2, 3))

        logits = self.classifier(pooled_x) if self.classifier else None
        return ImageModelOutput(img_embedding=pooled_x,
                                patch_embedding=patch_x,
                                class_logits=logits,
                                projected_patch_embeddings=projected_patch_embeddings,
                                projected_global_embedding=projected_global_embedding)

    def create_downstream_classifier(self, **kwargs: Any) -> MultiTaskModel:
        """
        Creates the classification module for the downstream task
        """
        downstream_classifier_kwargs = kwargs if kwargs else self.downstream_classifier_kwargs
        return MultiTaskModel(self.feature_size, **downstream_classifier_kwargs)

    @torch.no_grad()
    def get_patchwise_projected_embeddings(self, input_img: torch.Tensor, normalize: bool) -> torch.Tensor:
        """
        Get patchwise projected embeddings from the CNN model.

        :param input_img: input tensor image [B, C, H, W]
        :param normalize: If set to True, the embeddings are l2-normalized.
        :returns projected_embeddings: tensor of embeddings in shape [batch, n_patches_h, n_patches_w, feature_size]
        """
        assert self.training is False, "This function is only implemented for evaluation mode"
        outputs = self.forward(input_img)
        projected_embeddings = outputs.projected_patch_embeddings.detach()  # type: ignore
        if normalize:
            projected_embeddings = F.normalize(projected_embeddings, dim=1)
        projected_embeddings = projected_embeddings.permute([0, 2, 3, 1])  # B D H W -> B H W D (D: Features)
        return projected_embeddings


class ImageEncoder(nn.Module):
    """
    Image encoder trunk module for the ImageModel class.
    :param img_model_type: Type of image model to use {"resnet18", "resnet50"}
    """

    def __init__(self, img_model_type: str):
        super().__init__()
        self.img_model_type = img_model_type
        self.encoder = self._create_encoder()

    def _create_encoder(self, **kwargs: Any) -> nn.Module:
        if self.img_model_type == 'resnet18':
            encoder = resnet18(return_all_feature_maps=True, pretrained=True, **kwargs)
        elif self.img_model_type == 'resnet50':
            encoder = resnet50(return_all_feature_maps=True, pretrained=True, **kwargs)
        else:
            raise NotImplementedError

        return encoder

    def forward(self, x: torch.Tensor, return_patch_embeddings: bool = False) -> Union[T, Tuple[T, T]]:
        """Image encoder forward pass."""

        x = self.encoder(x)
        x = x[-1] if isinstance(x, list) else x
        avg_pooled_emb = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)), 1)
        if return_patch_embeddings:
            return x, avg_pooled_emb

        return avg_pooled_emb

    def reload_encoder_with_dilation(self, replace_stride_with_dilation: List[bool] = [False, False, True]) -> None:
        """
        This is a workaround for enabling dilated convolutions after the model initialization.

        :param replace_stride_with_dilation: for each layer to replace the 2x2 stride with a dilated convolution
        """
        if self.img_model_type == 'resnet18':
            # resnet18 uses BasicBlock implementation, which does not support dilated convolutions.
            raise NotImplementedError("resnet18 does not support dilated convolutions")

        device = next(self.encoder.parameters()).device
        new_encoder = self._create_encoder(replace_stride_with_dilation=replace_stride_with_dilation).to(device)

        if self.encoder.training:
            new_encoder.train()
        else:
            new_encoder.eval()

        new_encoder.load_state_dict(self.encoder.state_dict())
        self.encoder = new_encoder


def get_encoder_output_dim(module: torch.nn.Module) -> int:
    """
    Calculates the output dimension of ssl encoder by making a single forward pass.
    :param pl_module: pl encoder module
    :param dm: pl datamodule
    """
    # Target device
    device = next(module.parameters()).device  # type: ignore
    assert isinstance(device, torch.device)

    x = torch.rand((1, 3, 448, 448)).to(device)

    # Extract the number of output feature dimensions
    with torch.no_grad():
        representations = module(x)

    return representations.shape[1]
