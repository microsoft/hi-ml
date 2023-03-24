#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from health_multimodal.common.device import get_module_device

from .encoder import get_encoder_from_type, get_encoder_output_dim, MultiImageEncoder
from .modules import MLP, MultiTaskModel
from .types import ImageModelOutput


class BaseImageModel(nn.Module, ABC):
    """Abstract class for image models."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> ImageModelOutput:
        raise NotImplementedError

    @abstractmethod
    def get_patchwise_projected_embeddings(self, input_img: torch.Tensor, normalize: bool) -> torch.Tensor:
        raise NotImplementedError


class ImageModel(BaseImageModel):
    """Image encoder module"""

    def __init__(
        self,
        img_encoder_type: str,
        joint_feature_size: int,
        freeze_encoder: bool = False,
        pretrained_model_path: Optional[Union[str, Path]] = None,
        **downstream_classifier_kwargs: Any,
    ):
        super().__init__()

        # Initiate encoder, projector, and classifier
        self.encoder = get_encoder_from_type(img_encoder_type)
        self.feature_size = get_encoder_output_dim(self.encoder, device=get_module_device(self.encoder))
        self.projector = MLP(
            input_dim=self.feature_size,
            output_dim=joint_feature_size,
            hidden_dim=joint_feature_size,
            use_1x1_convs=True,
        )
        self.downstream_classifier_kwargs = downstream_classifier_kwargs
        self.classifier = self.create_downstream_classifier() if downstream_classifier_kwargs else None

        # Initialise the mode of modules
        self.freeze_encoder = freeze_encoder
        self.train()

        if pretrained_model_path is not None:
            if not isinstance(pretrained_model_path, (str, Path)):
                raise TypeError(f"Expected a string or Path, got {type(pretrained_model_path)}")
            state_dict = torch.load(pretrained_model_path, map_location="cpu")
            self.load_state_dict(state_dict)

    def train(self, mode: bool = True) -> Any:
        """Switch the model between training and evaluation modes."""
        super().train(mode=mode)
        if self.freeze_encoder:
            self.encoder.train(mode=False)
            self.projector.train(mode=False)
        return self

    def forward(self, x: torch.Tensor) -> ImageModelOutput:  # type: ignore[override]
        with torch.set_grad_enabled(not self.freeze_encoder):
            patch_x, pooled_x = self.encoder(x, return_patch_embeddings=True)
        return self.forward_post_encoder(patch_x, pooled_x)

    def forward_post_encoder(self, patch_x: torch.Tensor, pooled_x: torch.Tensor) -> ImageModelOutput:
        with torch.set_grad_enabled(not self.freeze_encoder):
            projected_patch_embeddings = self.projector(patch_x)
            projected_global_embedding = torch.mean(projected_patch_embeddings, dim=(2, 3))

        logits = self.classifier(pooled_x) if self.classifier else None
        return ImageModelOutput(
            img_embedding=pooled_x,
            patch_embeddings=patch_x,
            class_logits=logits,
            projected_patch_embeddings=projected_patch_embeddings,
            projected_global_embedding=projected_global_embedding,
        )

    def create_downstream_classifier(self, **kwargs: Any) -> MultiTaskModel:
        """Create the classification module for the downstream task."""
        downstream_classifier_kwargs = kwargs if kwargs else self.downstream_classifier_kwargs
        return MultiTaskModel(self.feature_size, **downstream_classifier_kwargs)

    @torch.no_grad()
    def get_patchwise_projected_embeddings(self, input_img: torch.Tensor, normalize: bool) -> torch.Tensor:
        """Get patch-wise projected embeddings from the CNN model.

        :param input_img: input tensor image [B, C, H, W].
        :param normalize: If ``True``, the embeddings are L2-normalized.
        :returns projected_embeddings: tensor of embeddings in shape [batch, n_patches_h, n_patches_w, feature_size].
        """
        assert not self.training, "This function is only implemented for evaluation mode"
        outputs = self.forward(input_img)
        projected_embeddings = outputs.projected_patch_embeddings.detach()  # type: ignore
        if normalize:
            projected_embeddings = F.normalize(projected_embeddings, dim=1)
        projected_embeddings = projected_embeddings.permute([0, 2, 3, 1])  # B D H W -> B H W D (D: Features)
        return projected_embeddings


class MultiImageModel(ImageModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert isinstance(self.encoder, MultiImageEncoder), "MultiImageModel only supports MultiImageEncoder"

    def forward(  # type: ignore[override]
        self, current_image: torch.Tensor, previous_image: Optional[torch.Tensor] = None
    ) -> ImageModelOutput:
        with torch.set_grad_enabled(not self.freeze_encoder):
            patch_x, pooled_x = self.encoder(
                current_image=current_image, previous_image=previous_image, return_patch_embeddings=True
            )
        return self.forward_post_encoder(patch_x, pooled_x)
