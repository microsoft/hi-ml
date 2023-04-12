#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import List

import torch


@dataclass
class ImageModelOutput:
    img_embedding: torch.Tensor
    patch_embeddings: torch.Tensor
    projected_global_embedding: torch.Tensor
    class_logits: torch.Tensor
    projected_patch_embeddings: torch.Tensor


@unique
class ImageEncoderType(str, Enum):
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    RESNET18_MULTI_IMAGE = "resnet18_multi_image"
    RESNET50_MULTI_IMAGE = "resnet50_multi_image"

    @classmethod
    def get_members(cls, multi_image_encoders_only: bool) -> List[ImageEncoderType]:
        if multi_image_encoders_only:
            return [cls.RESNET18_MULTI_IMAGE, cls.RESNET50_MULTI_IMAGE]
        else:
            return [member for member in cls]


@unique
class ImageEncoderWeightTypes(str, Enum):
    RANDOM = "random"
    IMAGENET = "imagenet"
    BIOVIL = "biovil"
    BIOVIL_T = "biovil_t"
