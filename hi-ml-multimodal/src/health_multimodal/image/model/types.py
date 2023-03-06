#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ImageModelInput():
    current_image: torch.Tensor
    previous_image: Optional[torch.Tensor] = None

@dataclass
class ImageModelOutput():
    img_embedding: torch.Tensor
    patch_embedding: torch.Tensor
    projected_global_embedding: torch.Tensor
    class_logits: torch.Tensor
    projected_patch_embeddings: torch.Tensor
