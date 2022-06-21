#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------


from pathlib import Path
from typing import Tuple, Union

import torch
from torchvision.transforms import Compose

from health_multimodal.image.data.io import load_image
from health_multimodal.image.data.transforms import infer_resize_params
from health_multimodal.image.model.model import ImageModel


class ImageInferenceEngine:
    """
    Encapsulate inference-time operations on an image model.
    """

    def __init__(self, image_model: ImageModel, transform: Compose):
        """
        :param img_model: Pretrained model
        :param transforms: Transforms to apply to the image after loading. Must return a torch.Tensor that can be
                           input directly to the image model.
        :param pretrained_model_path: Path to the pretrained image encoder model.
        """

        assert isinstance(image_model, ImageModel), f"Expected an ImageModel, got {type(image_model)}"

        self.model = image_model
        self.transform = transform
        self.device = next(self.model.parameters()).device

        self.model.eval()
        self.resize_size, self.crop_size = infer_resize_params(self.transform.transforms)

    def load_and_transform_input_image(self,
                                       image_path: Path,
                                       return_original_shape: bool = False) -> \
            Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int, int]]]:
        """
        1. Read the image from the given path
        2. Apply transforms
        3. Add the batch dimension
        4. Move to the correct device.

        :param return_original_shape: Whether to return an extra tuple that has the original shape of the image
            before the transforms. The tuple returned contains (width, height).
        """
        image = load_image(image_path)
        transformed_image = self.transform(image).unsqueeze(0).to(self.device)
        if return_original_shape:
            return transformed_image, image.size
        else:
            return transformed_image

    @torch.no_grad()
    def get_patch_embeddings_from_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Computes image embeddings in the joint latent space and preserves the image grid.
        :param image_path: Path to the image to compute embeddings for.
        """
        input_image, img_shape = self.load_and_transform_input_image(image_path, return_original_shape=True)
        projected_img_emb = self.model.get_patchwise_projected_embeddings(input_image, normalize=True)  # type: ignore
        assert projected_img_emb.shape[0] == 1

        return projected_img_emb[0], img_shape
