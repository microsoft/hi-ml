#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""Tools related to joint image and text inference"""

from math import ceil, floor
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from health_multimodal.image import ImageInferenceEngine
from health_multimodal.text import TextInferenceEngine


class ImageTextInferenceEngine:
    """Functions related to inference on :class:`ImageTextModel`."""

    def __init__(self,
                 image_inference_engine: ImageInferenceEngine,
                 text_inference_engine: TextInferenceEngine) -> None:
        self.image_inference_engine = image_inference_engine
        self.text_inference_engine = text_inference_engine

    @torch.no_grad()
    def get_similarity_score_from_raw_data(self,
                                           image_path: Path,
                                           query_text: Union[List[str], str]) -> float:
        """Compute the cosine similarity score between an image and one or more strings.

        If multiple strings are passed, their embeddings are averaged before L2-normalization.

        :param image_path: Path to the input chest X-ray, either a DICOM or JPEG file.
        :param query_text: Input radiology text phrase.
        :return: The similarity score between the image and the text.
        """
        assert not self.image_inference_engine.model.training
        assert not self.text_inference_engine.model.training

        query_text = [query_text] if isinstance(query_text, str) else query_text
        num_prompts = len(query_text)

        image_embedding = self.image_inference_engine.get_projected_global_embedding(image_path)
        text_embedding = self.text_inference_engine.get_embeddings_from_prompt(query_text, normalize=False)

        assert text_embedding.shape[0] == num_prompts
        text_embedding = text_embedding.mean(dim=0)
        text_embedding = F.normalize(text_embedding, dim=0, p=2)

        cos_similarity = image_embedding @ text_embedding.t()

        return cos_similarity.item()

    def get_similarity_map_from_raw_data(self,
                                         image_path: Path,
                                         query_text: str,
                                         interpolation: str = "nearest") -> np.ndarray:
        """Return a heatmap of the similarities between each patch embedding from the image and the text embedding.

        :param image_path: Path to the input chest X-ray, either a DICOM or JPEG file.
        :param query_text: Input radiology text phrase.
        :param interpolation: Interpolation method to upsample the heatmap so it matches the input image size.
            See :func:`torch.nn.functional.interpolate` for more details.
        :return: A heatmap of the similarities between each patch embedding from the image and the text embedding,
            with the same shape as the input image.
        """
        assert not self.image_inference_engine.model.training
        assert not self.text_inference_engine.model.training
        assert isinstance(query_text, str)

        # TODO: Add checks in here regarding the text query, etc.
        image_embedding, (width, height) = self.image_inference_engine.get_projected_patch_embeddings(image_path)
        text_embedding = self.text_inference_engine.get_embeddings_from_prompt(query_text)

        sim = self._get_similarity_map_from_embeddings(image_embedding, text_embedding)

        resized_sim_map = self.convert_similarity_to_image_size(
            sim,
            width=width,
            height=height,
            resize_size=self.image_inference_engine.resize_size,
            crop_size=self.image_inference_engine.crop_size,
            val_img_transform=self.image_inference_engine.transform,
            interpolation=interpolation,
        )
        return resized_sim_map

    @staticmethod
    def _get_similarity_map_from_embeddings(projected_patch_embeddings: torch.Tensor,
                                            projected_text_embeddings: torch.Tensor,
                                            sigma: float = 1.5) -> torch.Tensor:
        """Get smoothed similarity map for a given image patch embeddings and text embeddings.

        :param projected_patch_embeddings: [n_patches_h, n_patches_w, feature_size]
        :param projected_text_embeddings: [1, feature_size]
        :return: similarity_map: similarity map of shape [n_patches_h, n_patches_w]
        """
        n_patches_h, n_patches_w, feature_size = projected_patch_embeddings.shape
        assert feature_size == projected_text_embeddings.shape[1]
        assert projected_text_embeddings.shape[0] == 1
        assert projected_text_embeddings.dim() == 2
        patch_wise_similarity = projected_patch_embeddings.view(-1, feature_size) @ projected_text_embeddings.t()
        patch_wise_similarity = patch_wise_similarity.reshape(n_patches_h, n_patches_w).cpu().numpy()
        smoothed_similarity_map = torch.tensor(ndimage.gaussian_filter(
            patch_wise_similarity, sigma=(sigma, sigma), order=0))
        return smoothed_similarity_map

    @staticmethod
    def convert_similarity_to_image_size(
            similarity_map: torch.Tensor, width: int, height: int, resize_size: Optional[int],
            crop_size: Optional[int], val_img_transform: Optional[Callable] = None,
            interpolation: str = "nearest") -> np.ndarray:
        """
        Convert similarity map from raw patch grid to original image size,
        taking into account whether the image has been resized and/or cropped prior to entering the network.
        """
        n_patches_h, n_patches_w = similarity_map.shape[0], similarity_map.shape[1]
        target_shape = 1, 1, n_patches_h, n_patches_w
        smallest_dimension = min(height, width)

        # TODO:
        # verify_resize_params(val_img_transforms, resize_size, crop_size)

        reshaped_similarity = similarity_map.reshape(target_shape)
        align_corners_modes = "linear", "bilinear", "bicubic", "trilinear"
        align_corners = False if interpolation in align_corners_modes else None

        if crop_size is not None:
            if resize_size is not None:
                cropped_size_orig_space = int(crop_size * smallest_dimension / resize_size)
                target_size = cropped_size_orig_space, cropped_size_orig_space
            else:
                target_size = crop_size, crop_size
            similarity_map = F.interpolate(
                reshaped_similarity,
                size=target_size,
                mode=interpolation,
                align_corners=align_corners,
            )
            margin_w, margin_h = (width - target_size[0]), (height - target_size[1])
            margins_for_pad = (floor(margin_w / 2), ceil(margin_w / 2), floor(margin_h / 2), ceil(margin_h / 2))
            similarity_map = F.pad(similarity_map[0, 0], margins_for_pad, value=float("NaN"))
        else:
            similarity_map = F.interpolate(
                reshaped_similarity,
                size=(height, width),
                mode=interpolation,
                align_corners=align_corners,
            )[0, 0]
        return similarity_map.numpy()

    def to(self, device: torch.device) -> None:
        """Move models to the specified device."""
        self.image_inference_engine.to(device)
        self.text_inference_engine.to(device)
