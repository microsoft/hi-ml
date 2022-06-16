#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------


import numpy as np
import torch
from math import ceil, floor

from scipy import ndimage
from typing import Callable, Optional

import torch.nn.functional as F

from health_multimodal.image.inference_engine import ImageInferenceEngine
from health_multimodal.text.inference_engine import TextInferenceEngine


class ImageTextInferenceEngine:
    """
    Encapsulate functions related to inference on ImageTextModels.
    """

    def __init__(self,
                 image_inference_engine: ImageInferenceEngine,
                 text_inference_engine: TextInferenceEngine) -> None:
        """
        This class takes an ImageTextModel as well as an ImageInferenceEngine and TextInferenceEngine.
        The primary reason for this is our inability to use the inference engines in `get_image_to_text_similarity_map`
        without refactoring the code.
        """
        self.image_inference_engine = image_inference_engine
        self.text_inference_engine = text_inference_engine

    def get_similarity_map_from_raw_data(self, image_path: str, query_text: str, verbose: bool = False) -> np.ndarray:
        """
        Return a heatmap of the similarities between each patch embedding from the image and the text embedding.
        """
        assert not self.image_inference_engine.model.training
        assert not self.text_inference_engine.model.training

        # TODO: Add checks in here regarding the text query and etc.

        image_embedding, (W, H) = self.image_inference_engine.get_patch_embeddings_from_image(image_path)
        text_embedding = self.text_inference_engine.get_embeddings_from_prompt(query_text)

        sim = self._get_similarity_map_from_embeddings(image_embedding, text_embedding)

        resized_sim_map = self.convert_similarity_to_image_size(sim, width=W, height=H,
                                                                resize_size=self.image_inference_engine.resize_size,
                                                                crop_size=self.image_inference_engine.crop_size,
                                                                val_img_transforms=self.image_inference_engine.transforms)
        return resized_sim_map

    @staticmethod
    def _get_similarity_map_from_embeddings(projected_patch_embeddings: torch.Tensor,
                                            projected_text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get smoothed similarity map for a given image patch embeddings and text embeddings.
        :param projected_patch_embeddings: [n_patches_h, n_patches_w, feature_size]
        :param projected_text_embeddings: [1, feature_size]
        :return: similarity_map: similarity map of shape [n_patches_h, n_patches_w]
        """
        sigma = 1.5
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
            crop_size: Optional[int], val_img_transforms: Optional[Callable]) -> np.ndarray:
        """
        Converts similarity map from raw patch grid to original image size, taking into account whether the image has been
        resized and/or cropped prior to entering the network.
        """
        n_patches_h, n_patches_w = similarity_map.shape[0], similarity_map.shape[1]
        smallest_dimension = min(height, width)

        # TODO:
        # verify_resize_params(val_img_transforms, resize_size, crop_size)

        if crop_size is not None:
            if resize_size is not None:
                cropped_size_orig_space = int(crop_size * smallest_dimension / resize_size)
                target_size = (cropped_size_orig_space, cropped_size_orig_space)
            else:
                target_size = (crop_size, crop_size)
            similarity_map = F.interpolate(
                similarity_map.reshape((1, 1, n_patches_h, n_patches_w)), size=target_size)[0, 0]
            margin_w, margin_h = (width - target_size[0]), (height - target_size[1])
            margins_for_pad = (floor(margin_w / 2), ceil(margin_w / 2), floor(margin_h / 2), ceil(margin_h / 2))
            similarity_map = F.pad(similarity_map, margins_for_pad, value=float("NaN"))
        else:
            similarity_map = F.interpolate(
                similarity_map.reshape((1, 1, n_patches_h, n_patches_w)), size=(height, width))[0, 0]
        return similarity_map.numpy()
