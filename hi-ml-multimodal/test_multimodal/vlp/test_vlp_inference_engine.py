#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path
from typing import List, Union

import numpy as np
import pytest
import torch
from health_multimodal.image import ImageInferenceEngine, ImageModel, ResnetType
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.model.model import JOINT_FEATURE_SIZE
from health_multimodal.text.utils import get_cxr_bert_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine
from PIL import Image


CENTER_CROP_SIZE = 480


def _get_vlp_inference_engine() -> ImageTextInferenceEngine:

    image_inference = ImageInferenceEngine(
        image_model=ImageModel(img_model_type=ResnetType.RESNET50.value, joint_feature_size=JOINT_FEATURE_SIZE),
        transform=create_chest_xray_transform_for_inference(resize=512, center_crop_size=CENTER_CROP_SIZE))
    img_txt_inference = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=get_cxr_bert_inference(),
    )

    return img_txt_inference


@pytest.mark.parametrize("height", (400, 500, 650))
@pytest.mark.parametrize("query_text", ("", "hello", "this is a test"))
def test_vlp_inference(height: int, query_text: Union[str, List[str]]) -> None:
    image_embedding_shapes = {480: (15, 15), }
    width = 600

    img_txt_inference = _get_vlp_inference_engine()
    image_inference = img_txt_inference.image_inference_engine

    with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
        image_path = Path(f.name)
        image = Image.new('RGB', (width, height))
        image.save(image_path)

        # Test integrated VLP inference engine
        resampled_similarity_map = img_txt_inference.get_similarity_map_from_raw_data(
            image_path=image_path,
            query_text=query_text,
        )
        assert resampled_similarity_map.shape == (height, width)
        np.nan_to_num(resampled_similarity_map, copy=False)
        assert resampled_similarity_map.min() >= -1
        assert resampled_similarity_map.max() <= 1

        # Test individual components
        image_embedding, size = image_inference.get_projected_patch_embeddings(image_path)

        assert (width, height) == size
        expected_image_embedding_size = image_embedding_shapes[CENTER_CROP_SIZE]
        assert image_embedding.shape == (*expected_image_embedding_size, JOINT_FEATURE_SIZE)
        normalized_image_embedding = torch.norm(image_embedding, p=2, dim=-1)
        assert torch.allclose(normalized_image_embedding, torch.ones_like(normalized_image_embedding))

        text_embedding = img_txt_inference.text_inference_engine.get_embeddings_from_prompt(query_text)
        assert text_embedding.shape == (1, JOINT_FEATURE_SIZE)

        similarity_map = img_txt_inference._get_similarity_map_from_embeddings(image_embedding, text_embedding)
        assert similarity_map.shape == expected_image_embedding_size


@pytest.mark.parametrize("query_text", ("this is a test", ["Test prompt 1", "Test prompt 2"]))
def test_vlp_inference_global_similarity(query_text: str) -> None:

    img_txt_inference = _get_vlp_inference_engine()

    with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
        image_path = Path(f.name)
        height, width = 500, 600
        image = Image.new('RGB', (width, height))
        image.save(image_path)

        # Test global similarity score
        sim_score = img_txt_inference.get_similarity_score_from_raw_data(image_path=image_path,
                                                                         query_text=query_text)
        assert isinstance(sim_score, float)
        assert 1 >= sim_score >= -1
