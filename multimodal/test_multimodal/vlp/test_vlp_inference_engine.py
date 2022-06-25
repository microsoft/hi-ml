#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import torch
import pytest
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from health_multimodal.text import BIOMED_VLP_CXR_BERT_SPECIALIZED
from health_multimodal.text.inference_engine import TextInferenceEngine
from health_multimodal.image import ImageModel, ResnetType, ImageInferenceEngine
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.vlp.inference_engine import ImageTextInferenceEngine


text_inference = TextInferenceEngine(
    tokenizer=AutoTokenizer.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, trust_remote_code=True),
    text_model=AutoModel.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, trust_remote_code=True),
)


@pytest.mark.parametrize("height", (400, 500, 650))
@pytest.mark.parametrize("query_text", ("", "hello", "this is a test"))
def test_vlp_inference(height: int, query_text: str) -> None:
    image_embedding_shapes = {
        480: (15, 15),
    }

    joint_feature_size = 128
    resize = 512
    center_crop_size = 480

    width = 600
    image_inference = ImageInferenceEngine(
        image_model=ImageModel(img_model_type=ResnetType.RESNET50.value, joint_feature_size=joint_feature_size),
        transform=create_chest_xray_transform_for_inference(resize=resize, center_crop_size=center_crop_size))
    img_txt_inference = ImageTextInferenceEngine(
        image_inference_engine=image_inference,
        text_inference_engine=text_inference,
    )

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
        image_embedding, size = img_txt_inference.image_inference_engine.get_patch_embeddings_from_image(image_path)

        assert (width, height) == size
        expected_image_embedding_size = image_embedding_shapes[center_crop_size]
        assert image_embedding.shape == (*expected_image_embedding_size, joint_feature_size)
        normalized_image_embedding = torch.norm(image_embedding, p=2, dim=-1)
        assert torch.allclose(normalized_image_embedding, torch.ones_like(normalized_image_embedding))

        text_embedding = img_txt_inference.text_inference_engine.get_embeddings_from_prompt(query_text)
        assert text_embedding.shape == (1, joint_feature_size)

        similarity_map = img_txt_inference._get_similarity_map_from_embeddings(image_embedding, text_embedding)
        assert similarity_map.shape == expected_image_embedding_size
