#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from health_multimodal.image import ImageModel, ResnetType, ImageInferenceEngine
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference


@pytest.mark.parametrize("height", (400, 500, 650))
def test_image_inference_engine(height: int) -> None:
    """Test the image inference engine with a dummy image and ensure that the output is of the correct shape."""

    joint_feature_size = 128
    resize = 512
    center_crop_size = 480

    width = 600
    image_inference = ImageInferenceEngine(
        image_model=ImageModel(img_model_type=ResnetType.RESNET50.value, joint_feature_size=joint_feature_size),
        transform=create_chest_xray_transform_for_inference(resize=resize, center_crop_size=center_crop_size))

    with tempfile.NamedTemporaryFile(suffix='.jpg') as f:
        image_path = Path(f.name)
        image = Image.new('RGB', (width, height))
        image.save(image_path)

        # Test individual components
        image_embedding = image_inference.get_projected_global_embedding(image_path)
        assert image_embedding.shape == (joint_feature_size,)
        assert torch.allclose(torch.norm(image_embedding), torch.tensor([1.00]))
