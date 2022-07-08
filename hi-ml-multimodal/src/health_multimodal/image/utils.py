#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

from torchvision.datasets.utils import download_url

from .. import IMAGE_WEIGHTS_NAME
from .. import IMAGE_WEIGHTS_URL
from .. import IMAGE_WEIGHTS_MD5
from .model import ImageModel
from .inference_engine import ImageInferenceEngine
from .data.transforms import create_chest_xray_transform_for_inference


def _download_image_model_weights() -> Path:
    """Download image model weights from Hugging Face."""
    root_dir = tempfile.gettempdir()
    download_url(
        IMAGE_WEIGHTS_URL,
        root=tempfile.gettempdir(),
        filename=IMAGE_WEIGHTS_NAME,
        md5=IMAGE_WEIGHTS_MD5,
    )
    return Path(root_dir, IMAGE_WEIGHTS_NAME)


def get_cxr_resnet(model_type: str = "resnet50", joint_feature_size: int = 128) -> ImageModel:
    """Download weights from Hugging Face and instantiate the image model."""
    resnet_checkpoint_path = _download_image_model_weights()
    image_model = ImageModel(
        img_model_type=model_type,
        joint_feature_size=joint_feature_size,
        pretrained_model_path=resnet_checkpoint_path,
    )
    return image_model


def get_cxr_resnet_inference(resize: int = 512, center_crop_size: int = 480) -> ImageInferenceEngine:
    """Create a :class:`ImageInferenceEngine` for the image model.

    The model is downloaded from the Hugging Face Hub.
    The engine can be used to get embeddings from text prompts or masked token predictions.
    """
    image_model = get_cxr_resnet()
    transform = create_chest_xray_transform_for_inference(resize=resize, center_crop_size=center_crop_size)
    image_inference = ImageInferenceEngine(image_model=image_model, transform=transform)
    return image_inference
