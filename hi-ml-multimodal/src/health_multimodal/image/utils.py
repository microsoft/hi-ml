#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

from torchvision.datasets.utils import download_url

from .. import BIOVIL_IMAGE_WEIGHTS_NAME
from .. import BIOVIL_IMAGE_WEIGHTS_URL
from .. import BIOVIL_IMAGE_WEIGHTS_MD5
from .model import ImageModel
from .inference_engine import ImageInferenceEngine
from .data.transforms import create_chest_xray_transform_for_inference


MODEL_TYPE = "resnet50"
JOINT_FEATURE_SIZE = 128
TRANSFORM_RESIZE = 512
TRANSFORM_CENTER_CROP_SIZE = 480


def _download_biovil_image_model_weights() -> Path:
    """Download image model weights from Hugging Face.

    More information available at https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized.
    """
    root_dir = tempfile.gettempdir()
    download_url(
        BIOVIL_IMAGE_WEIGHTS_URL,
        root=root_dir,
        filename=BIOVIL_IMAGE_WEIGHTS_NAME,
        md5=BIOVIL_IMAGE_WEIGHTS_MD5,
    )
    return Path(root_dir, BIOVIL_IMAGE_WEIGHTS_NAME)


def get_biovil_resnet(pretrained: bool = True) -> ImageModel:
    """Download weights from Hugging Face and instantiate the image model."""
    resnet_checkpoint_path = _download_biovil_image_model_weights() if pretrained else None

    image_model = ImageModel(
        img_model_type=MODEL_TYPE,
        joint_feature_size=JOINT_FEATURE_SIZE,
        pretrained_model_path=resnet_checkpoint_path,
    )
    return image_model


def get_biovil_resnet_inference() -> ImageInferenceEngine:
    """Create a :class:`ImageInferenceEngine` for the image model.

    The model is downloaded from the Hugging Face Hub.
    The engine can be used to get embeddings from text prompts or masked token predictions.
    """
    image_model = get_biovil_resnet()
    transform = create_chest_xray_transform_for_inference(
        resize=TRANSFORM_RESIZE,
        center_crop_size=TRANSFORM_CENTER_CROP_SIZE,
    )
    image_inference = ImageInferenceEngine(image_model=image_model, transform=transform)
    return image_inference
