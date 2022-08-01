#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from .inference_engine import ImageInferenceEngine
from .data.transforms import create_chest_xray_transform_for_inference
from .model import get_biovil_resnet


TRANSFORM_RESIZE = 512
TRANSFORM_CENTER_CROP_SIZE = 480


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
