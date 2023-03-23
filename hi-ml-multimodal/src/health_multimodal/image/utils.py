#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from enum import Enum, unique

from .data.transforms import create_chest_xray_transform_for_inference
from .inference_engine import ImageInferenceEngine
from .model.pretrained import get_biovil_image_encoder
from .model.pretrained import get_biovil_t_image_encoder

TRANSFORM_RESIZE = 512


@unique
class ImageModelType(str, Enum):
    BIOVIL = "biovil"
    BIOVIL_T = "biovil_t"


def get_image_inference(image_model_type: ImageModelType = ImageModelType.BIOVIL_T) -> ImageInferenceEngine:
    """Create a :class:`ImageInferenceEngine` for the image model.

    :param image_model_type: The type of image model to use, `BIOVIL` or `BIOVIL_T`.

    The model is downloaded from the Hugging Face Hub.
    The engine can be used to get embeddings from text prompts or masked token predictions.
    """

    if image_model_type == ImageModelType.BIOVIL_T:
        image_model = get_biovil_t_image_encoder()
        transform_center_crop_size = 448
    elif image_model_type == ImageModelType.BIOVIL:
        image_model = get_biovil_image_encoder()
        transform_center_crop_size = 480
    else:
        raise ValueError(f"Unknown image_model_type: {image_model_type}")

    transform = create_chest_xray_transform_for_inference(
        resize=TRANSFORM_RESIZE, center_crop_size=transform_center_crop_size
    )
    image_inference = ImageInferenceEngine(image_model=image_model, transform=transform)

    return image_inference
