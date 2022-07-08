#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""Image-related tools


.. currentmodule:: health_multimodal.image

.. autosummary::
   :toctree:

   inference_engine
   utils


.. currentmodule:: health_multimodal.image.data

.. autosummary::
   :toctree:

   io
   transforms


.. currentmodule:: health_multimodal.image.model

.. autosummary::
   :toctree:

   model
   modules
   resnet
"""

from .. import REPO_URL
from .model import ImageModel
from .model import ResnetType
from .inference_engine import ImageInferenceEngine
from .utils import get_cxr_resnet_inference

IMAGE_WEIGHTS_NAME = ""  # TODO
IMAGE_WEIGHTS_URL = f"{REPO_URL}/raw/main/{IMAGE_WEIGHTS_NAME}"
IMAGE_WEIGHTS_MD5 = ""  # TODO

__all__ = [
    "ImageModel",
    "ResnetType",
    "ImageInferenceEngine",
    "IMAGE_WEIGHTS_NAME",
    "IMAGE_WEIGHTS_URL",
    "IMAGE_WEIGHTS_MD5",
    "get_cxr_resnet_inference",
]
