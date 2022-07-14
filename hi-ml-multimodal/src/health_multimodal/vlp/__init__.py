#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""Visual-language processing tools

.. currentmodule:: health_multimodal.vlp

.. autosummary::
   :toctree:

   inference_engine
"""

from .inference_engine import ImageTextInferenceEngine

__all__ = [
    "ImageTextInferenceEngine",
]
