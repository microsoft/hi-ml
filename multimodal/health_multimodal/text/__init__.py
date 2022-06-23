#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""Text-related tools

.. currentmodule:: health_multimodal.text

.. autosummary::
   :toctree:

   inference_engine


.. currentmodule:: health_multimodal.text.data

.. autosummary::
   :toctree:

   io


.. currentmodule:: health_multimodal.text.model

.. autosummary::
   :toctree:

   configuration_cxrbert
   modelling_cxrbert
"""

from .data.io import TypePrompts


BIOMED_VLP_CXR_BERT_SPECIALIZED = "microsoft/BiomedVLP-CXR-BERT-specialized"

__all__ = [
    "BIOMED_VLP_CXR_BERT_SPECIALIZED",
    "TypePrompts",
]
