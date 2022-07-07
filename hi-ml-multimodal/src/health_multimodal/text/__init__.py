#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

"""Text-related tools

.. currentmodule:: health_multimodal.text

.. autosummary::
   :toctree:

   inference_engine
   utils


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
from .inference_engine import TextInferenceEngine
from .model.modelling_cxrbert import CXRBertModel, CXRBertOutput
from .model.configuration_cxrbert import CXRBertConfig, CXRBertTokenizer


BIOMED_VLP_CXR_BERT_SPECIALIZED = "microsoft/BiomedVLP-CXR-BERT-specialized"
CXR_BERT_COMMIT_TAG = "v1.0"

__all__ = [
    "BIOMED_VLP_CXR_BERT_SPECIALIZED",
    "CXR_BERT_COMMIT_TAG",
    "TypePrompts",
    "TextInferenceEngine",
    "CXRBertConfig",
    "CXRBertTokenizer",
    "CXRBertModel",
    "CXRBertOutput",
]
