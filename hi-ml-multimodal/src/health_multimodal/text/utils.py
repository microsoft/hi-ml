#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------


from typing import Tuple

from ..image.model.pretrained import BIOVIL_T_COMMIT_TAG, CXR_BERT_COMMIT_TAG
from ..image.model.pretrained import BIOMED_VLP_BIOVIL_T, BIOMED_VLP_CXR_BERT_SPECIALIZED
from .inference_engine import TextInferenceEngine
from .model import CXRBertModel
from .model import CXRBertTokenizer


def get_biovil_t_bert() -> Tuple[CXRBertTokenizer, CXRBertModel]:
    """Load the BioViL-T Bert model and tokenizer from the `Hugging Face Hub <https://huggingface.co/microsoft/BiomedVLP-BioViL-T>`_."""  # noqa: E501
    tokenizer = CXRBertTokenizer.from_pretrained(BIOMED_VLP_BIOVIL_T, revision=BIOVIL_T_COMMIT_TAG)
    text_model = CXRBertModel.from_pretrained(BIOMED_VLP_BIOVIL_T, revision=BIOVIL_T_COMMIT_TAG)
    return tokenizer, text_model


def get_cxr_bert() -> Tuple[CXRBertTokenizer, CXRBertModel]:
    """Load the CXR-BERT model and tokenizer from the `Hugging Face Hub <https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized>`_."""  # noqa: E501
    tokenizer = CXRBertTokenizer.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG)
    text_model = CXRBertModel.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG)
    return tokenizer, text_model


def get_cxr_bert_inference() -> TextInferenceEngine:
    """Create a :class:`TextInferenceEngine` for the CXR-BERT model.

    The model is downloaded from the Hugging Face Hub.
    The engine can be used to get embeddings from text prompts or masked token predictions.
    """
    tokenizer, text_model = get_cxr_bert()
    text_inference = TextInferenceEngine(tokenizer=tokenizer, text_model=text_model)
    return text_inference
