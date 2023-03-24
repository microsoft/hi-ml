#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------


from enum import Enum, unique
from typing import Tuple

from ..image.model.pretrained import (
    BIOMED_VLP_BIOVIL_T,
    BIOMED_VLP_CXR_BERT_SPECIALIZED,
    BIOVIL_T_COMMIT_TAG,
    CXR_BERT_COMMIT_TAG,
)
from .inference_engine import TextInferenceEngine
from .model import CXRBertModel, CXRBertTokenizer


@unique
class BertEncoderType(str, Enum):
    CXR_BERT = "cxr_bert"
    BIOVIL_T_BERT = "biovil_t_bert"


def get_biovil_t_bert() -> Tuple[CXRBertTokenizer, CXRBertModel]:
    """Load the BioViL-T Bert model and tokenizer from the `Hugging Face Hub <https://huggingface.co/microsoft/BiomedVLP-BioViL-T>`_."""  # noqa: B950
    tokenizer = CXRBertTokenizer.from_pretrained(BIOMED_VLP_BIOVIL_T, revision=BIOVIL_T_COMMIT_TAG)
    text_model = CXRBertModel.from_pretrained(BIOMED_VLP_BIOVIL_T, revision=BIOVIL_T_COMMIT_TAG)
    return tokenizer, text_model


def get_cxr_bert() -> Tuple[CXRBertTokenizer, CXRBertModel]:
    """Load the CXR-BERT model and tokenizer from the `Hugging Face Hub <https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized>`_."""  # noqa: B950
    tokenizer = CXRBertTokenizer.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG)
    text_model = CXRBertModel.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, revision=CXR_BERT_COMMIT_TAG)
    return tokenizer, text_model


def get_bert_inference(bert_encoder_type: BertEncoderType = BertEncoderType.BIOVIL_T_BERT) -> TextInferenceEngine:
    """Create a :class:`TextInferenceEngine` for a text encoder model.

    :param bert_encoder_type: The type of text encoder model to use, `CXR_BERT` or `BIOVIL_T_BERT`.

    The model is downloaded from the Hugging Face Hub.
    The engine can be used to get embeddings from text prompts or masked token predictions.
    """
    if bert_encoder_type == BertEncoderType.CXR_BERT:
        tokenizer, text_model = get_cxr_bert()
    elif bert_encoder_type == BertEncoderType.BIOVIL_T_BERT:
        tokenizer, text_model = get_biovil_t_bert()
    else:
        raise ValueError(f"Unknown bert_encoder_type: {bert_encoder_type}")

    text_inference = TextInferenceEngine(tokenizer=tokenizer, text_model=text_model)
    return text_inference
