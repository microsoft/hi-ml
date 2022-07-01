from typing import Tuple

from health_multimodal.text import BIOMED_VLP_CXR_BERT_SPECIALIZED, CXR_BERT_COMMIT_ID
from health_multimodal.text.inference_engine import TextInferenceEngine
from health_multimodal.text.model.modelling_cxrbert import CXRBertModel
from health_multimodal.text.model.configuration_cxrbert import CXRBertTokenizer


def get_cxr_bert() -> Tuple[CXRBertTokenizer, CXRBertModel]:
    """Loads the CXR-BERT model and tokenizer from HUGGINGFACE HUB
       `https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized`
    """

    model_name = BIOMED_VLP_CXR_BERT_SPECIALIZED
    revision = CXR_BERT_COMMIT_ID
    tokenizer = CXRBertTokenizer.from_pretrained(model_name, revision=revision)
    text_model = CXRBertModel.from_pretrained(model_name, revision=revision)
    return tokenizer, text_model


def get_cxr_bert_inference() -> TextInferenceEngine:
    """Creates a TextInferenceEngine for the CXR-BERT model after downloading the model from HUGGINGFACE HUB
       The engine can be used to get embeddings from text prompts or masked token predictions.
    """

    tokenizer, text_model = get_cxr_bert()
    text_inference = TextInferenceEngine(tokenizer=tokenizer, text_model=text_model)

    return text_inference
