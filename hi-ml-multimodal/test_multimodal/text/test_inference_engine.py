#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------


import pytest
import torch

from health_multimodal.text.inference_engine import TextInferenceEngine
from health_multimodal.text.utils import get_cxr_bert


def test_text_inference_init_model_type() -> None:
    """
    Test that init fails if the wrong model type is passed in
    """
    tokenizer, _ = get_cxr_bert()
    false_model = torch.nn.Linear(4, 4)
    with pytest.raises(AssertionError) as ex:
        TextInferenceEngine(tokenizer=tokenizer, text_model=false_model)  # type: ignore[arg-type]
    assert f"Expected a BertForMaskedLM, got {type(false_model)}" in str(ex)


def test_l2_normalization() -> None:
    """
    Test that the text embeddings (CLS token) are l2 normalized.
    """
    tokenizer, text_model = get_cxr_bert()

    text_inference = TextInferenceEngine(tokenizer=tokenizer, text_model=text_model)
    input_query = ["There is a tumor in the left lung", "Lungs are all clear"]
    embedding = text_inference.get_embeddings_from_prompt(prompts=input_query)
    norm = torch.norm(embedding, p=2, dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm))


def test_sentence_semantic_similarity() -> None:
    """
    Test that the sentence embedding similarity computed by the text model is meaningful.
    """
    tokenizer, text_model = get_cxr_bert()

    # CLS token has no dedicated meaning, but we can expect vector similarity due to token overlap between the sentences
    text_inference = TextInferenceEngine(tokenizer=tokenizer, text_model=text_model)
    input_query = ["There is a tumor in the left lung", "Tumor is present", "Patient is admitted to the hospital today"]
    embedding = text_inference.get_embeddings_from_prompt(input_query)
    pos_sim = torch.dot(embedding[0], embedding[1])
    neg_sim_1 = torch.dot(embedding[0], embedding[2])
    neg_sim_2 = torch.dot(embedding[1], embedding[2])
    assert pos_sim > neg_sim_1
    assert pos_sim > neg_sim_2


def test_triplet_similarities_with_inference_engine() -> None:
    """
    Test that the triplet sentence similarities computed by the text model are meaningful.
    """

    tokenizer, text_model = get_cxr_bert()
    text_inference = TextInferenceEngine(tokenizer=tokenizer, text_model=text_model)

    reference = \
        ["Heart size is top normal.", "There is no pneumothorax or pleural effusion",
         "The patient has been extubated.", "The lungs are clear bilaterally.",
         "No pleural effusions."]
    synonyms = \
        ["The cardiac silhouette is normal in size.", "No pleural effusion or pneumothorax is seen",
         "There has been interval extubation.", "The lungs are unremarkable.",
         "Also, the lateral pleural sinuses are free, which excludes major pleural effusion."]
    contradictions = \
        ["The heart is largely enlarged.", "The extent of the pleural effusion is constant.",
         "The patient is intubated", "The lungs are mostly clear aside from lower lung atelectasis.",
         "The loculated right pleural effusion has increased, and is now moderate in size."]

    synonym_score = text_inference.get_pairwise_similarities(reference, synonyms)
    contradictory_score = text_inference.get_pairwise_similarities(reference, contradictions)

    print("Synonym score:", synonym_score)
    print("Contradictory score:", contradictory_score)

    assert torch.all(synonym_score > contradictory_score)
    assert torch.all(synonym_score > 0.5)
    assert torch.all(1.0 >= synonym_score)
    assert torch.all(contradictory_score < 0.5)
    assert torch.all(contradictory_score >= -1.0)


def test_mlm_with_inference_engine_with_hf_hub() -> None:
    """
    Test that the MLM model can be used with the inference engine and the filled masked tokens are correct.
    """

    tokenizer, text_model = get_cxr_bert()
    text_inference = TextInferenceEngine(tokenizer=tokenizer, text_model=text_model)

    # ##### Test Masked Language Modelling ######
    mlm_prompts = ["Moderate [MASK] pleural effusions and associated [MASK]",
                   "Right basilar [MASK], potentially due to infiltrate in the proper clinical setting",
                   "The right basilar chest [MASK] appears to be in unchanged position",
                   "The small basal pneumothorax has slightly [MASK] compared to the prior",
                   "Poorly defined [MASK] in the right lung is concerning for aspiration",
                   "Retrocardiac opacity likely reflects known hiatal [MASK]"]

    target_tokens = [['bilateral', 'atelectasis'], ['opacity'], ['tube'], ['increased'], ['opacity'], ['hernia']]

    output_top_1 = text_inference.predict_masked_tokens(mlm_prompts)
    assert output_top_1 == target_tokens
