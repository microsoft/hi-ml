#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import pytest
from transformers import AutoModel, AutoTokenizer

from health_multimodal.text import BIOMED_VLP_CXR_BERT_SPECIALIZED, TypePrompts
from health_multimodal.text.inference_engine import TextInferenceEngine


text_inference = TextInferenceEngine(
    tokenizer=AutoTokenizer.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, trust_remote_code=True),
    text_model=AutoModel.from_pretrained(BIOMED_VLP_CXR_BERT_SPECIALIZED, trust_remote_code=True),
)


@pytest.mark.parametrize("prompts", ("", "hello", "this is a test", ["this is", "also a test"]))
def test_good_prompts(prompts: TypePrompts) -> None:
    text_inference.tokenize_input_prompts(prompts)


@pytest.mark.parametrize("prompts", ("[CLS]", "hello [PAD]"))
def test_bad_prompts(prompts: TypePrompts) -> None:
    with pytest.raises(ValueError, match="The input .* contains at least one special token"):
        text_inference.tokenize_input_prompts(prompts)
