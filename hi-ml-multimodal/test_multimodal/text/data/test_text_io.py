#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import pytest

from health_multimodal.text import TypePrompts
from health_multimodal.text.utils import get_cxr_bert_inference

text_inference = get_cxr_bert_inference()


@pytest.mark.parametrize("prompts", ("", "hello", "this is a test", ["this is", "also a test"]))
def test_good_prompts(prompts: TypePrompts) -> None:
    text_inference.tokenize_input_prompts(prompts)


@pytest.mark.parametrize("prompts", ("[CLS]", "hello [PAD]"))
def test_bad_prompts(prompts: TypePrompts) -> None:
    with pytest.raises(ValueError, match="The input .* contains at least one special token"):
        text_inference.tokenize_input_prompts(prompts)
