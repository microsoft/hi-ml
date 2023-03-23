#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import logging
from typing import Any, List, Union

from transformers import BertTokenizer


TypePrompts = Union[str, List[str]]

logger = logging.getLogger(__name__)


class TextInput:
    """Text input class that can be used for inference and deployment.

    Implements tokenizer related operations and ensure that input strings
    conform with the standards expected from a BERT model.

    :param tokenizer: A BertTokenizer object.
    """

    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer

    def tokenize_input_prompts(self, prompts: TypePrompts, verbose: bool) -> Any:
        """
        Tokenizes the input sentence(s) and adds special tokens as defined by the tokenizer.
        :param prompts: Either a string containing a single sentence, or a list of strings each containing
            a single sentence. Note that this method will not correctly tokenize multiple sentences if they
            are input as a single string.
        :param verbose: If set to True, will log the sentence after tokenization.
        :return: A 2D tensor containing the tokenized sentences
        """
        prompts = [prompts] if isinstance(prompts, str) else prompts
        self.assert_special_tokens_not_present(" ".join(prompts))

        prompts = [prompt.rstrip("!?.") for prompt in prompts]  # removes punctuation from end of prompt
        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=prompts, add_special_tokens=True, padding='longest', return_tensors='pt'
        )
        if verbose:
            for prompt in tokenizer_output.input_ids:
                input_tokens = self.tokenizer.convert_ids_to_tokens(prompt.tolist())
                logger.info(f"Input tokens: {input_tokens}")

        return tokenizer_output

    def assert_special_tokens_not_present(self, prompt: str) -> None:
        """Check if the input prompts contain special tokens."""
        special_tokens = self.tokenizer.all_special_tokens
        special_tokens.remove(self.tokenizer.mask_token)  # [MASK] is allowed
        if any(map(lambda token: token in prompt, special_tokens)):
            raise ValueError(f"The input \"{prompt}\" contains at least one special token ({special_tokens})")
