#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------


from typing import Any, List, Union

import torch
from transformers import BertForMaskedLM, BertTokenizer

from health_multimodal.text.data.io import TextInput


class TextInferenceEngine(TextInput):
    """
    Text inference class that implements functionalities required to extract
    sentence embedding, similarity and MLM prediction tasks.

    :param tokenizer: A BertTokenizer object.
    :param text_model: Text model either default HuggingFace class
    """

    def __init__(self, tokenizer: BertTokenizer, text_model: BertForMaskedLM) -> None:
        super().__init__(tokenizer=tokenizer)

        assert isinstance(text_model, BertForMaskedLM), f"Expected a BertForMaskedLM, got {type(text_model)}"

        self.model = text_model
        self.max_allowed_input_length = self.model.config.max_position_embeddings
        self.to = self.model.to

    def is_in_eval(self) -> bool:
        """Returns True if the model is in eval mode."""
        return not self.model.training

    def tokenize_input_prompts(self, prompts: Union[str, List[str]], verbose: bool = True) -> Any:
        tokenizer_output = super().tokenize_input_prompts(prompts, verbose=verbose)
        device = next(self.model.parameters()).device
        tokenizer_output.input_ids = tokenizer_output.input_ids.to(device)
        tokenizer_output.attention_mask = tokenizer_output.attention_mask.to(device)

        max_length = tokenizer_output.input_ids.shape[1]
        if tokenizer_output.input_ids.shape[1] > self.max_allowed_input_length:
            raise ValueError(
                f"The sequence length of the input ({max_length}) is "
                f"longer than the maximum allowed sequence length ({self.max_allowed_input_length})."
            )

        return tokenizer_output

    @torch.no_grad()
    def get_embeddings_from_prompt(
        self, prompts: Union[str, List[str]], normalize: bool = True, verbose: bool = True
    ) -> torch.Tensor:
        """Generate L2-normalised embeddings for a list of input text prompts.

        :param prompts: Input text prompt(s) either in string or list of string format.
        :param normalize: If True, L2-normalise the embeddings.
        :param verbose: If set to True, tokenized words are displayed in the console.
        :return: Tensor of shape (batch_size, embedding_size).
        """

        assert self.is_in_eval()
        tokenizer_output = self.tokenize_input_prompts(prompts=prompts, verbose=verbose)
        txt_emb = self.model.get_projected_text_embeddings(  # type: ignore
            input_ids=tokenizer_output.input_ids,
            attention_mask=tokenizer_output.attention_mask,
            normalize_embeddings=normalize,
        )

        return txt_emb

    @torch.no_grad()
    def get_pairwise_similarities(
        self, prompt_set_1: Union[str, List[str]], prompt_set_2: Union[str, List[str]]
    ) -> torch.Tensor:
        """Compute pairwise cosine similarities between the embeddings of the given prompts."""

        emb_1 = self.get_embeddings_from_prompt(prompts=prompt_set_1, verbose=False)
        emb_2 = self.get_embeddings_from_prompt(prompts=prompt_set_2, verbose=False)
        sim = torch.diag(torch.mm(emb_1, emb_2.t())).detach()

        return sim

    @torch.no_grad()
    def predict_masked_tokens(self, prompts: Union[str, List[str]]) -> List[List[str]]:
        """Predict masked tokens for a single or list of input text prompts.

        Requires models to be trained with a MLM prediction head.

        :param prompts: Input text prompt(s) either in string or list of string format.
        :return: Predicted token candidates (Top-1) at masked position.
        """

        assert self.is_in_eval()

        # Tokenize the input prompts
        tokenized_prompts = self.tokenize_input_prompts(prompts)

        # Collect all token predictions
        text_model_output = self.model.forward(
            input_ids=tokenized_prompts.input_ids, attention_mask=tokenized_prompts.attention_mask
        )
        logits = text_model_output.logits
        logits = logits.detach()

        predicted_token_ids = torch.argmax(logits, dim=-1)  # Batch x Seq

        # Identify the masked token indices
        batch_size = predicted_token_ids.shape[0]
        mask_token_id = self.tokenizer.mask_token_id
        mlm_mask = tokenized_prompts.input_ids == mask_token_id  # Batch x Seq

        # Convert the predicted token ids to token strings
        output = list()
        for b in range(batch_size):
            _ids = predicted_token_ids[b, mlm_mask[b]].cpu().tolist()
            _tokens = self.tokenizer.convert_ids_to_tokens(_ids)
            output.append(_tokens)

        return output
