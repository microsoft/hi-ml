#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor as T
from transformers import BertForMaskedLM
from transformers.modeling_outputs import ModelOutput

from health_multimodal.text.model.configuration_cxrbert import CXRBertConfig

BERTTupleOutput = Tuple[T, T, T, T, T]


@dataclass
class CXRBertOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    logits: Optional[torch.FloatTensor] = None
    cls_projected_embedding: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertProjectionHead(nn.Module):
    """Projection head to be used with BERT CLS token.

    This is similar to ``BertPredictionHeadTransform`` in HuggingFace.

    :param config: Configuration for BERT.
    """

    def __init__(self, config: CXRBertConfig) -> None:
        super().__init__()
        self.dense_to_hidden = nn.Linear(config.hidden_size, config.projection_size)
        self.transform_act_fn = nn.functional.gelu
        self.LayerNorm = nn.LayerNorm(config.projection_size, eps=1e-12)
        self.dense_to_output = nn.Linear(config.projection_size, config.projection_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_to_hidden(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense_to_output(hidden_states)

        return hidden_states


class CXRBertModel(BertForMaskedLM):
    """
    Implements the CXR-BERT model outlined in the manuscript:
    Boecking et al. "Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing", 2022
    https://link.springer.com/chapter/10.1007/978-3-031-20059-5_1

    Extends the HuggingFace BertForMaskedLM model by adding a separate projection head. The projection "[CLS]" token is
    used to align the latent vectors of image and text modalities.
    """

    config_class = CXRBertConfig  # type: ignore

    def __init__(self, config: CXRBertConfig):
        super().__init__(config)

        self.cls_projection_head = BertProjectionHead(config)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_cls_projected_embedding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any
    ) -> Union[BERTTupleOutput, CXRBertOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bert_for_masked_lm_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = bert_for_masked_lm_output.hidden_states[-1]
        cls_projected_embedding = (
            self.cls_projection_head(last_hidden_state[:, 0, :]) if output_cls_projected_embedding else None
        )

        if return_dict:
            return CXRBertOutput(
                last_hidden_state=last_hidden_state,
                logits=bert_for_masked_lm_output.logits,
                cls_projected_embedding=cls_projected_embedding,
                hidden_states=bert_for_masked_lm_output.hidden_states if output_hidden_states else None,
                attentions=bert_for_masked_lm_output.attentions,
            )
        else:
            return (
                last_hidden_state,
                bert_for_masked_lm_output.logits,
                cls_projected_embedding,
                bert_for_masked_lm_output.hidden_states,
                bert_for_masked_lm_output.attentions,
            )

    def get_projected_text_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize_embeddings: bool = True
    ) -> torch.Tensor:
        """
        Returns l2-normalised projected cls token embeddings for the given input token ids and attention mask.
        The joint latent space is trained using a contrastive objective between image and text data modalities.

        :param input_ids: (batch_size, sequence_length)
        :param attention_mask: (batch_size, sequence_length)
        :param normalize_embeddings: Whether to l2-normalise the embeddings.
        :return: (batch_size, projection_size)
        """

        outputs = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, output_cls_projected_embedding=True, return_dict=True
        )
        assert isinstance(outputs, CXRBertOutput)

        cls_projected_embedding = outputs.cls_projected_embedding
        assert cls_projected_embedding is not None

        if normalize_embeddings:
            return F.normalize(cls_projected_embedding, dim=1)

        return cls_projected_embedding
