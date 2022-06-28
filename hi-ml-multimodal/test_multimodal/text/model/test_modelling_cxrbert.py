#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import pytest
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME

from health_multimodal.text.model.configuration_cxrbert import CXRBertConfig
from health_multimodal.text.model.modelling_cxrbert import CXRBertModel


def test_model_instantiation() -> None:

    def _test_model_forward(model: CXRBertModel) -> None:
        batch_size = 2
        seq_length = 5
        input_ids = torch.randint(0, 512, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        outputs = model(input_ids, attention_mask, output_cls_projected_embedding=True)
        assert outputs.last_hidden_state.shape == (batch_size, seq_length, config.hidden_size)
        assert outputs.cls_projected_embedding.shape == (batch_size, config.projection_size)

        projected_embeddings = model.get_projected_text_embeddings(input_ids, attention_mask)
        assert projected_embeddings.shape == (batch_size, config.projection_size)
        norm = torch.norm(projected_embeddings[0], p=2).item()
        assert pytest.approx(norm) == 1

        outputs = model(input_ids, attention_mask, output_hidden_states=False)
        assert outputs.hidden_states is None

        outputs_in_tuple = model(input_ids, attention_mask, return_dict=False)
        assert outputs.cls_projected_embedding == outputs_in_tuple[2]
        assert torch.allclose(outputs.last_hidden_state, outputs_in_tuple[0])

    config = CXRBertConfig(hidden_size=6,
                           projection_size=4,
                           num_hidden_layers=1,
                           num_attention_heads=2,
                           output_attentions=True,
                           return_dict=True)
    model = CXRBertModel(config)
    model = model.eval()
    _test_model_forward(model=model)

    # Try saving this model and check the saved model/config
    with TemporaryDirectory() as save_dir_as_str:
        save_dir = Path(save_dir_as_str)
        model.save_pretrained(save_dir)

        weights_file = save_dir / WEIGHTS_NAME
        assert weights_file.exists()
        saved_weights = torch.load(weights_file)
        # Make sure the MLM head was saved
        assert "cls.predictions.bias" in saved_weights
        # Make sure the project head was saved
        assert "cls_projection_head.dense_to_hidden.weight" in saved_weights

        # Check the config file
        config_file = save_dir / CONFIG_NAME
        assert config_file.exists()
        with config_file.open() as f:
            config_json = json.load(f)
        assert "projection_size" in config_json
        assert config_json["projection_size"] == config.projection_size
        assert config_json["architectures"] == ["CXRBertModel"]

        # Make sure we can load from the saved model
        model_from_pretrained = CXRBertModel.from_pretrained(save_dir)
        _test_model_forward(model=model_from_pretrained)
