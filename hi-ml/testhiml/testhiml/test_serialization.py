#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional
from unittest import mock

import torch
from torchvision.transforms import Compose, Resize, CenterCrop
from azureml.core import Run

from health_azure import object_to_yaml, create_aml_run_object
from health_azure.himl import effective_experiment_name
from health_azure.utils import is_running_in_azure_ml
from health_ml.utils.serialization import ModelInfo
from testazure.utils_testazure import DEFAULT_WORKSPACE


class MyTestModule(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.max(input)


@dataclass
class MyModelConfig:
    foo: str
    bar: float


class MyTokenizer:
    def __init__(self) -> None:
        self.token = torch.tensor([3.14])

    def tokenize(self, input: Any) -> torch.Tensor:
        return self.token


def torch_save_and_load(o: Any) -> Any:
    """
    Writes the given object via torch.save, and then loads it back in.
    """
    stream = BytesIO()
    torch.save(o, stream)
    stream.seek(0)
    return torch.load(stream)


def test_serialization_roundtrip() -> None:
    """
    Test that the core Torch model can be serialized and deserialized via torch.save/load.
    """
    model = MyTestModule()
    example_inputs = torch.randn((2, 3))
    model_output = model.forward(example_inputs)
    model_config = MyModelConfig(foo="foo", bar=3.14)
    tokenizer = MyTokenizer()
    image_preprocessing = Compose([Resize(size=20), CenterCrop(size=10)])
    example_image = torch.randn((3, 30, 30))
    image_output = image_preprocessing(example_image)
    other_info = b'\x01\x02'
    other_description = "a byte array"
    info1 = ModelInfo(
        model=model,
        model_example_input=example_inputs,
        model_config=model_config,
        text_tokenizer=tokenizer,
        git_repository="repo",
        git_commit_hash="hash",
        dataset_name="dataset",
        azure_ml_workspace="workspace",
        azure_ml_run_id="run_id",
        image_dimensions="dimensions",
        image_pre_processing=image_preprocessing,
        other_info=other_info,
        other_description=other_description,
    )

    state_dict = torch_save_and_load(info1.state_dict())
    info2 = ModelInfo()
    info2.load_state_dict(state_dict)
    # Test if the deserialized model gives the same output as the original model
    assert isinstance(info2.model, torch.jit.ScriptModule)
    assert info1.model_example_input is not None
    assert info2.model_example_input is not None
    assert torch.allclose(info2.model_example_input, info1.model_example_input, atol=0, rtol=0)
    serialized_output = info2.model.forward(info2.model_example_input)
    assert torch.allclose(serialized_output, model_output, atol=0, rtol=0)
    # Tokenizer should be written as a byte stream
    assert isinstance(state_dict[ModelInfo.TEXT_TOKENIZER], bytes)
    assert isinstance(state_dict[ModelInfo.IMAGE_PRE_PROCESSING], bytes)
    assert info2.model_config == object_to_yaml(model_config)
    assert info2.git_repository == "repo"
    assert info2.git_commit_hash == "hash"
    assert info2.dataset_name == "dataset"
    assert info2.azure_ml_workspace == "workspace"
    assert info2.azure_ml_run_id == "run_id"
    assert info2.image_dimensions == "dimensions"
    assert info2.other_info == other_info
    assert info2.other_description == other_description
    # Test if the deserialized preprocessing gives the same as the original object
    assert info2.image_pre_processing is not None
    image_output2 = info2.image_pre_processing(example_image)
    assert torch.allclose(image_output, image_output2, atol=0, rtol=0)


def test_get_metadata() -> None:
    """Test if model metadata is read correctly from the AzureML run."""
    run_name = "foo"
    experiment_name = effective_experiment_name("himl-tests")
    run: Optional[Run] = None
    try:
        run = create_aml_run_object(
            experiment_name=experiment_name, run_name=run_name, workspace=DEFAULT_WORKSPACE.workspace
        )
        assert is_running_in_azure_ml(aml_run=run)
        # This ModelInfo object has no fields pre-set
        model_info = ModelInfo()
        # If AzureML run info is already present in the object, those fields should be preserved.
        model_info2 = ModelInfo(azure_ml_run_id="foo", azure_ml_workspace="bar")
        with mock.patch("health_ml.utils.serialization.RUN_CONTEXT", run):
            with mock.patch("health_ml.utils.serialization.is_running_in_azure_ml", return_value=True):
                model_info.get_metadata_from_azureml()
                model_info2.get_metadata_from_azureml()
        assert model_info.azure_ml_run_id == run.id  # type: ignore
        assert model_info.azure_ml_workspace == DEFAULT_WORKSPACE.workspace.name
        assert model_info2.azure_ml_run_id == "foo"
        assert model_info2.azure_ml_workspace == "bar"
    finally:
        if run is not None:
            run.complete()
