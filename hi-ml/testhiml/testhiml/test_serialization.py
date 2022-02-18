#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import torch

from health_azure import object_to_yaml
from health_ml.utils.serialization import ModelInfo


class MyTestModule(torch.nn.Module):
    def forward(self, input: torch.Tensor):
        return torch.max(input)


@dataclass
class MyModelConfig:
    foo: str
    bar: float


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
    info1 = ModelInfo(model=model,
                      model_example_input=example_inputs,
                      model_config=model_config,
                      git_repository="repo",
                      git_commit_hash="hash",
                      dataset_name="dataset",
                      azure_ml_workspace="workspace",
                      azure_ml_run_id="run_id",
                      image_dimensions="dimensions"
                      )

    state_dict = torch_save_and_load(model.state_dict())
    info2 = ModelInfo()
    info2.load_state_dict(state_dict)
    assert isinstance(info2.model, torch.jit.ScriptModule)
    assert torch.allclose(info2.model_example_input, info1.model_example_input, atol=0, rtol=0)
    serialized_output = info2.model.forward(info2.model_example_input)
    assert torch.allclose(serialized_output, model_output)
    assert info2.model_config == object_to_yaml(model_config)
    assert info2.git_repository == "repo"
    assert info2.git_commit_hash == "hash"
    assert info2.dataset_name == "dataset"
    assert info2.azure_ml_workspace == "workspace"
    assert info2.azure_ml_run_id == "run_id"
    assert info2.image_dimensions == "dimensions"
