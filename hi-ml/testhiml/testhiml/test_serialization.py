#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from io import BytesIO
from typing import Any

import torch

from health_ml.utils.serialization import ModelInfo


class MyTestModule(torch.nn.Module):
    def forward(self, input: torch.Tensor):
        return torch.max(input)


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
    model_info = ModelInfo(model=model,
                           model_example_input=example_inputs,
                           )

    state_dict = torch_save_and_load(model.state_dict())
    model_info2 = ModelInfo()
    model_info2.load_state_dict(state_dict)
    assert isinstance(model_info2.model, torch.jit.ScriptModule)
    assert torch.allclose(model_info2.model_example_input, model_info.model_example_input, atol=0, rtol=0)
    serialized_output = model_info2.model.forward(model_info2.model_example_input)
    assert torch.allclose(serialized_output, model_output)
