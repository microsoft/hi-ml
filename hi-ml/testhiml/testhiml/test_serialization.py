#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from io import BytesIO

import torch

from health_ml.utils.serialization import ModelInfo


class MyTestModule(torch.nn.Module):
    def forward(self, input: torch.Tensor):
        return torch.max(input)


def test_serialization_roundtrip() -> None:
    model = MyTestModule()
    example_inputs = torch.randn((2, 3))
    model_output = model.forward(example_inputs)
    model_info = ModelInfo(model=model,
                           model_example_input=example_inputs,
                           )

    stream = BytesIO()
    torch.save(model_info.state_dict(), stream)
    stream.seek(0)
    state_dict = torch.load(stream)
    model_info2 = ModelInfo()
    model_info2.load_state_dict(state_dict)
    assert isinstance(model_info.model, torch.jit.ScriptModule)
    assert model_info.model_example_input == model_info2.model_example_input
    serialized_output = model_info2.model.forward(model_info2.model_example_input)
    assert torch.allclose(serialized_output, model_output)
