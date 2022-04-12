#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from torch.nn import Module
from health_ml.utils import set_model_to_eval_mode


def test_set_to_eval_mode() -> None:
    model = Module()
    model.train(True)
    assert model.training
    with set_model_to_eval_mode(model):
        assert not model.training
    assert model.training

    model.eval()
    assert not model.training
    with set_model_to_eval_mode(model):
        assert not model.training
    assert not model.training
