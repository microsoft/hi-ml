#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from random import randint
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from SSL.data.cxr_datasets import RSNAKaggleCXR
from SSL.lightning_modules.byol.byol_module import BootstrapYourOwnLatent
from SSL.lightning_modules.byol.byol_moving_average import ByolMovingAverageWeightUpdate

from testSSL.utils import TEST_OUTPUTS_PATH


@pytest.mark.fast
def test_update_tau() -> None:
    class DummyRSNADataset(RSNAKaggleCXR):
        def __getitem__(self, item: Any) -> Any:
            return (torch.rand([3, 224, 224], dtype=torch.float32),
                    torch.rand([3, 224, 224], dtype=torch.float32)
                    ), randint(0, 1)
    path_to_cxr_test_dataset = TEST_OUTPUTS_PATH / "cxr_test_dataset"
    dataset_dir = str(path_to_cxr_test_dataset)
    dummy_rsna_train_dataloader: DataLoader = torch.utils.data.DataLoader(
        DummyRSNADataset(root=dataset_dir, return_index=False, train=True),
        batch_size=20,
        num_workers=0,
        drop_last=True)

    initial_tau = 0.99
    byol_weight_update = ByolMovingAverageWeightUpdate(initial_tau=initial_tau)
    trainer = Trainer(max_epochs=5)
    trainer.train_dataloader = dummy_rsna_train_dataloader  # type: ignore
    total_steps = len(trainer.train_dataloader) * trainer.max_epochs  # type: ignore
    global_step = 15
    byol_module = BootstrapYourOwnLatent(num_samples=16,
                                         learning_rate=1e-3,
                                         batch_size=4,
                                         encoder_name="resnet50",
                                         warmup_epochs=10,
                                         max_epochs=100)
    with mock.patch("SSL.lightning_modules.byol.byol_module.BootstrapYourOwnLatent.global_step", global_step):
        new_tau = byol_weight_update.update_tau(pl_module=byol_module, trainer=trainer)
    expected_tau = 1 - (1 - initial_tau) * (math.cos(math.pi * global_step / total_steps) + 1) / 2
    assert new_tau == expected_tau


@pytest.mark.fast
def test_update_weights() -> None:
    online_network = torch.nn.Linear(in_features=3, out_features=1, bias=False)
    target_network = torch.nn.Linear(in_features=3, out_features=1, bias=False)
    byol_weight_update = ByolMovingAverageWeightUpdate(initial_tau=0.9)
    old_target_net_weight = target_network.weight.data.numpy().copy()
    byol_weight_update.update_weights(online_network, target_network)
    assert np.isclose(target_network.weight.data.numpy(),
                      0.9 * old_target_net_weight + 0.1 * online_network.weight.data.numpy()).all()
