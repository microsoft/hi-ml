#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum
from pathlib import Path
from typing import Any
import sys

from SSL.lightning_containers.ssl_container import EncoderName, SSLContainer, SSLDatasetName
from SSL.utils import SSLTrainingType
from health_azure.utils import is_running_in_azure_ml
from histopathology.datasets.panda_tiles_dataset import PandaTilesDatasetWithReturnIndex
from histopathology.configs.SSL.HistoSimCLRContainer import HistoSSLContainer
from histopathology.datasets.default_paths import PANDA_TILES_DATASET_ID


current_file = Path(__file__)
print(f"Running container from {current_file}")
print(f"Sys path container level {sys.path}")


class SSLDatasetNameHiml(SSLDatasetName, Enum):  # type: ignore
    PANDA = "PandaTilesDataset"


class PANDA_SimCLR(HistoSSLContainer):
    """
    Config to train SSL model on Panda tiles dataset.
    Augmentation can be configured by using a configuration yml file or by specifying the set of transformations
    in the _get_transforms method.
    It has been tested on a toy local dataset (2 slides) and on AML on (~25 slides).
    """
    SSLContainer._SSLDataClassMappings.update({SSLDatasetNameHiml.PANDA.value: PandaTilesDatasetWithReturnIndex})

    def __init__(self, **kwargs: Any) -> None:
        if not is_running_in_azure_ml():
            is_debug_model = True
            num_workers = 0
            max_epochs = 2

        super().__init__(ssl_training_dataset_name=SSLDatasetNameHiml.PANDA,
                         linear_head_dataset_name=SSLDatasetNameHiml.PANDA,
                         azure_datasets=[PANDA_TILES_DATASET_ID],
                         random_seed=1,
                         num_workers=num_workers,
                         is_debug_model=is_debug_model,
                         model_checkpoint_save_interval=50,
                         model_checkpoints_save_last_k=3,
                         model_monitor_metric='ssl_online_evaluator/val/AccuracyAtThreshold05',
                         model_monitor_mode='max',
                         max_epochs=max_epochs,
                         ssl_training_batch_size=128,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLTrainingType.SimCLR,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_augmentation_config=None,  # Change to path_augmentation to use the config
                         linear_head_augmentation_config=None,  # Change to path_augmentation to use the config
                         drop_last=False,
                         **kwargs)
        self.pl_check_val_every_n_epoch = 10
        PandaTilesDatasetWithReturnIndex.occupancy_threshold = 0
