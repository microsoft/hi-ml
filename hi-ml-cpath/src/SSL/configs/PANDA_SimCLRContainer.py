#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

from SSL.lightning_containers.ssl_container import EncoderName, SSLContainer
from SSL.utils import SSLTrainingType
from health_azure.utils import is_running_in_azure_ml
from health_cpath.datasets.panda_tiles_dataset import PandaTilesDatasetWithReturnIndex
from health_cpath.datasets.default_paths import PANDA_5X_TILES_DATASET_ID
from SSL.configs.HistoSimCLRContainer import HistoSSLContainer


SSL_Dataset_PANDA = "PandaTilesDataset"


class PANDA_SimCLR(HistoSSLContainer):
    """
    Config to train SSL model on Panda tiles dataset.
    Augmentation can be configured by using a configuration yml file or by specifying the set of transformations
    in the _get_transforms method.
    It has been tested on a toy local dataset (2 slides) and on AML on (~25 slides).
    """
    SSLContainer.DatasetToClassMapping.update({SSL_Dataset_PANDA: PandaTilesDatasetWithReturnIndex})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(ssl_training_dataset_name=SSL_Dataset_PANDA,
                         linear_head_dataset_name=SSL_Dataset_PANDA,
                         azure_datasets=[PANDA_5X_TILES_DATASET_ID],
                         random_seed=1,
                         num_workers=5,
                         is_debug_model=False,
                         model_checkpoint_save_interval=50,
                         model_checkpoints_save_last_k=3,
                         model_monitor_metric='ssl_online_evaluator/val/AccuracyAtThreshold05',
                         model_monitor_mode='max',
                         max_epochs=200,
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
        PandaTilesDatasetWithReturnIndex.random_subset_fraction = 1
        if not is_running_in_azure_ml():
            self.is_debug_model = True
            self.num_workers = 0
            self.max_epochs = 2
