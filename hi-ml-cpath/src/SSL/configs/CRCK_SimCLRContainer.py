#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

from SSL.lightning_containers.ssl_container import EncoderName, SSLContainer
from SSL.utils import SSLTrainingType
from health_cpath.datasets.default_paths import TCGA_CRCK_DATASET_ID
from health_cpath.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDatasetWithReturnIndex
from SSL.configs.HistoSimCLRContainer import HistoSSLContainer


SSL_Dataset_TCGA_CRCK = "CRCKTilesDataset"


class CRCK_SimCLR(HistoSSLContainer):
    """
    Config to train SSL model on CRCK tiles dataset.
    Augmentation can be configured by using a configuration yml file or by specifying the set of transformations
    in the _get_transforms method.
    It has been tested locally and on AML on the full training dataset (93408 tiles).
    """
    SSLContainer.DatasetToClassMapping.update({SSL_Dataset_TCGA_CRCK:
                                               TcgaCrck_TilesDatasetWithReturnIndex})

    def __init__(self, **kwargs: Any) -> None:
        # if not running in Azure ML, you may want to override certain properties on the command line, such as:
        # --is_debug_model = True
        # --num_workers = 0
        # --max_epochs = 2

        super().__init__(ssl_training_dataset_name=SSL_Dataset_TCGA_CRCK,
                         linear_head_dataset_name=SSL_Dataset_TCGA_CRCK,
                         azure_datasets=[TCGA_CRCK_DATASET_ID],
                         random_seed=1,
                         num_workers=8,
                         is_debug_model=False,
                         model_checkpoint_save_interval=50,
                         model_checkpoints_save_last_k=3,
                         model_monitor_metric='ssl_online_evaluator/val/AreaUnderRocCurve',
                         model_monitor_mode='max',
                         max_epochs=50,
                         ssl_training_batch_size=48,  # GPU memory is at 70% with batch_size=32, 2GPUs
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLTrainingType.SimCLR,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_augmentation_config=None,  # Change to path_augmentation to use the config
                         linear_head_augmentation_config=None,  # Change to path_augmentation to use the config
                         drop_last=False,
                         **kwargs)
