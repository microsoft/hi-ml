from enum import Enum
from pathlib import Path
from typing import Any, Optional
import sys

from SSL.lightning_containers.ssl_container import EncoderName, SSLContainer, SSLDatasetName
from SSL.utils import SSLTrainingType
from histopathology.datasets.panda_tiles_dataset import PandaTilesDatasetWithReturnIndex
from SSL.configs.HistoSimCLRContainer import HistoSSLContainer

current_file = Path(__file__)
print(f"Running container from {current_file}")
print(f"Sys path container level {sys.path}")

local_mode = False
path_local_data: Optional[Path]
if local_mode:
    is_debug_model = True
    num_workers = 0

else:
    is_debug_model = False
    num_workers = 5


class SSLDatasetNameRadiomicsNN(SSLDatasetName, Enum):
    PANDA = "PandaTilesDataset"


class PANDA_SimCLR(HistoSSLContainer):
    """
    Config to train SSL model on Panda tiles dataset.
    Augmentation can be configured by using a configuration yml file or by specifying the set of transformations
    in the _get_transforms method.
    It has been tested on a toy local dataset (2 slides) and on AML on (~25 slides).
    """
    SSLContainer._SSLDataClassMappings.update({SSLDatasetNameRadiomicsNN.PANDA.value: PandaTilesDatasetWithReturnIndex})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(ssl_training_dataset_name=SSLDatasetNameRadiomicsNN.PANDA,
                         linear_head_dataset_name=SSLDatasetNameRadiomicsNN.PANDA,
                         azure_datasets=['PANDA_tiles'],
                         random_seed=1,
                         num_workers=num_workers,
                         is_debug_model=is_debug_model,
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
