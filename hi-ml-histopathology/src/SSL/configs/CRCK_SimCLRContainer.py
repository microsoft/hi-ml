from enum import Enum
from pathlib import Path
from typing import Any, Optional
import sys

from SSL.lightning_containers.ssl_container import EncoderName, SSLContainer, SSLDatasetName
from SSL.utils import SSLTrainingType
from histopathology.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDatasetWithReturnIndex
from SSL.configs.HistoSimCLRContainer import HistoSSLContainer

current_file = Path(__file__)
print(f"Running container from {current_file}")
print(f"Sys path container level {sys.path}")

local_mode = False
path_local_data: Optional[Path]
if local_mode:
    is_debug_model = True
    drop_last = False
    # This dataset has been used for test purposes on a local machine, change to your local path
    path_local_data = Path("/tmp/datasets/TCGA-CRCk")
    azure_datasets = 'Dummy'
    num_workers = 0
    num_epochs = 2
else:
    is_debug_model = False
    drop_last = False
    path_local_data = None
    azure_datasets = ["TCGA-CRCk"]
    num_workers = 12
    num_epochs = 200


class SSLDatasetNameRadiomicsNN(SSLDatasetName, Enum):
    TCGA_CRCK = "CRCKTilesDataset"


class CRCK_SimCLR(HistoSSLContainer):
    """
    Config to train SSL model on CRCK tiles dataset.
    Augmentation can be configured by using a configuration yml file or by specifying the set of transformations
    in the _get_transforms method.

    It has been tested locally and on AML on the full training dataset (93408 tiles).
    """
    SSLContainer._SSLDataClassMappings.update({SSLDatasetNameRadiomicsNN.TCGA_CRCK.value:
                                               TcgaCrck_TilesDatasetWithReturnIndex})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(ssl_training_dataset_name=SSLDatasetNameRadiomicsNN.TCGA_CRCK,
                         linear_head_dataset_name=SSLDatasetNameRadiomicsNN.TCGA_CRCK,
                        #  local_datasets=[None],
                         azure_dataset_id=["TCGA-CRCk"],
                         random_seed=1,
                         num_workers=num_workers,
                         is_debug_model=is_debug_model,
                         model_checkpoint_save_interval=50,
                         model_checkpoints_save_last_k=3,
                         model_monitor_metric='ssl_online_evaluator/val/AreaUnderRocCurve',
                         model_monitor_mode='max',
                         num_epochs=num_epochs,
                         ssl_training_batch_size=48,  # GPU memory is at 70% with batch_size=32, 2GPUs
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLTrainingType.SimCLR,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_augmentation_config=None,  # Change to path_augmentation to use the config
                         linear_head_augmentation_config=None,  # Change to path_augmentation to use the config
                         drop_last=drop_last,
                         **kwargs)
