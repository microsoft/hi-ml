#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import sys
from pathlib import Path

from SSL.lightning_containers.ssl_container import EncoderName, SSLContainer, SSLDatasetName
from SSL.lightning_containers.ssl_image_classifier import SSLClassifierContainer
from SSL.utils import SSLTrainingType

RSNA_AZURE_DATASET_ID = "rsna_pneumonia_detection_kaggle_dataset"
NIH_AZURE_DATASET_ID = "nih-training-set"

configs_path = Path(sys.modules["SSL.configs"].__path__._path[0])  # type: ignore
path_encoder_augmentation_cxr = configs_path / "cxr_ssl_encoder_augmentations.yaml"
path_linear_head_augmentation_cxr = configs_path / "cxr_linear_head_augmentations.yaml"


class NIH_RSNA_BYOL(SSLContainer):
    """
    Config to train SSL model on NIHCXR ChestXray dataset and use the RSNA Pneumonia detection Challenge dataset to
    finetune the linear head on top for performance monitoring.
    """

    def __init__(self) -> None:
        super().__init__(ssl_training_dataset_name=SSLDatasetName.NIHCXR,
                         linear_head_dataset_name=SSLDatasetName.RSNAKaggleCXR,
                         # the first Azure dataset is for training, the second is for the linear head
                         azure_datasets=[NIH_AZURE_DATASET_ID, RSNA_AZURE_DATASET_ID],
                         random_seed=1,
                         max_epochs=1000,
                         # We usually train this model with 16 GPUs, giving an effective batch size of 1200
                         ssl_training_batch_size=75,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLTrainingType.BYOL,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_augmentation_config=path_encoder_augmentation_cxr,
                         linear_head_augmentation_config=path_linear_head_augmentation_cxr)


class NIH_RSNA_SimCLR(SSLContainer):
    def __init__(self) -> None:
        super().__init__(ssl_training_dataset_name=SSLDatasetName.NIHCXR,
                         linear_head_dataset_name=SSLDatasetName.RSNAKaggleCXR,
                         # the first Azure dataset is for training, the second is for the linear head
                         azure_datasets=[NIH_AZURE_DATASET_ID, RSNA_AZURE_DATASET_ID],
                         random_seed=1,
                         max_epochs=1000,
                         # We usually train this model with 16 GPUs, giving an effective batch size of 1200
                         ssl_training_batch_size=75,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLTrainingType.SimCLR,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_augmentation_config=path_encoder_augmentation_cxr,
                         linear_head_augmentation_config=path_linear_head_augmentation_cxr)


class CXRImageClassifier(SSLClassifierContainer):
    def __init__(self) -> None:
        super().__init__(linear_head_dataset_name=SSLDatasetName.RSNAKaggleCXR,
                         random_seed=1,
                         max_epochs=200,
                         use_balanced_binary_loss_for_linear_head=True,
                         azure_datasets=[RSNA_AZURE_DATASET_ID],
                         linear_head_augmentation_config=path_linear_head_augmentation_cxr)
