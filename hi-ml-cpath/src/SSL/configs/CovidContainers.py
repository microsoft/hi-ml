from typing import Any

from SSL.lightning_containers.ssl_container import EncoderName, SSLContainer, SSLDatasetName
from SSL.utils import SSLTrainingType
from SSL.configs.CXR_SSL_configs import NIH_AZURE_DATASET_ID, path_encoder_augmentation_cxr, \
    path_linear_head_augmentation_cxr

COVID_DATASET_ID = "id-of-your-dataset"


class NIH_COVID_BYOL(SSLContainer):
    """
    Class to train a SSL model on NIH dataset and monitor embeddings quality on a Covid Dataset.
    """

    def __init__(self,
                 covid_dataset_id: str = COVID_DATASET_ID,
                 pretraining_dataset_id: str = NIH_AZURE_DATASET_ID,
                 **kwargs: Any):
        super().__init__(ssl_training_dataset_name=SSLDatasetName.NIHCXR,
                         linear_head_dataset_name=SSLDatasetName.Covid,
                         random_seed=1,
                         max_epochs=500,
                         ssl_training_batch_size=75,  # This runs  with 16 gpus (4 nodes)
                         num_workers=12,
                         ssl_encoder=EncoderName.densenet121,
                         ssl_training_type=SSLTrainingType.BYOL,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_augmentation_config=path_encoder_augmentation_cxr,
                         # the first Azure dataset is for training, the second is for the linear head
                         azure_datasets=[pretraining_dataset_id, covid_dataset_id],
                         linear_head_augmentation_config=path_linear_head_augmentation_cxr,
                         online_evaluator_lr=1e-5,
                         linear_head_batch_size=64,
                         pl_find_unused_parameters=True,
                         **kwargs)
