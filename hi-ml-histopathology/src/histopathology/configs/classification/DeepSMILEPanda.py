#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, List
from pathlib import Path
import os
from monai.transforms import Compose
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

from health_azure.utils import CheckpointDownloader
from health_azure.utils import get_workspace, is_running_in_azure_ml
from health_ml.networks.layers.attention_layers import AttentionLayer
from health_ml.utils import fixed_paths
from histopathology.datamodules.base_module import CacheMode, CacheLocation
from histopathology.datamodules.panda_module import PandaTilesDataModule
from histopathology.datasets.panda_tiles_dataset import PandaTilesDataset
from health_ml.utils.checkpoint_utils import get_best_checkpoint_path

from histopathology.models.encoders import (
    HistoSSLEncoder,
    ImageNetEncoder,
    ImageNetSimCLREncoder,
    SSLEncoder,
)
from histopathology.models.transforms import (
    EncodeTilesBatchd,
    LoadTilesBatchd,
)

from histopathology.configs.classification.BaseMIL import BaseMIL
from histopathology.datasets.panda_dataset import PandaDataset


class DeepSMILEPanda(BaseMIL):
    """`is_finetune` sets the fine-tuning mode. If this is set, setting cache_mode=CacheMode.NONE takes ~30 min/epoch and
    cache_mode=CacheMode.MEMORY, precache_location=CacheLocation.CPU takes ~[5-10] min/epoch.
    Fine-tuning with caching completes using batch_size=4, max_bag_size=1000, max_epochs=20, max_num_gpus=1 on PANDA.
    """
    def __init__(self, **kwargs: Any) -> None:
        default_kwargs = dict(
            # declared in BaseMIL:
            pool_type=AttentionLayer.__name__,
            num_transformer_pool_layers=4,
            num_transformer_pool_heads=4,
            # average number of tiles is 56 for PANDA
            encoding_chunk_size=60,
            cache_mode=CacheMode.MEMORY,
            precache_location=CacheLocation.CPU,
            is_finetune=False,

            # declared in DatasetParams:
            local_datasets=[Path("/tmp/datasets/PANDA_tiles"), Path("/tmp/datasets/PANDA")],
            azure_datasets=["PANDA_tiles", "PANDA"],
            # To mount the dataset instead of downloading in AML, pass --use_dataset_mount in the CLI
            # declared in TrainerParams:
            max_epochs=200,
            # use_mixed_precision = True,
            # declared in WorkflowParams:
            crossval_count=1,
            # declared in OptimizerParams:
            l_rate=5e-4,
            weight_decay=1e-4,
            adam_betas=(0.9, 0.99))
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)
        self.class_names = ["ISUP 0", "ISUP 1", "ISUP 2", "ISUP 3", "ISUP 4", "ISUP 5"]
        if not is_running_in_azure_ml():
            self.max_epochs = 1
        self.best_checkpoint_filename = "checkpoint_max_val_auroc"
        self.best_checkpoint_filename_with_suffix = (
            self.best_checkpoint_filename + ".ckpt"
        )
        self.checkpoint_folder_path = "outputs/checkpoints/"
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_folder_path,
            monitor="val/accuracy",
            filename=self.best_checkpoint_filename,
            auto_insert_metric_name=False,
            mode="max",
        )
        self.callbacks = best_checkpoint_callback

    @property
    def cache_dir(self) -> Path:
        return Path(
            f"/tmp/innereye_cache1/{self.__class__.__name__}-{self.encoder_type}/"
        )

    def setup(self) -> None:
        if self.encoder_type == SSLEncoder.__name__:
            from histopathology.configs.run_ids import innereye_ssl_checkpoint_binary
            self.downloader = CheckpointDownloader(
                aml_workspace=get_workspace(),
                run_id=innereye_ssl_checkpoint_binary,  # innereye_ssl_checkpoint
                checkpoint_filename="best_checkpoint.ckpt",  # "last.ckpt",
                download_dir="outputs/",
                remote_checkpoint_dir=Path("outputs/checkpoints")
            )
            os.chdir(fixed_paths.repository_root_directory().parent)
            self.downloader.download_checkpoint_if_necessary()
        self.encoder = self.get_encoder()
        if not self.is_finetune:
            self.encoder.eval()

    def get_data_module(self) -> PandaTilesDataModule:
        image_key = PandaTilesDataset.IMAGE_COLUMN
        if self.is_finetune:
            transform = Compose([LoadTilesBatchd(image_key, progress=True)])
        else:
            transform = Compose([
                                LoadTilesBatchd(image_key, progress=True),
                                EncodeTilesBatchd(image_key, self.encoder, chunk_size=self.encoding_chunk_size)
                                ])

        return PandaTilesDataModule(
            root_path=self.local_datasets[0],
            max_bag_size=self.max_bag_size,
            batch_size=self.batch_size,
            max_bag_size_inf=self.max_bag_size_inf,
            transform=transform,
            cache_mode=self.cache_mode,
            precache_location=self.precache_location,
            cache_dir=self.cache_dir,
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
        )

    def get_slides_dataset(self) -> PandaDataset:
        return PandaDataset(root=self.local_datasets[1])                             # type: ignore

    def get_callbacks(self) -> List[Callback]:
        return super().get_callbacks() + [self.callbacks]

    def get_path_to_best_checkpoint(self) -> Path:
        """
        Returns the full path to a checkpoint file that was found to be best during training, whatever criterion
        was applied there. This is necessary since for some models the checkpoint is in a subfolder of the checkpoint
        folder.
        """
        # absolute path is required for registering the model.
        absolute_checkpoint_path = Path(fixed_paths.repository_root_directory(),
                                        self.checkpoint_folder_path,
                                        self.best_checkpoint_filename_with_suffix)
        if absolute_checkpoint_path.is_file():
            return absolute_checkpoint_path

        absolute_checkpoint_path_parent = Path(fixed_paths.repository_root_directory().parent,
                                               self.checkpoint_folder_path,
                                               self.best_checkpoint_filename_with_suffix)
        if absolute_checkpoint_path_parent.is_file():
            return absolute_checkpoint_path_parent

        checkpoint_path = get_best_checkpoint_path(Path(self.checkpoint_folder_path))
        if checkpoint_path.is_file():
            return checkpoint_path

        raise ValueError("Path to best checkpoint not found")


class PandaImageNetMIL(DeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetEncoder.__name__, **kwargs)


class PandaImageNetSimCLRMIL(DeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class PandaSSLMIL(DeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=SSLEncoder.__name__, **kwargs)


class PandaHistoSSLMIL(DeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)
