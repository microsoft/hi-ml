#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""DeepSMILECrck is the container for experiments relating to DeepSMILE using the TCGA-CRCk dataset.

For convenience, this module also defines encoder-specific containers e.g.TcgaCrckImageNetMIL

Reference:
- Schirris (2021). DeepSMILE: Self-supervised heterogeneity-aware multiple instance learning for DNA
damage response defect classification directly from H&E whole-slide images. arXiv:2107.09405
"""
from typing import Any, List
from pathlib import Path
import os
from monai.transforms import Compose
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
import torch

from health_azure.utils import CheckpointDownloader
from health_azure import get_workspace
from health_ml.networks.layers.attention_layers import AttentionLayer
from health_ml.utils import fixed_paths
from histopathology.datamodules.base_module import CacheMode, CacheLocation
from histopathology.datamodules.base_module import TilesDataModule
from histopathology.datamodules.tcga_crck_module import TcgaCrckTilesDataModule
from health_ml.utils.checkpoint_utils import get_best_checkpoint_path

from histopathology.models.transforms import (
    EncodeTilesBatchd,
    LoadTilesBatchd,
)
from histopathology.models.encoders import (
    HistoSSLEncoder,
    ImageNetEncoder,
    ImageNetSimCLREncoder,
    SSLEncoder,
)

from histopathology.configs.classification.BaseMIL import BaseMIL
from histopathology.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDataset


class DeepSMILECrck(BaseMIL):
    def __init__(self, **kwargs: Any) -> None:
        # Define dictionary with default params that can be overridden from subclasses or CLI
        default_kwargs = dict(
            # declared in BaseMIL:
            pool_type=AttentionLayer.__name__,
            num_transformer_pool_layers=4,
            num_transformer_pool_heads=4,
            encoding_chunk_size=60,
            cache_mode=CacheMode.MEMORY,
            precache_location=CacheLocation.CPU,
            # declared in DatasetParams:
            local_datasets=[Path("/tmp/datasets/TCGA-CRCk")],
            azure_datasets=["TCGA-CRCk"],
            # declared in TrainerParams:
            max_epochs=50,
            # declared in WorkflowParams:
            crossval_count=5,
            # declared in OptimizerParams:
            l_rate=5e-4,
            weight_decay=1e-4,
            adam_betas=(0.9, 0.99),
        )
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

        self.best_checkpoint_filename = "checkpoint_max_val_auroc"
        self.best_checkpoint_filename_with_suffix = (
            self.best_checkpoint_filename + ".ckpt"
        )
        self.checkpoint_folder_path = "outputs/checkpoints/"

        best_checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_folder_path,
            monitor="val/auroc",
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
            from histopathology.configs.run_ids import innereye_ssl_checkpoint_crck_4ws
            self.downloader = CheckpointDownloader(
                aml_workspace=get_workspace(),
                run_id=innereye_ssl_checkpoint_crck_4ws,
                checkpoint_filename="last.ckpt",
                download_dir="outputs/",
                remote_checkpoint_dir=Path("outputs/checkpoints")
            )
            os.chdir(fixed_paths.repository_root_directory().parent)
            self.downloader.download_checkpoint_if_necessary()

        self.encoder = self.get_encoder()
        self.encoder.cuda()
        self.encoder.eval()

    def get_data_module(self) -> TilesDataModule:
        image_key = TcgaCrck_TilesDataset.IMAGE_COLUMN
        if self.is_finetune:
            transform = LoadTilesBatchd(image_key, progress=True)
            num_cpus = os.cpu_count()
            assert num_cpus is not None  # for mypy
            workers_per_gpu = num_cpus // torch.cuda.device_count()
            dataloader_kwargs = dict(num_workers=workers_per_gpu, pin_memory=True)
        else:
            transform = Compose([
                LoadTilesBatchd(image_key, progress=True),
                EncodeTilesBatchd(image_key, self.encoder, chunk_size=self.encoding_chunk_size)
            ])
            dataloader_kwargs = dict(num_workers=0, pin_memory=False)

        return TcgaCrckTilesDataModule(
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
            dataloader_kwargs=dataloader_kwargs,
        )

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


class TcgaCrckImageNetMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetEncoder.__name__, **kwargs)


class TcgaCrckImageNetSimCLRMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class TcgaCrckSSLMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=SSLEncoder.__name__, **kwargs)


class TcgaCrckHistoSSLMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)
