#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import py
import os
import torch

from pathlib import Path
from typing import Any, List, Union

from monai.transforms import Compose
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from health_ml.utils import fixed_paths
from health_ml.utils.checkpoint_utils import get_best_checkpoint_path
from health_ml.networks.layers.attention_layers import AttentionLayer

from histopathology.configs.classification.BaseMIL import BaseMIL
from histopathology.models.encoders import ImageNetEncoder
from histopathology.datamodules.base_module import CacheMode, CacheLocation
from histopathology.models.transforms import EncodeTilesBatchd, LoadTilesBatchd

from testhisto.mocks.tiles_datamodule import MockTilesDataModule, MockTilesDataset


class MockDeepSMILE(BaseMIL):
    def __init__(self, tmp_path: Union[py.path.local, Path], **kwargs: Any) -> None:
        default_kwargs = dict(
            # Model parameters:
            pool_type=AttentionLayer.__name__,
            pool_hidden_dim=16,
            num_transformer_pool_layers=1,
            num_transformer_pool_heads=1,
            is_finetune=False,
            class_names=["ISUP 0", "ISUP 1", "ISUP 2", "ISUP 3", "ISUP 4", "ISUP 5"],
            # Encoder parameters
            encoder_type=ImageNetEncoder.__name__,
            tile_size=28,
            # Data Module parameters
            batch_size=2,
            max_bag_size=4,
            max_bag_size_inf=4,
            encoding_chunk_size=4,
            cache_mode=CacheMode.NONE,
            precache_location=CacheLocation.NONE,
            # declared in DatasetParams:
            local_datasets=[Path(tmp_path)],
            # declared in TrainerParams:
            max_epochs=2,
            crossval_count=1,
            # declared in OptimizerParams:
            l_rate=5e-4,
            weight_decay=1e-4,
            adam_betas=(0.9, 0.99),
        )
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

        self.tmp_path = tmp_path
        self.best_checkpoint_filename = "checkpoint_max_val_auroc"
        self.best_checkpoint_filename_with_suffix = self.best_checkpoint_filename + ".ckpt"
        self.checkpoint_folder_path = tmp_path / "outputs/checkpoints/"
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
        return Path(self.tmp_path / f"innereye_cache1/{self.__class__.__name__}-{self.encoder_type}/")

    def get_data_module(self) -> MockTilesDataModule:
        image_key = MockTilesDataset.IMAGE_COLUMN
        if self.is_finetune:
            transform = LoadTilesBatchd(image_key, progress=True)
            num_cpus = os.cpu_count()
            assert num_cpus is not None  # for mypy
            workers_per_gpu = num_cpus // torch.cuda.device_count()
            dataloader_kwargs = dict(num_workers=workers_per_gpu, pin_memory=True)
        else:
            transform = Compose(
                [
                    LoadTilesBatchd(image_key, progress=True),
                    EncodeTilesBatchd(image_key, self.encoder, chunk_size=self.encoding_chunk_size),
                ]
            )
            dataloader_kwargs = dict(num_workers=0, pin_memory=False)

        return MockTilesDataModule(
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
        absolute_checkpoint_path = Path(
            fixed_paths.repository_root_directory(),
            self.checkpoint_folder_path,
            self.best_checkpoint_filename_with_suffix,
        )
        if absolute_checkpoint_path.is_file():
            return absolute_checkpoint_path

        absolute_checkpoint_path_parent = Path(
            fixed_paths.repository_root_directory().parent,
            self.checkpoint_folder_path,
            self.best_checkpoint_filename_with_suffix,
        )
        if absolute_checkpoint_path_parent.is_file():
            return absolute_checkpoint_path_parent

        checkpoint_path = get_best_checkpoint_path(Path(self.checkpoint_folder_path))
        if checkpoint_path.is_file():
            return checkpoint_path

        raise ValueError("Path to best checkpoint not found")
