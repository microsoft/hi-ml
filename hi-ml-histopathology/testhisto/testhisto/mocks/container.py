#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any

from health_ml.networks.layers.attention_layers import AttentionLayer
from histopathology.configs.classification.BaseMIL import BaseMIL
from histopathology.models.encoders import ImageNetEncoder
from histopathology.datamodules.base_module import CacheMode, CacheLocation
from testhisto.mocks.tiles_datamodule import MockTilesDataModule, MockTilesDataset


class MockDeepSMILE(BaseMIL):
    def __init__(self, tmp_path: Path, **kwargs: Any) -> None:
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
            local_datasets=[tmp_path],
            # declared in TrainerParams:
            max_epochs=2,
            crossval_count=1,
        )
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)
        self.tmp_path = tmp_path

    @property
    def cache_dir(self) -> Path:
        return Path(self.tmp_path / f"innereye_cache1/{self.__class__.__name__}-{self.encoder_type}/")

    def get_data_module(self) -> MockTilesDataModule:
        return MockTilesDataModule(
            root_path=self.local_datasets[0],
            max_bag_size=self.max_bag_size,
            batch_size=self.batch_size,
            max_bag_size_inf=self.max_bag_size_inf,
            transform=self.get_transform(MockTilesDataset.IMAGE_COLUMN),
            cache_mode=self.cache_mode,
            precache_location=self.precache_location,
            cache_dir=self.cache_dir,
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=self.get_dataloader_kwargs(),
        )
