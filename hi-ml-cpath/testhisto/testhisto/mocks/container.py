#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Optional, Set

from health_ml.networks.layers.attention_layers import AttentionLayer
from health_cpath.configs.classification.DeepSMILEPanda import DeepSMILESlidesPanda, DeepSMILETilesPanda
from health_cpath.datasets.panda_dataset import PandaDataset
from health_cpath.models.encoders import ImageNetEncoder
from health_cpath.datamodules.base_module import CacheMode, CacheLocation
from health_cpath.utils.naming import PlotOption


class MockDeepSMILETilesPanda(DeepSMILETilesPanda):
    def __init__(self, tmp_path: Path, **kwargs: Any) -> None:
        default_kwargs = dict(
            # Model parameters:
            pool_type=AttentionLayer.__name__,
            pool_hidden_dim=16,
            num_transformer_pool_layers=1,
            num_transformer_pool_heads=1,
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
            ssl_checkpoint_run_id=""
        )
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)
        self.tmp_path = tmp_path

    @property
    def cache_dir(self) -> Path:
        return Path(self.tmp_path / f"himl_cache/{self.__class__.__name__}-{self.encoder_type}/")

    def get_slides_dataset(self) -> Optional[PandaDataset]:
        return None

    def get_test_plot_options(self) -> Set[PlotOption]:
        return {PlotOption.HISTOGRAM}


class MockDeepSMILESlidesPanda(DeepSMILESlidesPanda):
    def __init__(self, tmp_path: Path, **kwargs: Any) -> None:
        default_kwargs = dict(
            # Model parameters:
            pool_type=AttentionLayer.__name__,
            pool_hidden_dim=16,
            num_transformer_pool_layers=1,
            num_transformer_pool_heads=1,
            class_names=["ISUP 0", "ISUP 1", "ISUP 2", "ISUP 3", "ISUP 4", "ISUP 5"],
            # Encoder parameters
            encoder_type=ImageNetEncoder.__name__,
            tile_size=28,
            # Data Module parameters
            batch_size=2,
            encoding_chunk_size=4,
            level=0,
            max_bag_size=4,
            max_bag_size_inf=0,
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
        return Path(self.tmp_path / f"himl_cache/{self.__class__.__name__}-{self.encoder_type}/")

    def get_test_plot_options(self) -> Set[PlotOption]:
        return {PlotOption.HISTOGRAM}
