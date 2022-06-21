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
from typing import Any
from pathlib import Path

from health_ml.networks.layers.attention_layers import AttentionLayer

from histopathology.datamodules.base_module import TilesDataModule
from histopathology.datamodules.tcga_crck_module import TcgaCrckTilesDataModule
from histopathology.models.encoders import (
    HistoSSLEncoder,
    ImageNetEncoder,
    ImageNetSimCLREncoder,
    SSLEncoder,
)
from histopathology.configs.classification.BaseMIL import BaseMILTiles
from histopathology.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDataset


class DeepSMILECrck(BaseMILTiles):
    def __init__(self, **kwargs: Any) -> None:
        # Define dictionary with default params that can be overridden from subclasses or CLI
        default_kwargs = dict(
            # declared in BaseMIL:
            pool_type=AttentionLayer.__name__,
            num_transformer_pool_layers=4,
            num_transformer_pool_heads=4,
            encoding_chunk_size=60,
            is_finetune=False,
            is_caching=True,
            num_top_slides=0,
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

    def setup(self) -> None:
        if self.encoder_type == SSLEncoder.__name__:
            from histopathology.configs.run_ids import innereye_ssl_checkpoint_crck_4ws
            self.downloader = self.download_ssl_checkpoint(innereye_ssl_checkpoint_crck_4ws)
        super().setup()

    def get_data_module(self) -> TilesDataModule:
        return TcgaCrckTilesDataModule(
            root_path=self.local_datasets[0],
            max_bag_size=self.max_bag_size,
            batch_size=self.batch_size,
            max_bag_size_inf=self.max_bag_size_inf,
            transforms_dict=self.get_transforms_dict(TcgaCrck_TilesDataset.IMAGE_COLUMN),
            cache_mode=self.cache_mode,
            precache_location=self.precache_location,
            cache_dir=self.cache_dir,
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=self.get_dataloader_kwargs(),
            seed=self.get_effective_random_seed(),
        )


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
