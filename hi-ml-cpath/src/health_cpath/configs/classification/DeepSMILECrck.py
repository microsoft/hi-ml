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
from typing import Any, Set

from health_ml.networks.layers.attention_layers import AttentionLayer
from health_cpath.configs.run_ids import innereye_ssl_checkpoint_crck_4ws
from health_cpath.datamodules.base_module import TilesDataModule
from health_cpath.datamodules.tcga_crck_module import TcgaCrckTilesDataModule
from health_cpath.datasets.default_paths import TCGA_CRCK_DATASET_ID
from health_cpath.models.encoders import (
    HistoSSLEncoder,
    ImageNetSimCLREncoder,
    Resnet18,
    SSLEncoder,
)
from health_cpath.configs.classification.BaseMIL import BaseMILTiles
from health_cpath.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDataset
from health_cpath.utils.naming import PlotOption
from health_ml.utils.checkpoint_utils import CheckpointParser


class DeepSMILECrck(BaseMILTiles):
    def __init__(self, **kwargs: Any) -> None:
        # Define dictionary with default params that can be overridden from subclasses or CLI
        default_kwargs = dict(
            # declared in BaseMIL:
            pool_type=AttentionLayer.__name__,
            num_transformer_pool_layers=4,
            num_transformer_pool_heads=4,
            encoding_chunk_size=60,
            tune_encoder=False,
            is_caching=True,
            num_top_slides=0,
            azure_datasets=[TCGA_CRCK_DATASET_ID],
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
        super().setup()

    def get_data_module(self) -> TilesDataModule:
        return TcgaCrckTilesDataModule(
            root_path=self.local_datasets[0],
            max_bag_size=self.max_bag_size,
            batch_size=self.batch_size,
            batch_size_inf=self.batch_size_inf,
            max_bag_size_inf=self.max_bag_size_inf,
            transforms_dict=self.get_transforms_dict(TcgaCrck_TilesDataset.IMAGE_COLUMN),
            cache_mode=self.cache_mode,
            precache_location=self.precache_location,
            cache_dir=self.cache_dir,
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=self.get_dataloader_kwargs(),
            seed=self.get_effective_random_seed(),
            pl_replace_sampler_ddp=self.pl_replace_sampler_ddp,
        )

    def get_test_plot_options(self) -> Set[PlotOption]:
        plot_options = super().get_test_plot_options()
        plot_options.add(PlotOption.PR_CURVE)
        return plot_options


class TcgaCrckImageNetMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=Resnet18.__name__, **kwargs)


class TcgaCrckImageNetSimCLRMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class TcgaCrckSSLMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        # If no SSL checkpoint is provided, use the default one
        self.ssl_checkpoint = self.ssl_checkpoint or CheckpointParser(innereye_ssl_checkpoint_crck_4ws)
        super().__init__(encoder_type=SSLEncoder.__name__, **kwargs)


class TcgaCrckHistoSSLMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)
