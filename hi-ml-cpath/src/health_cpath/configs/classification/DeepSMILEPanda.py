#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Optional, Set
from health_azure.utils import is_running_in_azure_ml, create_from_matching_params
from health_cpath.configs.classification.BaseMIL import BaseMIL, BaseMILSlides, BaseMILTiles
from health_cpath.configs.run_ids import innereye_ssl_checkpoint_binary
from health_cpath.datamodules.panda_module import PandaSlidesDataModule, PandaTilesDataModule
from health_cpath.datasets.default_paths import PANDA_5X_TILES_DATASET_ID, PANDA_DATASET_ID
from health_cpath.datasets.panda_dataset import PandaDataset
from health_cpath.datasets.panda_tiles_dataset import PandaTilesDataset
from health_cpath.models.encoders import HistoSSLEncoder, ImageNetSimCLREncoder, Resnet18, SSLEncoder
from health_cpath.preprocessing.loading import LoadingParams, ROIType, WSIBackend
from health_cpath.utils.naming import PlotOption, SlideKey
from health_cpath.utils.wsi_utils import TilingParams
from health_ml.networks.layers.attention_layers import AttentionLayer
from health_ml.utils.checkpoint_utils import CheckpointParser


class BaseDeepSMILEPanda(BaseMIL):
    """Base class for DeepSMILEPanda common configs between tiles and slides piplines."""

    def __init__(self, **kwargs: Any) -> None:
        default_kwargs = dict(
            # declared in BaseMIL:
            pool_type=AttentionLayer.__name__,
            num_transformer_pool_layers=4,
            num_transformer_pool_heads=4,
            tune_encoder=False,
            # average number of tiles is 56 for PANDA
            encoding_chunk_size=60,
            max_bag_size=56,
            max_bag_size_inf=0,
            # declared in TrainerParams:
            max_epochs=200,
            # declared in OptimizerParams:
            l_rate=5e-4,
            weight_decay=1e-4,
            adam_betas=(0.9, 0.99),
            # loading params:
            backend=WSIBackend.CUCIM,
            roi_type=ROIType.WHOLE,
            image_key=SlideKey.IMAGE,
            mask_key=SlideKey.MASK,
        )
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)
        self.class_names = ["ISUP 0", "ISUP 1", "ISUP 2", "ISUP 3", "ISUP 4", "ISUP 5"]
        if not is_running_in_azure_ml():
            self.max_epochs = 2

    def get_test_plot_options(self) -> Set[PlotOption]:
        plot_options = super().get_test_plot_options()
        plot_options.update([PlotOption.SLIDE_THUMBNAIL, PlotOption.ATTENTION_HEATMAP])
        return plot_options


class DeepSMILETilesPanda(BaseMILTiles, BaseDeepSMILEPanda):
    """ DeepSMILETilesPanda is derived from BaseMILTiles and BaseDeepSMILEPanda to inherit common behaviors from both
    tiles basemil and panda specific configuration.

    `tune_encoder` sets the fine-tuning mode of the encoder. For fine-tuning the encoder, batch_size = 2
    runs on multiple GPUs with ~ 6:24 min/epoch (train) and ~ 00:50 min/epoch (validation).
    """

    def __init__(self, **kwargs: Any) -> None:
        default_kwargs = dict(
            # declared in BaseMILTiles:
            is_caching=False,
            batch_size=8,
            batch_size_inf=8,
            azure_datasets=[PANDA_5X_TILES_DATASET_ID, PANDA_DATASET_ID])
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def setup(self) -> None:
        BaseMILTiles.setup(self)

    def get_data_module(self) -> PandaTilesDataModule:
        return PandaTilesDataModule(
            root_path=self.local_datasets[0],
            batch_size=self.batch_size,
            batch_size_inf=self.batch_size_inf,
            max_bag_size=self.max_bag_size,
            max_bag_size_inf=self.max_bag_size_inf,
            transforms_dict=self.get_transforms_dict(PandaTilesDataset.IMAGE_COLUMN),
            cache_mode=self.cache_mode,
            precache_location=self.precache_location,
            cache_dir=self.cache_dir,
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=self.get_dataloader_kwargs(),
            seed=self.get_effective_random_seed(),
            pl_replace_sampler_ddp=self.pl_replace_sampler_ddp,
        )

    def get_slides_dataset(self) -> Optional[PandaDataset]:
        return PandaDataset(root=self.local_datasets[1])                             # type: ignore


class TilesPandaImageNetMIL(DeepSMILETilesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=Resnet18.__name__, **kwargs)


class TilesPandaImageNetSimCLRMIL(DeepSMILETilesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class TilesPandaSSLMIL(DeepSMILETilesPanda):
    def __init__(self, **kwargs: Any) -> None:
        # If no SSL checkpoint is provided, use the default one
        self.ssl_checkpoint = self.ssl_checkpoint or CheckpointParser(innereye_ssl_checkpoint_binary)
        super().__init__(encoder_type=SSLEncoder.__name__, **kwargs)


class TilesPandaHistoSSLMIL(DeepSMILETilesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)


class DeepSMILESlidesPanda(BaseMILSlides, BaseDeepSMILEPanda):
    """DeepSMILESlidesPanda is derived from BaseMILSlides and BaseDeeppSMILEPanda to inherits common behaviors from both
    slides basemil and panda specific configuration.
    """

    def __init__(self, **kwargs: Any) -> None:
        default_kwargs = dict(
            level=1,
            tile_size=224,
            background_val=255,
            azure_datasets=[PANDA_DATASET_ID],)
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def setup(self) -> None:
        BaseMILSlides.setup(self)

    def get_dataloader_kwargs(self) -> dict:
        return dict(
            multiprocessing_context="spawn",
            **super().get_dataloader_kwargs()
        )

    def get_data_module(self) -> PandaSlidesDataModule:
        return PandaSlidesDataModule(
            root_path=self.local_datasets[0],
            batch_size=self.batch_size,
            batch_size_inf=self.batch_size_inf,
            max_bag_size=self.max_bag_size,
            max_bag_size_inf=self.max_bag_size_inf,
            tiling_params=create_from_matching_params(self, TilingParams),
            loading_params=create_from_matching_params(self, LoadingParams),
            seed=self.get_effective_random_seed(),
            transforms_dict=self.get_transforms_dict(PandaDataset.IMAGE_COLUMN),
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=self.get_dataloader_kwargs(),
            pl_replace_sampler_ddp=self.pl_replace_sampler_ddp,
        )

    def get_slides_dataset(self) -> PandaDataset:
        return PandaDataset(root=self.local_datasets[0])                             # type: ignore


class SlidesPandaImageNetMIL(DeepSMILESlidesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=Resnet18.__name__, **kwargs)


class SlidesPandaImageNetSimCLRMIL(DeepSMILESlidesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class SlidesPandaSSLMIL(DeepSMILESlidesPanda):
    def __init__(self, **kwargs: Any) -> None:
        # If no SSL checkpoint is provided, use the default one
        self.ssl_checkpoint = self.ssl_checkpoint or CheckpointParser(innereye_ssl_checkpoint_binary)
        super().__init__(encoder_type=SSLEncoder.__name__, **kwargs)


class SlidesPandaHistoSSLMIL(DeepSMILESlidesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)
