#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Optional

from health_azure.utils import is_running_in_azure_ml
from health_ml.networks.layers.attention_layers import AttentionLayer
from histopathology.configs.run_ids import innereye_ssl_checkpoint_binary
from histopathology.datamodules.panda_module import (
    PandaSlidesDataModule,
    PandaTilesDataModule)
from histopathology.datasets.panda_tiles_dataset import PandaTilesDataset
from histopathology.models.encoders import (
    HistoSSLEncoder,
    ImageNetEncoder,
    ImageNetSimCLREncoder,
    SSLEncoder)
from histopathology.configs.classification.BaseMIL import BaseMILSlides, BaseMILTiles, BaseMIL
from histopathology.datasets.panda_dataset import PandaDataset


class BaseDeepSMILEPanda(BaseMIL):
    """Base class for DeepSMILEPanda common configs between tiles and slides piplines."""
    def __init__(self, **kwargs: Any) -> None:
        default_kwargs = dict(
            # declared in BaseMIL:
            pool_type=AttentionLayer.__name__,
            num_transformer_pool_layers=4,
            num_transformer_pool_heads=4,
            is_finetune=False,
            # average number of tiles is 56 for PANDA
            encoding_chunk_size=60,
            # declared in TrainerParams:
            max_epochs=200,
            # use_mixed_precision = True,
            # declared in WorkflowParams:
            crossval_count=5,
            # declared in OptimizerParams:
            l_rate=5e-4,
            weight_decay=1e-4,
            adam_betas=(0.9, 0.99))
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)
        self.class_names = ["ISUP 0", "ISUP 1", "ISUP 2", "ISUP 3", "ISUP 4", "ISUP 5"]
        if not is_running_in_azure_ml():
            self.max_epochs = 1


class DeepSMILETilesPanda(BaseMILTiles, BaseDeepSMILEPanda):
    """ DeepSMILETilesPanda is derived from BaseMILTiles and BaseDeepSMILEPanda to inherit common behaviors from both
    tiles basemil and panda specific configuration.

    `is_finetune` sets the fine-tuning mode. `is_finetune` sets the fine-tuning mode. For fine-tuning,
    max_bag_size_inf=max_bag_size and batch_size = 2 runs on multiple GPUs with
    ~ 6:24 min/epoch (train) and ~ 00:50 min/epoch (validation).
    """
    def __init__(self, **kwargs: Any) -> None:
        default_kwargs = dict(
            # declared in BaseMILTiles:
            is_caching=False,
            # declared in DatasetParams:
            local_datasets=[Path("/tmp/datasets/PANDA_tiles"), Path("/tmp/datasets/PANDA")],
            azure_datasets=["PANDA_tiles", "PANDA"])
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def setup(self) -> None:
        if self.encoder_type == SSLEncoder.__name__:
            self.downloader = self.download_ssl_checkpoint(innereye_ssl_checkpoint_binary)
        BaseMILTiles.setup(self)

    def get_data_module(self) -> PandaTilesDataModule:
        return PandaTilesDataModule(
            root_path=self.local_datasets[0],
            max_bag_size=self.max_bag_size,
            batch_size=self.batch_size,
            max_bag_size_inf=self.max_bag_size_inf,
            transforms_dict=self.get_transforms_dict(PandaTilesDataset.IMAGE_COLUMN),
            cache_mode=self.cache_mode,
            precache_location=self.precache_location,
            cache_dir=self.cache_dir,
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=self.get_dataloader_kwargs(),
        )

    def get_slides_dataset(self) -> Optional[PandaDataset]:
        return PandaDataset(root=self.local_datasets[1])                             # type: ignore


class TilesPandaImageNetMIL(DeepSMILETilesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetEncoder.__name__, **kwargs)


class TilesPandaImageNetSimCLRMIL(DeepSMILETilesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class TilesPandaSSLMIL(DeepSMILETilesPanda):
    def __init__(self, **kwargs: Any) -> None:
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
            # declared in BaseMILSlides:
            # N.B: For the moment we only support running the pipeline with a fixed tile_count.
            # Padding to the same shape or collating to a List of Tensors  will be adressed in another PR.
            tile_count=60,
            # declared in DatasetParams:
            local_datasets=[Path("/tmp/datasets/PANDA")],
            azure_datasets=["PANDA"])
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def setup(self) -> None:
        if self.encoder_type == SSLEncoder.__name__:
            self.downloader = self.download_ssl_checkpoint(innereye_ssl_checkpoint_binary)
        BaseMILSlides.setup(self)

    def get_data_module(self) -> PandaSlidesDataModule:
        return PandaSlidesDataModule(
            root_path=self.local_datasets[0],
            batch_size=self.batch_size,
            tile_count=self.tile_count,
            transforms_dict=self.get_transforms_dict(PandaDataset.IMAGE_COLUMN),
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=self.get_dataloader_kwargs(),
        )

    def get_slides_dataset(self) -> PandaDataset:
        return PandaDataset(root=self.local_datasets[0])                             # type: ignore


class SlidesPandaImageNetMIL(DeepSMILESlidesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetEncoder.__name__, **kwargs)


class SlidesPandaImageNetSimCLRMIL(DeepSMILESlidesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class SlidesPandaSSLMIL(DeepSMILESlidesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=SSLEncoder.__name__, **kwargs)


class SlidesPandaHistoSSLMIL(DeepSMILESlidesPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)
