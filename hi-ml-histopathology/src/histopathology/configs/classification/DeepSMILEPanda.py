#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any
from pathlib import Path
import os
import numpy as np
from monai.transforms import Compose
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch

from health_azure.utils import CheckpointDownloader
from health_azure.utils import get_workspace, is_running_in_azure_ml
from health_ml.networks.layers.attention_layers import AttentionLayer
from health_ml.utils import fixed_paths
from histopathology.datamodules.base_module import CacheMode, CacheLocation, HistoDataModule
from histopathology.datamodules.panda_module import (
    PandaSlidesDataModule,
    PandaTilesDataModule,
    SubPandaSlidesDataModule,
    SubPandaTilesDataModule,
)
from histopathology.datasets.panda_tiles_dataset import PandaTilesDataset

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

from histopathology.configs.classification.BaseMIL import SlidesBaseMIL, TilesBaseMIL, BaseMIL
from histopathology.datasets.panda_dataset import PandaDataset


class BaseDeepSMILEPanda(BaseMIL):
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

    def get_data_module(self) -> HistoDataModule:
        raise NotImplementedError                            # type: ignore


class TilesDeepSMILEPanda(TilesBaseMIL, BaseDeepSMILEPanda):
    """`is_finetune` sets the fine-tuning mode. If this is set, setting cache_mode=CacheMode.NONE takes ~30 min/epoch and
    cache_mode=CacheMode.MEMORY, precache_location=CacheLocation.CPU takes ~[5-10] min/epoch.
    Fine-tuning with caching completes using batch_size=4, max_bag_size=1000, max_epochs=20, max_num_gpus=1 on PANDA.
    """
    def __init__(self, **kwargs: Any) -> None:
        default_kwargs = dict(
            # declared in TilesBaseMIL:
            cache_mode=CacheMode.MEMORY,
            precache_location=CacheLocation.CPU,
            # declared in DatasetParams:
            local_datasets=[Path("/tmp/datasets/PANDA_tiles"), Path("/tmp/datasets/PANDA")],
            azure_datasets=["PANDA_tiles", "PANDA"])
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def get_data_module(self) -> PandaTilesDataModule:
        image_key = PandaTilesDataset.IMAGE_COLUMN
        if self.is_finetune:
            transform = LoadTilesBatchd(image_key, progress=True)
            workers_per_gpu = os.cpu_count() // torch.cuda.device_count()
            dataloader_kwargs = dict(num_workers=workers_per_gpu, pin_memory=True)
        else:
            transform = Compose([
                LoadTilesBatchd(image_key, progress=True),
                EncodeTilesBatchd(image_key, self.encoder, chunk_size=self.encoding_chunk_size)
            ])
            dataloader_kwargs = dict(num_workers=0, pin_memory=False)

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
            dataloader_kwargs=dataloader_kwargs,
        )

    def get_slides_dataset(self) -> PandaDataset:
        return PandaDataset(root=self.local_datasets[1])                             # type: ignore


class TilesPandaImageNetMIL(TilesDeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetEncoder.__name__, **kwargs)


class TilesPandaImageNetSimCLRMIL(TilesDeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class TilesPandaSSLMIL(TilesDeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=SSLEncoder.__name__, **kwargs)


class TilesPandaHistoSSLMIL(TilesDeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)


class SubPandaImageNetMIL(TilesPandaImageNetMIL):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        root_path = os.path.join(fixed_paths.repository_root_directory(), "hi-ml-histopathology/src/histopathology")
        self.crossval_count = 1
        self.train_csv = os.path.join(root_path, "configs/classification/panda/sub_train_tiles.csv")
        self.val_csv = os.path.join(root_path, "configs/classification/panda/sub_val_tiles.csv")

    def get_data_module(self) -> SubPandaTilesDataModule:
        image_key = PandaTilesDataset.IMAGE_COLUMN
        if self.is_finetune:
            transform = Compose([LoadTilesBatchd(image_key, progress=True)])
        else:
            transform = Compose(
                [
                    LoadTilesBatchd(image_key, progress=True),
                    EncodeTilesBatchd(image_key, self.encoder, chunk_size=self.encoding_chunk_size),
                ]
            )

        return SubPandaTilesDataModule(
            train_csv=self.train_csv,
            val_csv=self.val_csv,
            root_path=self.local_datasets[0],
            max_bag_size=self.max_bag_size,
            batch_size=self.batch_size,
            transform=transform,
            cache_mode=self.cache_mode,
            precache_location=self.precache_location,
            cache_dir=self.cache_dir,
        )


class SlidesDeepSMILEPanda(SlidesBaseMIL, BaseDeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        default_kwargs = dict(
            # declared in SlidesBaseMIL: TODO check if there are other parameters to set for PANDA (MONAI pipe)
            tile_count=60,
            # declared in DatasetParams:
            local_datasets=[Path("/tmp/datasets/PANDA")],
            azure_datasets=["PANDA"])
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def get_data_module(self) -> PandaSlidesDataModule:
        # TODO define which transform to apply
        image_key = PandaDataset.IMAGE_COLUMN
        normalize_transform = ScaleIntensityRanged(keys=image_key, a_min=np.float(0),
                                                   a_max=np.float(self.background_val))
        if self.is_finetune:
            transform = normalize_transform
            workers_per_gpu = os.cpu_count() // torch.cuda.device_count()
            dataloader_kwargs = dict(num_workers=workers_per_gpu, pin_memory=True)
        else:
            transform = Compose([normalize_transform,
                                EncodeTilesBatchd(image_key, self.encoder, chunk_size=self.encoding_chunk_size)])
            dataloader_kwargs = dict(num_workers=0, pin_memory=False)

        return PandaSlidesDataModule(
            root_path=self.local_datasets[0],
            batch_size=self.batch_size,
            tile_count=self.tile_count,
            transform=transform,
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=dataloader_kwargs,
        )

    def get_slides_dataset(self) -> PandaDataset:
        return PandaDataset(root=self.local_datasets[0])                             # type: ignore


class SlidesPandaImageNetMIL(SlidesDeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetEncoder.__name__, **kwargs)


class SlidesPandaImageNetSimCLRMIL(SlidesDeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class SlidesPandaSSLMIL(SlidesDeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=SSLEncoder.__name__, **kwargs)


class SlidesPandaHistoSSLMIL(SlidesDeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)


class SubSlidesPandaImageNetMIL(SlidesPandaImageNetMIL):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        root_path = os.path.join(fixed_paths.repository_root_directory(), "hi-ml-histopathology/src/histopathology")
        self.crossval_count = 1
        self.train_csv = os.path.join(root_path, "configs/classification/panda/sub_train_slides.csv")
        self.val_csv = os.path.join(root_path, "configs/classification/panda/sub_val_slides.csv")

    def get_data_module(self) -> SubPandaSlidesDataModule:
        # TODO define which transform to apply
        image_key = PandaDataset.IMAGE_COLUMN
        normalize_transform = ScaleIntensityRanged(keys=image_key, a_min=np.float(0),
                                                   a_max=np.float(self.background_val))
        if self.is_finetune:
            transform = normalize_transform
            workers_per_gpu = os.cpu_count() // torch.cuda.device_count()
            dataloader_kwargs = dict(num_workers=workers_per_gpu, pin_memory=True)
        else:
            transform = Compose([normalize_transform,
                                EncodeTilesBatchd(image_key, self.encoder, chunk_size=self.encoding_chunk_size)])
            dataloader_kwargs = dict(num_workers=0, pin_memory=False)

        return SubPandaSlidesDataModule(
            root_path=self.local_datasets[0],
            train_csv=self.train_csv,
            val_csv=self.val_csv,
            transform=transform,
            batch_size=self.batch_size,
            tile_count=self.tile_count,
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=dataloader_kwargs,
        )
