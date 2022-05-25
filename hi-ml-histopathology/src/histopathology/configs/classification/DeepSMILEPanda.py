#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from torch import nn
from monai.transforms import Compose, ScaleIntensityRanged
from torchvision.models.resnet import resnet50
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

from health_azure.utils import is_running_in_azure_ml

from health_ml.networks.layers.attention_layers import (
    AttentionLayer,
    GatedAttentionLayer,
    MaxPoolingLayer,
    MeanPoolingLayer,
    TransformerPooling
)
from health_ml.utils.data_augmentations import RandomRotationByMultiplesOf90
from histopathology.configs.run_ids import innereye_ssl_checkpoint_binary
from histopathology.datamodules.panda_module_baseline import PandaTilesDataModuleBaseline
from histopathology.datamodules.panda_module import (
    PandaSlidesDataModule,
    PandaTilesDataModule)
from histopathology.datasets.panda_tiles_dataset import PandaTilesDataset
from histopathology.models.deepmil import PandaTilesDeepMILModule
from histopathology.models.encoders import (
    HistoSSLEncoder,
    ImageNetEncoder,
    ImageNetEncoder_Resnet50,
    ImageNetSimCLREncoder,
    SSLEncoder,
    TileEncoder
)
from histopathology.configs.classification.BaseMIL import BaseMILSlides, BaseMILTiles, BaseMIL
from histopathology.datasets.panda_dataset import PandaDataset
from histopathology.datasets.default_paths import (
    PANDA_DATASET_DIR,
    PANDA_DATASET_ID,
    PANDA_TILES_DATASET_DIR,
    PANDA_TILES_DATASET_ID)
from histopathology.layers.transformerpooling import TransformerPooling
from histopathology.models.deepmil import TilesDeepMILModule
from histopathology.models.transforms import transform_dict_adaptor, EncodeTilesBatchd, LoadTilesBatchd
from histopathology.utils.naming import MetricsKey, ModelKey


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
            self.max_epochs = 2


class DeepSMILETilesPandaBasic(BaseMILTiles, BaseDeepSMILEPanda):
    """ DeepSMILETilesPandaBasic is derived from BaseMILTiles and BaseDeepSMILEPanda to inherit common behaviors from both
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
            local_datasets=[Path(PANDA_TILES_DATASET_DIR), Path(PANDA_DATASET_DIR)],
            azure_datasets=[PANDA_TILES_DATASET_ID, PANDA_DATASET_ID])
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
        return PandaDataset(root=self.local_datasets[1])  # type: ignore


class DeepSMILETilesPanda(DeepSMILETilesPandaBasic):
    """
    Configuration for PANDA experiments from Myronenko et al. 2021:
    (https://link.springer.com/chapter/10.1007/978-3-030-87237-3_32)

    `is_finetune` sets the fine-tuning mode. For fine-tuning,
    batch_size = 2 runs on 8 GPUs with
    ~ 6:24 min/epoch (train) and ~ 00:50 min/epoch (validation).
    """
    def __init__(self, **kwargs: Any) -> None:
        default_kwargs = dict(
            pool_type=AttentionLayer.__name__,
            num_transformer_pool_layers=4,
            num_transformer_pool_heads=8,
            pool_hidden_dim=2048,
            encoding_chunk_size=60,
            max_bag_size=56,
            batch_size=8,  # effective batch size = batch_size * num_GPUs
            max_epochs=50,
            l_rate=3e-4,
            weight_decay=0,
            primary_val_metric=MetricsKey.ACC)
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def setup(self) -> None:
        # Params specific to transformer pooling
        if self.pool_type in [TransformerPooling.__name__, TransformerPooling.__name__]:
            self.l_rate = 3e-5
            self.weight_decay = 0.1
        # Params specific to fine-tuning
        if self.is_finetune:
            self.batch_size = 2
        super().setup()

    def get_transforms_dict(self, image_key: str) -> Dict[ModelKey, Union[Callable, None]]:
        if self.is_caching:
            transform_train = Compose([
                                      LoadTilesBatchd(image_key, progress=False),
                                      ScaleIntensityRanged(keys=image_key, a_min=0.0, a_max=255.0),
                                      EncodeTilesBatchd(image_key, self.encoder, chunk_size=self.encoding_chunk_size)
                                      ])
        else:
            # Use same transforms as demonstrated in
            # https://github.com/Project-MONAI/tutorials/blob/master/pathology/multiple_instance_learning/panda_mil_train_evaluate_pytorch_gpu.py
            transform_train = Compose([
                                      LoadTilesBatchd(image_key, progress=False),
                                      transform_dict_adaptor(RandomHorizontalFlip(p=0.5), image_key, image_key),
                                      transform_dict_adaptor(RandomVerticalFlip(p=0.5), image_key, image_key),
                                      transform_dict_adaptor(RandomRotationByMultiplesOf90(), image_key, image_key),
                                      ScaleIntensityRanged(keys=image_key, a_min=0.0, a_max=255.0),
                                      ])

        if self.is_caching:
            transform_inf = Compose([
                                    LoadTilesBatchd(image_key, progress=False),
                                    ScaleIntensityRanged(keys=image_key, a_min=0.0, a_max=255.0),
                                    EncodeTilesBatchd(image_key, self.encoder, chunk_size=self.encoding_chunk_size)
                                    ])
        else:
            transform_inf = Compose([
                                    LoadTilesBatchd(image_key, progress=False),
                                    ScaleIntensityRanged(keys=image_key, a_min=0.0, a_max=255.0)
                                    ])

        # in case the transformations for training contain augmentations, val and test transform will be different
        return {ModelKey.TRAIN: transform_train, ModelKey.VAL: transform_inf, ModelKey.TEST: transform_inf}

    def get_data_module(self) -> PandaTilesDataModuleBaseline:
        # Myronenko et al. 2021 uses 80-20 cross-val split and no hold-out test set
        # Hence, inherited `PandaTilesDataModuleBaseline` from `TilesDataModule`
        return PandaTilesDataModuleBaseline(
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

    def create_model(self) -> TilesDeepMILModule:
        self.data_module = self.get_data_module()
        pooling_layer, num_features = self.get_pooling_layer()
        outputs_handler = self.get_outputs_handler()
        # Myronenko et al. 2021 uses a cosine LR scheduler which needs to be defined in the PL module
        # Hence, inherited `PandaTilesDeepMILModule` from `TilesDeepMILModule`
        deepmil_module = PandaTilesDeepMILModule(encoder=self.get_model_encoder(),
                                                 label_column=self.data_module.train_dataset.LABEL_COLUMN,
                                                 n_classes=self.data_module.train_dataset.N_CLASSES,
                                                 pooling_layer=pooling_layer,
                                                 num_features=num_features,
                                                 dropout_rate=self.dropout_rate,
                                                 class_weights=self.data_module.class_weights,
                                                 l_rate=self.l_rate,
                                                 weight_decay=self.weight_decay,
                                                 adam_betas=self.adam_betas,
                                                 is_finetune=self.is_finetune,
                                                 class_names=self.class_names,
                                                 outputs_handler=outputs_handler,
                                                 chunk_size=self.encoding_chunk_size,
                                                 n_epochs=self.max_epochs)
        outputs_handler.set_slides_dataset(self.get_slides_dataset())
        return deepmil_module

    def get_encoder(self) -> TileEncoder:
        # Myronenko et al. 2021 uses Resnet50 CNN encoder
        if self.encoder_type == ImageNetEncoder.__name__:
            return ImageNetEncoder_Resnet50(feature_extraction_model=resnet50,
                                            tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == ImageNetSimCLREncoder.__name__:
            return ImageNetSimCLREncoder(tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == HistoSSLEncoder.__name__:
            return HistoSSLEncoder(tile_size=self.tile_size, n_channels=self.n_channels)

        elif self.encoder_type == SSLEncoder.__name__:
            return SSLEncoder(pl_checkpoint_path=self.downloader.local_checkpoint_path,
                              tile_size=self.tile_size, n_channels=self.n_channels)

        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder_type}")

    def get_pooling_layer(self) -> Tuple[nn.Module, int]:
        # Myronenko et al. 2021 has a different transformer pooling layer.
        # This is defined in `TransformerPooling`.
        num_encoding = self.encoder.num_encoding

        pooling_layer: nn.Module
        if self.pool_type == AttentionLayer.__name__:
            pooling_layer = AttentionLayer(num_encoding,
                                           self.pool_hidden_dim,
                                           self.pool_out_dim)
        elif self.pool_type == GatedAttentionLayer.__name__:
            pooling_layer = GatedAttentionLayer(num_encoding,
                                                self.pool_hidden_dim,
                                                self.pool_out_dim)
        elif self.pool_type == MeanPoolingLayer.__name__:
            pooling_layer = MeanPoolingLayer()
        elif self.pool_type == MaxPoolingLayer.__name__:
            pooling_layer = MaxPoolingLayer()
        elif self.pool_type == TransformerPooling.__name__:
            pooling_layer = TransformerPooling(self.num_transformer_pool_layers,
                                               self.num_transformer_pool_heads,
                                               num_encoding)
            self.pool_out_dim = 1  # currently this is hardcoded in forward of the TransformerPooling
        elif self.pool_type == TransformerPooling.__name__:
            pooling_layer = TransformerPooling(self.num_transformer_pool_layers,
                                               self.num_transformer_pool_heads,
                                               num_encoding,
                                               self.pool_hidden_dim)
            self.pool_out_dim = 1  # currently this is hardcoded in forward of the TransformerPooling
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        num_features = num_encoding * self.pool_out_dim
        return pooling_layer, num_features


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
            level=1,
            max_bag_size=56,
            max_bag_size_inf=0,
            tile_size=224,
            random_offset=True,
            background_val=255,
            # declared in DatasetParams:
            local_datasets=[Path("/tmp/datasets/PANDA")],
            azure_datasets=["PANDA"],
            save_output_tiles=False,)
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def setup(self) -> None:
        if self.encoder_type == SSLEncoder.__name__:
            self.downloader = self.download_ssl_checkpoint(innereye_ssl_checkpoint_binary)
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
            level=self.level,
            max_bag_size=self.max_bag_size,
            max_bag_size_inf=self.max_bag_size_inf,
            tile_size=self.tile_size,
            step=self.step,
            random_offset=self.random_offset,
            pad_full=self.pad_full,
            background_val=self.background_val,
            filter_mode=self.filter_mode,
            transforms_dict=self.get_transforms_dict(PandaDataset.IMAGE_COLUMN),
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=self.get_dataloader_kwargs(),
        )

    def get_slides_dataset(self) -> PandaDataset:
        return PandaDataset(root=self.local_datasets[0])  # type: ignore


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
