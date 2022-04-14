#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""BaseMIL is an abstract container defining basic functionality for running MIL experiments.
It is responsible for instantiating the encoder and full DeepMIL model. Subclasses should define
their datamodules and configure experiment-specific parameters.
"""
import os
from pytorch_lightning.callbacks.base import Callback
import torch
import param

from torch import nn
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from torchvision.models import resnet18
from monai.transforms.compose import Compose

from health_ml.lightning_container import LightningContainer
from health_ml.networks.layers.attention_layers import (AttentionLayer, GatedAttentionLayer, MaxPoolingLayer,
                                                        MeanPoolingLayer, TransformerPooling)
from health_ml.utils import fixed_paths
from health_ml.utils.checkpoint_utils import get_best_checkpoint_path
from histopathology.datamodules.base_module import CacheLocation, CacheMode, TilesDataModule
from histopathology.datasets.base_dataset import SlidesDataset
from histopathology.models.deepmil import DeepMILModule
from histopathology.models.encoders import (HistoSSLEncoder, IdentityEncoder, ImageNetEncoder, ImageNetSimCLREncoder,
                                            SSLEncoder, TileEncoder)
from histopathology.models.transforms import EncodeTilesBatchd, LoadTilesBatchd
from histopathology.utils.output_utils import DeepMILOutputsHandler
from histopathology.utils.naming import MetricsKey


class BaseMIL(LightningContainer):
    # Model parameters:
    pool_type: str = param.String(doc="Name of the pooling layer class to use.")
    pool_hidden_dim: int = param.Integer(128, doc="If pooling has a learnable part, this defines the number of the\
        hidden dimensions.")
    pool_out_dim: int = param.Integer(1, doc="Dimension of the pooled representation.")
    num_transformer_pool_layers: int = param.Integer(4, doc="If transformer pooling is chosen, this defines the number\
         of encoding layers.")
    num_transformer_pool_heads: int = param.Integer(4, doc="If transformer pooling is chosen, this defines the number\
         of attention heads.")
    is_finetune: bool = param.Boolean(False, doc="If True, fine-tune the encoder during training. If False (default), "
                                                 "keep the encoder frozen.")
    dropout_rate: Optional[float] = param.Number(None, bounds=(0, 1), doc="Pre-classifier dropout rate.")
    # l_rate, weight_decay, adam_betas are already declared in OptimizerParams superclass

    class_names: Optional[Sequence[str]] = param.List(None, item_type=str, doc="List of class names. If `None`, "
                                                                               "defaults to `('0', '1', ...)`.")

    # Encoder parameters:
    encoder_type: str = param.String(doc="Name of the encoder class to use.")
    tile_size: int = param.Integer(224, bounds=(1, None), doc="Tile width/height, in pixels.")
    n_channels: int = param.Integer(3, bounds=(1, None), doc="Number of channels in the tile.")

    # Data module parameters:
    batch_size: int = param.Integer(16, bounds=(1, None), doc="Number of slides to load per batch.")
    max_bag_size: int = param.Integer(1000, bounds=(0, None),
                                      doc="Upper bound on number of tiles in each loaded bag during training stage. "
                                          "If 0 (default), will return all samples in each bag. "
                                          "If > 0, bags larger than `max_bag_size` will yield "
                                          "random subsets of instances.")
    max_bag_size_inf: int = param.Integer(0, bounds=(0, None),
                                          doc="Upper bound on number of tiles in each loaded bag during "
                                          "validation and test stages."
                                          "If 0 (default), will return all samples in each bag. "
                                          "If > 0 , bags larger than `max_bag_size_inf` will yield "
                                          "random subsets of instances.")
    cache_mode: CacheMode = param.ClassSelector(default=CacheMode.MEMORY, class_=CacheMode,
                                                doc="The type of caching to perform: "
                                                    "'memory' (default), 'disk', or 'none'.")
    precache_location: CacheLocation = param.ClassSelector(default=CacheLocation.NONE, class_=CacheLocation,
                                                           doc="Whether to pre-cache the entire transformed dataset "
                                                               "upfront and save it to disk and if re-load in cpu or "
                                                               "gpu. Options: `none` (default),`cpu`, `gpu`")
    encoding_chunk_size: int = param.Integer(0, doc="If > 0 performs encoding in chunks, by loading"
                                                    "enconding_chunk_size tiles per chunk")
    is_caching: bool = param.Boolean(False, doc="If True, cache the encoded tile features "
                                     "(disables random subsampling of tiles). "
                                     "If False (default), load the tiles without caching "
                                     "(enables random subsampling of tiles).")
    # local_dataset (used as data module root_path) is declared in DatasetParams superclass

    @property
    def cache_dir(self) -> Path:
        raise NotImplementedError

    def setup(self) -> None:
        if self.encoder_type == SSLEncoder.__name__:
            raise NotImplementedError("SSLEncoder requires a pre-trained checkpoint.")

        self.encoder = self.get_encoder()
        if not self.is_finetune:
            self.encoder.eval()
        # Fine-tuning requires tiles to be loaded on-the-fly, hence, caching is disabled by default.
        if self.is_finetune:
            self.is_caching = False
        if self.is_caching:
            self.cache_mode = CacheMode.MEMORY
            self.precache_location = CacheLocation.CPU
        else:
            self.cache_mode = CacheMode.NONE
            self.precache_location = CacheLocation.NONE

    def get_encoder(self) -> TileEncoder:
        if self.encoder_type == ImageNetEncoder.__name__:
            return ImageNetEncoder(feature_extraction_model=resnet18,
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
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        num_features = num_encoding * self.pool_out_dim
        return pooling_layer, num_features

    def create_model(self) -> DeepMILModule:
        self.data_module = self.get_data_module()
        # Encoding is done in the datamodule, so here we provide instead a dummy
        # no-op IdentityEncoder to be used inside the model
        if not self.is_caching:
            self.model_encoder = self.encoder
            if self.is_finetune:
                for params in self.model_encoder.parameters():
                    params.requires_grad = True
        else:
            self.model_encoder = IdentityEncoder(input_dim=(self.encoder.num_encoding,))

        # Construct pooling layer
        pooling_layer, num_features = self.get_pooling_layer()

        outputs_handler = DeepMILOutputsHandler(outputs_root=self.outputs_folder,
                                                n_classes=self.data_module.train_dataset.N_CLASSES,
                                                tile_size=self.tile_size,
                                                level=1,
                                                class_names=self.class_names,
                                                primary_val_metric=MetricsKey.AUROC,
                                                maximise=True)
        deepmil_module = DeepMILModule(encoder=self.model_encoder,
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
                                       outputs_handler=outputs_handler
                                       )
        outputs_handler.set_slides_dataset(self.get_slides_dataset())
        return deepmil_module

    def get_data_module(self) -> TilesDataModule:
        raise NotImplementedError

    def get_slides_dataset(self) -> Optional[SlidesDataset]:
        return None

    def get_transform(self, image_key: str) -> Callable:
        if not self.is_caching:
            return LoadTilesBatchd(image_key, progress=True)
        else:
            return Compose([
                LoadTilesBatchd(image_key, progress=True),
                EncodeTilesBatchd(image_key, self.encoder, chunk_size=self.encoding_chunk_size)
            ])

    def get_dataloader_kwargs(self) -> dict:
        if not self.is_caching:
            num_cpus = os.cpu_count()
            num_devices = max(torch.cuda.device_count(), 1)
            assert num_cpus is not None  # for mypy
            workers_per_gpu = num_cpus // num_devices
            dataloader_kwargs = dict(num_workers=workers_per_gpu, pin_memory=True)
        else:
            dataloader_kwargs = dict(num_workers=0, pin_memory=False)
        return dataloader_kwargs

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
