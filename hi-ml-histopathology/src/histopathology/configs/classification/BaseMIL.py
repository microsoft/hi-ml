#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
import torch
import param
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from torch import nn
from pathlib import Path
from monai.transforms import Compose
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torchvision.models.resnet import resnet18, resnet50

from health_azure.utils import CheckpointDownloader, get_workspace

from health_ml.utils import fixed_paths
from health_ml.lightning_container import LightningContainer
from health_ml.utils.checkpoint_utils import LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, get_best_checkpoint_path
from health_ml.networks.layers.attention_layers import (AttentionLayer, GatedAttentionLayer, MaxPoolingLayer,
                                                        MeanPoolingLayer, TransformerPooling,
                                                        TransformerPoolingBenchmark)
from health_ml.utils.common_utils import CHECKPOINT_FOLDER, DEFAULT_AML_UPLOAD_DIR

from histopathology.datamodules.base_module import CacheLocation, CacheMode, HistoDataModule
from histopathology.datasets.base_dataset import SlidesDataset
from histopathology.models.deepmil import TilesDeepMILModule, SlidesDeepMILModule, BaseDeepMILModule
from histopathology.models.encoders import (
    HistoSSLEncoder, IdentityEncoder, ImageNetEncoder, ImageNetEncoder_Resnet50, ImageNetSimCLREncoder,
    SSLEncoder, TileEncoder)
from histopathology.models.transforms import EncodeTilesBatchd, LoadTilesBatchd
from histopathology.utils.output_utils import DeepMILOutputsHandler
from histopathology.utils.naming import MetricsKey, PlotOption, SlideKey, ModelKey
from histopathology.utils.tiles_selection_utils import TilesSelector


class BaseMIL(LightningContainer):
    """BaseMIL is an abstract container defining basic functionality for running MIL experiments in both slides and
    tiles settings. It is responsible for instantiating the encoder and pooling layer. Subclasses should define the
    full DeepMIL model depending on the type of dataset (tiles/slides based).
    """
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
    primary_val_metric: MetricsKey = param.ClassSelector(default=MetricsKey.AUROC, class_=MetricsKey,
                                                         doc="Primary validation metric to track for checkpointing and "
                                                             "generating outputs.")
    maximise_primary_metric: bool = param.Boolean(True, doc="Whether the primary validation metric should be "
                                                            "maximised (otherwise minimised).")

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
    encoding_chunk_size: int = param.Integer(0, doc="If > 0 performs encoding in chunks, by loading"
                                                    "enconding_chunk_size tiles per chunk")
    # local_dataset (used as data module root_path) is declared in DatasetParams superclass
    level: int = param.Integer(1, bounds=(0, None), doc="The whole slide image level at which the image is extracted."
                                                        "Whole slide images are represented in a pyramid consisting of"
                                                        "multiple images at different resolutions."
                                                        "If 1 (default), will extract baseline image at the resolution"
                                                        "at level 1.")

    # Outputs Handler parameters:
    save_output_slides: bool = param.Boolean(True, doc="a boolean parameter to enable saving heatmaps and thumbnails.")
    num_top_slides: int = param.Integer(10, bounds=(0, None), doc="Number of slides to select when saving top and "
                                                                  "bottom tiles. If set to 10 (default), it selects 10 "
                                                                  "top and 10 bottom slides. To disable tiles plots "
                                                                  "saving, set `num_top_slides=0`")
    num_top_tiles: int = param.Integer(12, bounds=(1, None), doc="Number of tiles to select when saving top and bottom"
                                                                 "tiles. If set to 12 (default), it saves 12 top and 12"
                                                                 "bottom tiles.")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.best_checkpoint_filename = "checkpoint_max_val_auroc"
        self.best_checkpoint_filename_with_suffix = self.best_checkpoint_filename + ".ckpt"

    @property
    def cache_dir(self) -> Path:
        return Path(f"/tmp/himl_cache/{self.__class__.__name__}-{self.encoder_type}/")

    def setup(self) -> None:
        self.encoder = self.get_encoder()
        if not self.is_finetune:
            self.encoder.eval()

    def download_ssl_checkpoint(self, run_id: str) -> CheckpointDownloader:
        downloader = CheckpointDownloader(
            aml_workspace=get_workspace(),
            run_id=run_id,
            checkpoint_filename=LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX,
            download_dir=self.outputs_folder,
            remote_checkpoint_dir=Path(f"{DEFAULT_AML_UPLOAD_DIR}/{CHECKPOINT_FOLDER}/")
        )
        downloader.download_checkpoint_if_necessary()
        return downloader

    def get_encoder(self) -> TileEncoder:
        if self.encoder_type == ImageNetEncoder.__name__:
            return ImageNetEncoder(feature_extraction_model=resnet18,
                                   tile_size=self.tile_size, n_channels=self.n_channels)
        elif self.encoder_type == ImageNetEncoder_Resnet50.__name__:
            # Myronenko et al. 2021 uses Resnet50 CNN encoder
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
        elif self.pool_type == TransformerPoolingBenchmark.__name__:
            pooling_layer = TransformerPoolingBenchmark(self.num_transformer_pool_layers,
                                                        self.num_transformer_pool_heads,
                                                        num_encoding,
                                                        self.pool_hidden_dim)
            self.pool_out_dim = 1  # currently this is hardcoded in forward of the TransformerPooling
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        num_features = num_encoding * self.pool_out_dim
        return pooling_layer, num_features

    def get_test_plot_options(self) -> Set[PlotOption]:
        if self.num_top_slides > 0:
            return {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX, PlotOption.TOP_BOTTOM_TILES}
        else:
            return {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX}

    def get_val_plot_options(self) -> Set[PlotOption]:
        return {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX}

    def get_outputs_handler(self) -> DeepMILOutputsHandler:
        n_classes = self.data_module.train_dataset.N_CLASSES
        outputs_handler = DeepMILOutputsHandler(
            outputs_root=self.outputs_folder,
            n_classes=n_classes,
            tile_size=self.tile_size,
            level=self.level,
            class_names=self.class_names,
            primary_val_metric=self.primary_val_metric,
            maximise=self.maximise_primary_metric,
            val_plot_options=self.get_val_plot_options(),
            test_plot_options=self.get_test_plot_options(),
        )
        if self.num_top_slides > 0:
            outputs_handler.tiles_selector = TilesSelector(
                n_classes=n_classes, num_slides=self.num_top_slides, num_tiles=self.num_top_tiles
            )
        return outputs_handler

    def get_model_encoder(self) -> TileEncoder:
        model_encoder = self.encoder
        if self.is_finetune:
            for params in model_encoder.parameters():
                params.requires_grad = True
        return model_encoder

    def get_callbacks(self) -> List[Callback]:
        return [*super().get_callbacks(),
                ModelCheckpoint(dirpath=self.checkpoint_folder,
                                monitor=f"{ModelKey.VAL}/{self.primary_val_metric}",
                                filename=self.best_checkpoint_filename,
                                auto_insert_metric_name=False,
                                mode="max" if self.maximise_primary_metric else "min")]

    def get_checkpoint_to_test(self) -> Path:
        """
        Returns the full path to a checkpoint file that was found to be best during training, whatever criterion
        was applied there. This is necessary since for some models the checkpoint is in a subfolder of the checkpoint
        folder.
        """
        # absolute path is required for registering the model.
        absolute_checkpoint_path = Path(fixed_paths.repository_root_directory(),
                                        f"{DEFAULT_AML_UPLOAD_DIR}/{CHECKPOINT_FOLDER}/",
                                        self.best_checkpoint_filename_with_suffix)
        if absolute_checkpoint_path.is_file():
            return absolute_checkpoint_path

        absolute_checkpoint_path_parent = Path(fixed_paths.repository_root_directory().parent,
                                               f"{DEFAULT_AML_UPLOAD_DIR}/{CHECKPOINT_FOLDER}/",
                                               self.best_checkpoint_filename_with_suffix)
        if absolute_checkpoint_path_parent.is_file():
            return absolute_checkpoint_path_parent

        checkpoint_path = get_best_checkpoint_path(self.checkpoint_folder)
        if checkpoint_path.is_file():
            return checkpoint_path

        raise ValueError("Path to best checkpoint not found")

    def get_dataloader_kwargs(self) -> dict:
        num_cpus = os.cpu_count()
        # We ensure num_devices is not 0 for non-GPU machines
        # to avoid division by zero error when computing `workers_per_gpu`
        num_devices = max(torch.cuda.device_count(), 1)
        assert num_cpus is not None  # for mypy
        workers_per_gpu = num_cpus // num_devices
        dataloader_kwargs = dict(num_workers=workers_per_gpu, pin_memory=True)
        return dataloader_kwargs

    def get_transforms_dict(self, image_key: str) -> Optional[Dict[ModelKey, Union[Callable, None]]]:
        return None

    def create_model(self) -> BaseDeepMILModule:
        raise NotImplementedError

    def get_data_module(self) -> HistoDataModule:
        raise NotImplementedError

    def get_slides_dataset(self) -> Optional[SlidesDataset]:
        return None


class BaseMILTiles(BaseMIL):
    """BaseMILTiles is an abstract subclass of BaseMIL for running MIL experiments on tiles datasets. It is responsible
    for instantiating the full DeepMIL model in tiles settings. Subclasses should define their datamodules and
    configure experiment-specific parameters.
    """
    # Tiles Data module parameters:
    cache_mode: CacheMode = param.ClassSelector(default=CacheMode.MEMORY, class_=CacheMode,
                                                doc="The type of caching to perform: "
                                                    "'memory' (default), 'disk', or 'none'.")
    precache_location: CacheLocation = param.ClassSelector(default=CacheLocation.CPU, class_=CacheLocation,
                                                           doc="Whether to pre-cache the entire transformed dataset "
                                                               "upfront and save it to disk and if re-load in cpu or "
                                                               "gpu. Options: `none`,`cpu` (default), `gpu`")
    is_caching: bool = param.Boolean(False, doc="If True, cache the encoded tile features "
                                     "(disables random subsampling of tiles). "
                                     "If False (default), load the tiles without caching "
                                     "(enables random subsampling of tiles).")

    def setup(self) -> None:
        super().setup()
        # Fine-tuning requires tiles to be loaded on-the-fly, hence, caching is disabled by default.
        # When is_finetune and is_caching are both set, below lines should disable caching automatically.
        if self.is_finetune:
            self.is_caching = False
        if not self.is_caching:
            self.cache_mode = CacheMode.NONE
            self.precache_location = CacheLocation.NONE

    def get_dataloader_kwargs(self) -> dict:
        if self.is_caching:
            dataloader_kwargs = dict(num_workers=0, pin_memory=False)
        else:
            dataloader_kwargs = super().get_dataloader_kwargs()
        return dataloader_kwargs

    def get_transforms_dict(self, image_key: str) -> Dict[ModelKey, Union[Callable, None]]:
        if self.is_caching:
            transform = Compose([
                LoadTilesBatchd(image_key, progress=True),
                EncodeTilesBatchd(image_key, self.encoder, chunk_size=self.encoding_chunk_size)
            ])
        else:
            transform = LoadTilesBatchd(image_key, progress=True)
        # in case the transformations for training contain augmentations, val and test transform will be different
        return {ModelKey.TRAIN: transform, ModelKey.VAL: transform, ModelKey.TEST: transform}

    def get_model_encoder(self) -> TileEncoder:
        if self.is_caching:
            # Encoding is done in the datamodule, so here we provide instead a dummy
            # no-op IdentityEncoder to be used inside the model
            return IdentityEncoder(input_dim=(self.encoder.num_encoding,))
        else:
            return super().get_model_encoder()

    def create_model(self) -> TilesDeepMILModule:
        self.data_module = self.get_data_module()
        pooling_layer, num_features = self.get_pooling_layer()
        outputs_handler = self.get_outputs_handler()
        deepmil_module = TilesDeepMILModule(encoder=self.get_model_encoder(),
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
                                            chunk_size=self.encoding_chunk_size)
        outputs_handler.set_slides_dataset_for_plots_handlers(self.get_slides_dataset())
        outputs_handler.set_conf_matrix_for_plots_handlers(deepmil_module.get_metrics())
        return deepmil_module


class BaseMILSlides(BaseMIL):
    """BaseSlidesMIL is an abstract subclass of BaseMIL for running MIL experiments on slides datasets. It is
    responsible for instantiating the full DeepMIL model in slides settings. Subclasses should define their datamodules
    and configure experiment-specific parameters.
    """
    # Slides Data module parameters:
    tile_size: int = param.Integer(224, bounds=(0, None), doc="Size of the square tile, defaults to 224.")
    step: int = param.Integer(None, bounds=(0, None),
                              doc="Step size to define the offset between tiles."
                              "If None (default), it takes the same value as tile_size."
                              "If step < tile_size, it creates overlapping tiles."
                              "If step > tile_size, it skips some chunks in the wsi.")
    random_offset: bool = param.Boolean(False, doc="If True, randomize position of the grid, instead of starting at"
                                                   "the top-left corner,")
    pad_full: bool = param.Boolean(False, doc="If True, pad image to the size evenly divisible by tile_size")
    background_val: int = param.Integer(255, bounds=(0, None),
                                        doc="Threshold to estimate the foreground in a whole slide image.")
    filter_mode: str = param.String("min", doc="mode must be in ['min', 'max', 'random']. If total number of tiles is"
                                               "greater than tile_count, then sort by intensity sum, and take the "
                                               "smallest (for min), largest (for max) or random (for random) subset, "
                                               "defaults to 'min' (which assumes background is high value).")

    def create_model(self) -> SlidesDeepMILModule:
        self.data_module = self.get_data_module()
        pooling_layer, num_features = self.get_pooling_layer()
        outputs_handler = self.get_outputs_handler()
        deepmil_module = SlidesDeepMILModule(encoder=self.get_model_encoder(),
                                             label_column=SlideKey.LABEL,
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
                                             outputs_handler=outputs_handler)
        outputs_handler.set_slides_dataset_for_plots_handlers(self.get_slides_dataset())
        outputs_handler.set_conf_matrix_for_plots_handlers(deepmil_module.get_metrics())
        return deepmil_module
