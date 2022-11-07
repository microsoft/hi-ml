#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
import logging
import param
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

from pathlib import Path
from monai.transforms import Compose
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from health_azure.utils import create_from_matching_params
from health_cpath.utils.callbacks import LossAnalysisCallback, LossCallbackParams

from health_ml.utils import fixed_paths
from health_ml.deep_learning_config import OptimizerParams
from health_ml.lightning_container import LightningContainer
from health_ml.utils.checkpoint_utils import get_best_checkpoint_path, CheckpointParser
from health_ml.utils.common_utils import DEFAULT_AML_CHECKPOINT_DIR

from health_cpath.datamodules.base_module import CacheLocation, CacheMode, HistoDataModule
from health_cpath.datasets.base_dataset import SlidesDataset
from health_cpath.models.deepmil import TilesDeepMILModule, SlidesDeepMILModule, BaseDeepMILModule
from health_cpath.models.transforms import EncodeTilesBatchd, LoadTilesBatchd
from health_cpath.utils.deepmil_utils import ClassifierParams, EncoderParams, PoolingParams
from health_cpath.utils.output_utils import DeepMILOutputsHandler
from health_cpath.utils.naming import MetricsKey, PlotOption, SlideKey, ModelKey
from health_cpath.utils.tiles_selection_utils import TilesSelector


class BaseMIL(LightningContainer, EncoderParams, PoolingParams, ClassifierParams, LossCallbackParams):
    """BaseMIL is an abstract container defining basic functionality for running MIL experiments in both slides and
    tiles settings. It is responsible for instantiating the encoder and pooling layer. Subclasses should define the
    full DeepMIL model depending on the type of dataset (tiles/slides based).
    """
    class_names: Optional[Sequence[str]] = param.List(None, item_type=str, doc="List of class names. If `None`, "
                                                                               "defaults to `('0', '1', ...)`.")
    # Data module parameters:
    batch_size: int = param.Integer(16, bounds=(1, None), doc="Number of slides to load per batch.")
    batch_size_inf: int = param.Integer(16, bounds=(1, None), doc="Number of slides per batch during inference.")
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
    # local_dataset (used as data module root_path) is declared in DatasetParams superclass
    level: int = param.Integer(1, bounds=(0, None), doc="The whole slide image level at which the image is extracted."
                                                        "Whole slide images are represented in a pyramid consisting of"
                                                        "multiple images at different resolutions."
                                                        "If 1 (default), will extract baseline image at the resolution"
                                                        "at level 1.")
    # Outputs Handler parameters:
    num_top_slides: int = param.Integer(10, bounds=(0, None), doc="Number of slides to select when saving top and "
                                                                  "bottom tiles. If set to 10 (default), it selects 10 "
                                                                  "top and 10 bottom slides. To disable tiles plots "
                                                                  "saving, set `num_top_slides=0`")
    num_top_tiles: int = param.Integer(12, bounds=(1, None), doc="Number of tiles to select when saving top and bottom"
                                                                 "tiles. If set to 12 (default), it saves 12 top and 12"
                                                                 "bottom tiles.")
    primary_val_metric: MetricsKey = param.ClassSelector(default=MetricsKey.AUROC, class_=MetricsKey,
                                                         doc="Primary validation metric to track for checkpointing and "
                                                             "generating outputs.")
    maximise_primary_metric: bool = param.Boolean(True, doc="Whether the primary validation metric should be "
                                                            "maximised (otherwise minimised).")
    max_num_workers: int = param.Integer(10, bounds=(0, None),
                                         doc="The maximum number of worker processes for dataloaders. Dataloaders use"
                                             "a heuristic num_cpus/num_gpus to set the number of workers, which can be"
                                             "very high for small num_gpus. This parameters sets an upper bound.")
    wsi_has_mask: bool = param.Boolean(default=True,
                                       doc="Whether the WSI has a mask. If True, will use the mask to load a specific"
                                           "region of the WSI. If False, will load the whole WSI.")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_extra_val_epoch = True  # Enable running an additional validation step to save tiles/slides thumbnails
        metric_optim = "max" if self.maximise_primary_metric else "min"
        self.best_checkpoint_filename = f"checkpoint_{metric_optim}_val_{self.primary_val_metric.value}"
        self.best_checkpoint_filename_with_suffix = self.best_checkpoint_filename + ".ckpt"
        self.validate()

    def validate(self) -> None:
        super().validate()
        EncoderParams.validate(self)
        if not any([self.tune_encoder, self.tune_pooling, self.tune_classifier]) and not self.run_inference_only:
            raise ValueError(
                "At least one of the encoder, pooling or classifier should be fine tuned. Turn on one of the tune "
                "arguments `tune_encoder`, `tune_pooling`, `tune_classifier`. Otherwise, activate inference only "
                "mode via `run_inference_only` flag."
            )
        if (
            any([self.pretrained_encoder, self.pretrained_pooling, self.pretrained_classifier])
            and not self.src_checkpoint
        ):
            raise ValueError(
                "You need to specify a source checkpoint, to use a pretrained encoder, pooling or classifier."
                f" {CheckpointParser.INFO_MESSAGE}"
            )
        if (
            self.tune_encoder and self.encoding_chunk_size < self.max_bag_size
            and self.pl_sync_batchnorm and self.max_num_gpus > 1
        ):
            raise ValueError(
                "The encoding chunk size should be at least as large as the maximum bag size when fine tuning the "
                "encoder. You might encounter Batch Norm synchronization issues if the chunk size is smaller than "
                "the maximum bag size causing the processes to hang silently. This is due to the encoder being called "
                "different number of times on each device, which cause Batch Norm running statistics to be updated "
                "inconsistently across processes. In case you can't increase the `encoding_chunk_size` any further, set"
                " `pl_sync_batchnorm=False` to simply skip Batch Norm synchronization across devices. Note that this "
                "might come with some performance penalty."
            )

    @property
    def cache_dir(self) -> Path:
        return Path(f"/tmp/himl_cache/{self.__class__.__name__}-{self.encoder_type}/")

    def get_test_plot_options(self) -> Set[PlotOption]:
        options = {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX}
        if self.num_top_slides > 0:
            options.add(PlotOption.TOP_BOTTOM_TILES)
        return options

    # overwrite this method if you want to produce validation plots at each epoch. By default, at the end of the
    # training an extra validation epoch is run where val_plot_options = test_plot_options
    def get_val_plot_options(self) -> Set[PlotOption]:
        return set()

    def get_outputs_handler(self) -> DeepMILOutputsHandler:
        n_classes = self.data_module.train_dataset.n_classes
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
            wsi_has_mask=self.wsi_has_mask,
            val_set_is_dist=not self.pl_replace_sampler_ddp,
        )
        if self.num_top_slides > 0:
            outputs_handler.tiles_selector = TilesSelector(
                n_classes=n_classes, num_slides=self.num_top_slides, num_tiles=self.num_top_tiles
            )
        return outputs_handler

    def get_callbacks(self) -> List[Callback]:
        callbacks = [*super().get_callbacks(),
                     ModelCheckpoint(dirpath=self.checkpoint_folder,
                                     monitor=f"{ModelKey.VAL}/{self.primary_val_metric}",
                                     filename=self.best_checkpoint_filename,
                                     auto_insert_metric_name=False,
                                     mode="max" if self.maximise_primary_metric else "min")]
        if self.analyse_loss:
            callbacks.append(LossAnalysisCallback(outputs_folder=self.outputs_folder,
                                                  max_epochs=self.max_epochs,
                                                  patience=self.loss_analysis_patience,
                                                  epochs_interval=self.loss_analysis_epochs_interval,
                                                  num_slides_scatter=self.num_slides_scatter,
                                                  num_slides_heatmap=self.num_slides_heatmap,
                                                  save_tile_ids=self.save_tile_ids,
                                                  log_exceptions=self.log_exceptions))
        return callbacks

    def get_checkpoint_to_test(self) -> Path:
        """
        Returns the full path to a checkpoint file that was found to be best during training, whatever criterion
        was applied there. This is necessary since for some models the checkpoint is in a subfolder of the checkpoint
        folder.
        """
        # absolute path is required for registering the model.
        absolute_checkpoint_path = Path(fixed_paths.repository_root_directory(),
                                        DEFAULT_AML_CHECKPOINT_DIR,
                                        self.best_checkpoint_filename_with_suffix)
        if absolute_checkpoint_path.is_file():
            return absolute_checkpoint_path

        absolute_checkpoint_path_parent = Path(fixed_paths.repository_root_directory().parent,
                                               DEFAULT_AML_CHECKPOINT_DIR,
                                               self.best_checkpoint_filename_with_suffix)
        if absolute_checkpoint_path_parent.is_file():
            return absolute_checkpoint_path_parent

        checkpoint_path = Path(self.checkpoint_folder, self.best_checkpoint_filename_with_suffix)
        if checkpoint_path.is_file():
            return checkpoint_path

        checkpoint_path = get_best_checkpoint_path(self.checkpoint_folder)
        if checkpoint_path.is_file():
            return checkpoint_path

        raise ValueError("Path to best checkpoint not found")

    def get_dataloader_kwargs(self) -> dict:
        num_cpus = os.cpu_count()
        assert num_cpus is not None  # for mypy
        logging.info(f"os.cpu_count()={num_cpus}")
        num_devices = self.num_gpus_per_node()
        # We ensure num_devices is not 0 for non-GPU machines
        # to avoid division by zero error when computing `workers_per_gpu`
        workers_per_gpu = num_cpus // (num_devices or 1)
        workers_per_gpu = min(self.max_num_workers, workers_per_gpu)
        print(f"Using {workers_per_gpu} data loader worker processes per GPU")
        dataloader_kwargs = dict(num_workers=workers_per_gpu, pin_memory=True)
        return dataloader_kwargs

    def get_transforms_dict(self, image_key: str) -> Optional[Dict[ModelKey, Union[Callable, None]]]:
        """Returns the image transforms that the training, validation, and test dataloaders should use.

        For reproducible results, those may need to be made deterministic via setting a fixed
        random see. See `SlidesDataModule` for an example how to achieve that."""
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

    def setup(self) -> None:
        super().setup()
        # Fine-tuning requires tiles to be loaded on-the-fly, hence, caching is disabled by default.
        # When tune_encoder and is_caching are both set, below lines should disable caching automatically.
        if self.tune_encoder:
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
            encoder = create_from_matching_params(self, EncoderParams).get_encoder(self.outputs_folder)
            transform = Compose([
                LoadTilesBatchd(image_key, progress=True),
                EncodeTilesBatchd(image_key, encoder, chunk_size=self.encoding_chunk_size)  # type: ignore
            ])
        else:
            transform = LoadTilesBatchd(image_key, progress=True)  # type: ignore
        # in case the transformations for training contain augmentations, val and test transform will be different
        return {ModelKey.TRAIN: transform, ModelKey.VAL: transform, ModelKey.TEST: transform}

    def create_model(self) -> TilesDeepMILModule:
        self.data_module = self.get_data_module()
        outputs_handler = self.get_outputs_handler()
        deepmil_module = TilesDeepMILModule(label_column=self.data_module.train_dataset.label_column,
                                            n_classes=self.data_module.train_dataset.n_classes,
                                            class_names=self.class_names,
                                            class_weights=self.data_module.class_weights,
                                            encoder_params=create_from_matching_params(self, EncoderParams),
                                            pooling_params=create_from_matching_params(self, PoolingParams),
                                            classifier_parms=create_from_matching_params(self, ClassifierParams),
                                            optimizer_params=create_from_matching_params(self, OptimizerParams),
                                            outputs_folder=self.outputs_folder,
                                            outputs_handler=outputs_handler,
                                            analyse_loss=self.analyse_loss,
                                            )
        deepmil_module.transfer_weights(self.trained_weights_path)
        outputs_handler.set_slides_dataset_for_plots_handlers(self.get_slides_dataset())
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
        outputs_handler = self.get_outputs_handler()
        deepmil_module = SlidesDeepMILModule(label_column=SlideKey.LABEL,
                                             n_classes=self.data_module.train_dataset.n_classes,
                                             class_names=self.class_names,
                                             class_weights=self.data_module.class_weights,
                                             outputs_folder=self.outputs_folder,
                                             encoder_params=create_from_matching_params(self, EncoderParams),
                                             pooling_params=create_from_matching_params(self, PoolingParams),
                                             classifier_params=create_from_matching_params(self, ClassifierParams),
                                             optimizer_params=create_from_matching_params(self, OptimizerParams),
                                             outputs_handler=outputs_handler,
                                             analyse_loss=self.analyse_loss,

                                             )
        deepmil_module.transfer_weights(self.trained_weights_path)
        outputs_handler.set_slides_dataset_for_plots_handlers(self.get_slides_dataset())
        return deepmil_module
