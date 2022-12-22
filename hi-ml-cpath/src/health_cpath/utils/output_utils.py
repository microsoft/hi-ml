#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from copy import deepcopy
import shutil
from itertools import chain
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import logging

from ruamel.yaml import YAML
from torchmetrics.metric import Metric

from health_azure.utils import replace_directory
from health_cpath.datasets.base_dataset import SlidesDataset
from health_cpath.preprocessing.loading import LoadingParams
from health_cpath.utils.plots_utils import DeepMILPlotsHandler, TilesSelector
from health_cpath.utils.naming import MetricsKey, ModelKey, PlotOption, ResultsKey

OUTPUTS_CSV_FILENAME = "test_output.csv"
VAL_OUTPUTS_SUBDIR = "val"
PREV_VAL_OUTPUTS_SUBDIR = "val_old"
TEST_OUTPUTS_SUBDIR = "test"
EXTRA_VAL_OUTPUTS_SUBDIR = "extra_val"
EXTRA_PREFIX = "extra_"

AML_OUTPUTS_DIR = "outputs"
AML_LEGACY_TEST_OUTPUTS_CSV = "/".join([AML_OUTPUTS_DIR, OUTPUTS_CSV_FILENAME])
AML_VAL_OUTPUTS_CSV = "/".join([AML_OUTPUTS_DIR, VAL_OUTPUTS_SUBDIR, OUTPUTS_CSV_FILENAME])
AML_TEST_OUTPUTS_CSV = "/".join([AML_OUTPUTS_DIR, TEST_OUTPUTS_SUBDIR, OUTPUTS_CSV_FILENAME])

BatchResultsType = Dict[ResultsKey, Any]
EpochResultsType = List[BatchResultsType]
ResultsType = Dict[ResultsKey, List[Any]]


def validate_class_names(class_names: Optional[Sequence[str]], n_classes: int) -> Tuple[str, ...]:
    """Return valid names for the specified number of classes.

    :param class_names: List of class names. If `None`, will return `('0', '1', ...)`.
    :param n_classes: Number of classes. If `1` (binary), expects `len(class_names) == 2`.
    :return: Validated class names tuple with length `2` for binary classes (`n_classes == 1`), otherwise `n_classes`.
    """
    effective_n_classes = n_classes if n_classes > 1 else 2
    if class_names is None:
        class_names = [str(i) for i in range(effective_n_classes)]
    if len(class_names) != effective_n_classes:
        raise ValueError(f"Mismatch in number of class names ({class_names}) and number"
                         f"of classes ({effective_n_classes})")
    return tuple(class_names)


def validate_slide_datasets_for_plot_options(
    plot_options: Collection[PlotOption], slides_dataset: Optional[SlidesDataset]
) -> None:
    """Validate that the specified plot options are compatible with the specified slides dataset.

    :param plot_options: Plot options to validate.
    :param slides_dataset: Slides dataset to validate against.
    """

    def _validate_slide_plot_option(plot_option: PlotOption) -> None:
        if plot_option in plot_options and not slides_dataset:
            raise ValueError(f"Plot option {plot_option} requires a slides dataset")

    _validate_slide_plot_option(PlotOption.SLIDE_THUMBNAIL)
    _validate_slide_plot_option(PlotOption.ATTENTION_HEATMAP)


def normalize_dict_for_df(dict_old: Dict[ResultsKey, Any]) -> Dict[str, Any]:
    # slide-level dictionaries are processed by making value dimensions uniform and converting to numpy arrays.
    # these steps are required to convert the dictionary to pandas dataframe.
    dict_new: Dict[str, Any] = dict()
    bag_size = len(dict_old[ResultsKey.SLIDE_ID])
    for key, value in dict_old.items():
        if key not in [ResultsKey.CLASS_PROBS, ResultsKey.PROB]:
            if isinstance(value, torch.Tensor):
                value = value.squeeze(0).cpu().numpy()
                if value.ndim == 0:
                    value = np.full(bag_size, fill_value=value)
            dict_new[key] = value
        elif key == ResultsKey.CLASS_PROBS:
            if isinstance(value, torch.Tensor):
                value = value.squeeze(0).cpu().numpy()
                for i in range(len(value)):
                    dict_new[key + str(i)] = np.repeat(value[i], bag_size)
    return dict_new


def gather_results(epoch_results: EpochResultsType) -> EpochResultsType:
    """Gather epoch results across all DDP processes into a single list.

    :param epoch_results: The collected epoch results (i.e. list of batch results) for the current process.
    :return: A list in the same format as `epoch_results`, containing batch results from all DDP processes. Returns
        `epoch_results` unchanged if not in distributed mode.
    """
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            object_list: EpochResultsType = [None] * world_size  # type: ignore
            torch.distributed.all_gather_object(object_list, epoch_results)
            epoch_results = list(chain(*object_list))  # type: ignore
    return epoch_results


def collate_results_on_cpu(epoch_results: EpochResultsType) -> ResultsType:
    """Convert a list of results dictionaries into a dictionary of lists, with all tensors on CPU.

    :param epoch_results: Collected epoch results, whose elements are dictionaries of :py:class:`ResultsKey` to the
        outputs of the respective batch (i.e. indexed as `epoch_results[batch_index][results_key]`).
    :return: A dictionary mapping each :py:class:`ResultsKey` to a list containing the corresponding outputs for every
        batch (i.e. indexed as `collated_results[results_key][batch_index]`). All tensors will have been placed on CPU.
    """
    results: ResultsType = {}
    for key in epoch_results[0].keys():
        results[key] = []
        for batch_results in epoch_results:
            batch_elements = batch_results[key]
            if key == ResultsKey.LOSS:
                batch_elements = [batch_elements]
            batch_elements = [elem.cpu() if isinstance(elem, torch.Tensor) else elem
                              for elem in batch_elements]
            results[key].extend(batch_elements)
    return results


def save_outputs_csv(results: ResultsType, outputs_dir: Path) -> None:
    logging.info("Saving outputs ...")
    # collate at slide level
    list_slide_dicts: List[Dict[ResultsKey, Any]] = []
    # any column can be used here, the assumption is that the first dimension is the N of slides
    for slide_idx in range(len(results[ResultsKey.SLIDE_ID])):
        slide_dict = {key: results[key][slide_idx] for key in results
                      if key not in [ResultsKey.FEATURES, ResultsKey.LOSS]}
        list_slide_dicts.append(slide_dict)

    assert outputs_dir.is_dir(), f"No such dir: {outputs_dir}"
    logging.info(f"Metrics results will be output to {outputs_dir}")
    csv_filename = outputs_dir / OUTPUTS_CSV_FILENAME

    # Collect the list of dictionaries in a list of pandas dataframe and save
    df_list = []
    for slide_dict in list_slide_dicts:
        slide_dict = normalize_dict_for_df(slide_dict)  # type: ignore
        df_list.append(pd.DataFrame.from_dict(slide_dict))
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(csv_filename, mode='w+', header=True)


def save_features(results: ResultsType, outputs_dir: Path) -> None:
    # Collect all features in a list and save
    features_list = [features.squeeze(0).cpu() for features in results[ResultsKey.FEATURES]]
    torch.save(features_list, outputs_dir / 'test_encoded_features.pickle')


class OutputsPolicy:
    """Utility class that defines when to save validation epoch outputs."""

    _BEST_EPOCH_KEY = 'best_epoch'
    _BEST_VALUE_KEY = 'best_value'
    _PRIMARY_METRIC_KEY = 'primary_metric'

    def __init__(self, outputs_root: Path, primary_val_metric: MetricsKey, maximise: bool) -> None:
        """
        :param outputs_root: Root directory where to save a recovery file with best epoch and metric value.
        :param primary_val_metric: Name of the validation metric to track for saving best epoch outputs.
        :param maximise: Whether higher is better for `primary_val_metric`.
        """
        self.outputs_root = outputs_root
        self.primary_val_metric = primary_val_metric
        self.maximise = maximise

        self._init_best_metric()

    @property
    def best_metric_file_path(self) -> Path:
        return self.outputs_root / "best_val_metric.yml"

    def _init_best_metric(self) -> None:
        """Initialise running best metric epoch and value (recovered from disk if available).

        :raises ValueError: If the primary metric name does not match the one saved on disk.
        """
        if self.best_metric_file_path.exists():
            contents = YAML().load(self.best_metric_file_path)
            self._best_metric_epoch = contents[self._BEST_EPOCH_KEY]
            self._best_metric_value = contents[self._BEST_VALUE_KEY]
            if contents[self._PRIMARY_METRIC_KEY] != self.primary_val_metric:
                raise ValueError(f"Expected primary metric '{self.primary_val_metric}', but found "
                                 f"'{contents[self._PRIMARY_METRIC_KEY]}' in {self.best_metric_file_path}")
        else:
            self._best_metric_epoch = 0
            self._best_metric_value = float('-inf') if self.maximise else float('inf')

    def _save_best_metric(self) -> None:
        """Save best metric epoch, value, and name to disk, to allow recovery (e.g. in case of pre-emption)."""
        contents = {self._BEST_EPOCH_KEY: self._best_metric_epoch,
                    self._BEST_VALUE_KEY: self._best_metric_value,
                    self._PRIMARY_METRIC_KEY: self.primary_val_metric.value}
        YAML().dump(contents, self.best_metric_file_path)

    def should_save_validation_outputs(self, metrics_dict: Mapping[MetricsKey, Metric], epoch: int,
                                       is_global_rank_zero: bool = True, on_extra_val: bool = False) -> bool:
        """Determine whether validation outputs should be saved given the current epoch's metrics.

        :param metrics_dict: Current epoch's metrics dictionary from
            :py:class:`~health_cpath.models.deepmil.DeepMILModule`.
        :param epoch: Current epoch number.
        :param is_global_rank_zero: Whether this is the global rank-0 process in distributed scenarios.
            Set to `True` (default) if running a single process.
        :param on_extra_val: Whether this is an extra validation epoch (e.g. after training).
        :return: Whether this is the best validation epoch so far.
        """
        if on_extra_val:
            return False
        metric = metrics_dict[self.primary_val_metric]
        # If the metric hasn't been updated we don't want to save it
        if not metric._update_called:
            logging.warning("Encountered metric that hasn't been updated. Not saving.")
            return False
        # The metric needs to be computed on all ranks to allow synchronisation
        metric_value = float(metric.compute())

        # Validation outputs and best metric should be saved only by the global rank-0 process
        if not is_global_rank_zero:
            return False

        if self.maximise:
            is_best = metric_value > self._best_metric_value
        else:
            is_best = metric_value < self._best_metric_value

        if is_best:
            self._best_metric_value = metric_value
            self._best_metric_epoch = epoch
            self._save_best_metric()

        return is_best

    def should_save_test_outputs(self, is_global_rank_zero: bool = True) -> bool:
        """Determine whether test outputs should be saved.

        :param is_global_rank_zero: Whether this is the global rank-0 process in distributed scenarios.
            Set to `True` (default) if running a single process.
        """
        # This is implemented as a method in case we want to add custom logic in the future
        return is_global_rank_zero


class DeepMILOutputsHandler:
    """Class that manages writing validation and test outputs for DeepMIL models."""

    def __init__(self, outputs_root: Path, n_classes: int, tile_size: int, loading_params: LoadingParams,
                 class_names: Optional[Sequence[str]], primary_val_metric: MetricsKey,
                 maximise: bool, val_plot_options: Collection[PlotOption],
                 test_plot_options: Collection[PlotOption], val_set_is_dist: bool = True,
                 save_intermediate_outputs: bool = True) -> None:
        """
        :param outputs_root: Root directory where to save all produced outputs.
        :param n_classes: Number of MIL classes (set `n_classes=1` for binary).
        :param tile_size: The size of each tile.
        :param loading_params: Parameters for loading WSI to create plots. This paramter is passed to PlotsHandler.
        :param class_names: List of class names. For binary (`n_classes == 1`), expects `len(class_names) == 2`.
            If `None`, will return `('0', '1', ...)`.
        :param primary_val_metric: Name of the validation metric to track for saving best epoch outputs.
        :param maximise: Whether higher is better for `primary_val_metric`.
        :param val_plot_options: The desired plot options for validation time.
        :param test_plot_options: The desired plot options for test time.
        :param val_set_is_dist: If True, the validation set is distributed across processes. Otherwise, the validation
            set is replicated on each process. This shouldn't affect the results, as we take the mean of the validation
            set metrics across processes. This is only relevant for the outputs_handler, which needs to know whether to
            gather the validation set outputs across processes or not before saving them.
        :param save_intermediate_outputs: Whether to save intermediate outputs (e.g. after each epoch).
        """
        self.outputs_root = outputs_root
        self.n_classes = n_classes
        self.class_names = validate_class_names(class_names, self.n_classes)
        self.outputs_policy = OutputsPolicy(outputs_root=outputs_root,
                                            primary_val_metric=primary_val_metric,
                                            maximise=maximise)
        self.save_intermediate_outputs = save_intermediate_outputs
        self.tiles_selector: Optional[TilesSelector] = None
        self.val_plots_handler = DeepMILPlotsHandler(
            plot_options=val_plot_options,
            tile_size=tile_size,
            class_names=self.class_names,
            stage=ModelKey.VAL,
            loading_params=deepcopy(loading_params),
        )
        self.test_plots_handler = DeepMILPlotsHandler(
            plot_options=test_plot_options,
            tile_size=tile_size,
            class_names=self.class_names,
            stage=ModelKey.TEST,
            loading_params=deepcopy(loading_params),
        )
        self.val_set_is_dist = val_set_is_dist

    @property
    def validation_outputs_dir(self) -> Path:
        return self.outputs_root / VAL_OUTPUTS_SUBDIR

    @property
    def extra_validation_outputs_dir(self) -> Path:
        return self.outputs_root / EXTRA_VAL_OUTPUTS_SUBDIR

    @property
    def previous_validation_outputs_dir(self) -> Path:
        return self.validation_outputs_dir.with_name(PREV_VAL_OUTPUTS_SUBDIR)

    @property
    def test_outputs_dir(self) -> Path:
        return self.outputs_root / TEST_OUTPUTS_SUBDIR

    def set_slides_dataset_for_plots_handlers(self, slides_dataset: Optional[SlidesDataset]) -> None:
        validate_slide_datasets_for_plot_options(self.test_plots_handler.plot_options, slides_dataset)
        validate_slide_datasets_for_plot_options(self.val_plots_handler.plot_options, slides_dataset)
        self.test_plots_handler.slides_dataset = slides_dataset
        self.val_plots_handler.slides_dataset = slides_dataset

    def should_gather_tiles(self, plots_handler: DeepMILPlotsHandler) -> bool:
        return PlotOption.TOP_BOTTOM_TILES in plots_handler.plot_options and self.tiles_selector is not None

    def _save_outputs(self, epoch_results: EpochResultsType, outputs_dir: Path, stage: ModelKey = ModelKey.VAL) -> None:
        """Trigger the rendering and saving of DeepMIL outputs and figures.

        :param epoch_results: Aggregated results from all epoch batches.
        :param outputs_dir: Specific directory into which outputs should be saved (different for validation and test).
        :param stage: The stage of the model (e.g. `ModelKey.VAL` or `ModelKey.TEST`).
        """
        # outputs object consists of a list of dictionaries (of metadata and results, including encoded features)
        # It can be indexed as outputs[batch_idx][batch_key][bag_idx][tile_idx]
        # example of batch_key ResultsKey.SLIDE_ID_COL
        # for batch keys that contains multiple values for slides e.g. ResultsKey.BAG_ATTN_COL
        # outputs[batch_idx][batch_key][bag_idx][tile_idx]
        # contains the tile value
        # TODO: Synchronise this with checkpoint saving (e.g. on_save_checkpoint())
        results = collate_results_on_cpu(epoch_results)
        outputs_dir.mkdir(exist_ok=True, parents=True)
        save_outputs_csv(results, outputs_dir)

        plots_handler = self.val_plots_handler if stage == ModelKey.VAL else self.test_plots_handler
        plots_handler.save_plots(outputs_dir, self.tiles_selector, results)

    def save_validation_outputs(self, epoch_results: EpochResultsType, metrics_dict: Mapping[MetricsKey, Metric],
                                epoch: int, is_global_rank_zero: bool = True, on_extra_val: bool = False) -> None:
        """Render and save validation epoch outputs, according to the configured :py:class:`OutputsPolicy`.

        :param epoch_results: Aggregated results from all epoch batches, as passed to :py:meth:`validation_epoch_end()`.
        :param metrics_dict: Current epoch's validation metrics dictionary from
            :py:class:`~health_cpath.models.deepmil.DeepMILModule`.
        :param is_global_rank_zero: Whether this is the global rank-0 process in distributed scenarios.
            Set to `True` (default) if running a single process.
        :param epoch: Current epoch number.
        :param on_extra_val: Whether this is an extra validation epoch (e.g. after training).
        """
        # All DDP processes must reach this point to allow synchronising epoch results if val_set_is_dist is True
        if self.val_set_is_dist:
            epoch_results = gather_results(epoch_results)

            if self.should_gather_tiles(self.val_plots_handler):
                self.tiles_selector.gather_selected_tiles_across_devices()  # type: ignore

        # Only global rank-0 process should actually render and save the outputs
        # We also want to save the plots of the extra validation epoch
        if self.save_intermediate_outputs and self.outputs_policy.should_save_validation_outputs(
            metrics_dict, epoch, is_global_rank_zero, on_extra_val
        ):
            # First move existing outputs to a temporary directory, to avoid mixing
            # outputs of different epochs in case writing fails halfway through
            if self.validation_outputs_dir.exists():
                replace_directory(source=self.validation_outputs_dir,
                                  target=self.previous_validation_outputs_dir)

            self._save_outputs(epoch_results, self.validation_outputs_dir, ModelKey.VAL)

            # Writing completed successfully; delete temporary back-up
            if self.previous_validation_outputs_dir.exists():
                shutil.rmtree(self.previous_validation_outputs_dir, ignore_errors=True)
        elif on_extra_val and is_global_rank_zero:
            self._save_outputs(epoch_results, self.extra_validation_outputs_dir, ModelKey.VAL)

        # Reset the top and bottom slides heaps
        if self.should_gather_tiles(self.val_plots_handler):
            self.tiles_selector._clear_cached_slides_heaps()  # type: ignore

    def save_test_outputs(self, epoch_results: EpochResultsType, is_global_rank_zero: bool = True) -> None:
        """Render and save test epoch outputs.

        :param epoch_results: Aggregated results from all epoch batches, as passed to :py:meth:`test_epoch_end()`.
        :param metrics_dict: Test metrics dictionary from :py:class:`~health_cpath.models.deepmil.DeepMILModule`.
        :param is_global_rank_zero: Whether this is the global rank-0 process in distributed scenarios.
            Set to `True` (default) if running a single process.
        """
        # All DDP processes must reach this point to allow synchronising epoch results
        gathered_epoch_results = gather_results(epoch_results)
        if self.should_gather_tiles(self.test_plots_handler):
            self.tiles_selector.gather_selected_tiles_across_devices()  # type: ignore

        # Only global rank-0 process should actually render and save the outputs-
        if self.outputs_policy.should_save_test_outputs(is_global_rank_zero):
            self._save_outputs(gathered_epoch_results, self.test_outputs_dir, ModelKey.TEST)
