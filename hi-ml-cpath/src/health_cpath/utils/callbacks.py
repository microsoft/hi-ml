#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
import torch
import param
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from health_cpath.models.deepmil import BaseDeepMILModule
from health_cpath.utils.naming import ModelKey, ResultsKey
from health_cpath.utils.output_utils import BatchResultsType

LossCacheDictType = Dict[ResultsKey, List]
LossDictType = Dict[str, List]


class LossCallbackParams(param.Parameterized):
    """Parameters class to group all attributes for loss values analysis callback"""

    analyse_loss: bool = param.Boolean(
        False,
        doc="If True, will use `LossValuesAnalysisCallback` to cache loss values per slide/epoch for further analysis."
        "See `LossValuesAnalysisCallback` for more details.",
    )
    loss_analysis_patience: int = param.Integer(
        0,
        bounds=(0, None),
        doc="Number of epochs to wait before starting loss values per slide analysis. Default: 0, It will start"
        "caching loss values per epoch immediately. Use loss_analysis_patience=n>0 to wait for a few epochs "
        "before starting the analysis.",
    )
    loss_analysis_epochs_interval: int = param.Integer(
        1,
        bounds=(1, None),
        doc="Epochs interval to save loss values. Default: 1, It will save loss values every epoch. Use "
        "loss_analysis_epochs_interval=n>1 to save loss values every n epochs.",
    )
    num_slides_scatter: int = param.Integer(
        20,
        bounds=(1, None),
        doc="Number of slides to plot in the scatter plot. Default: 10, It will plot a scatter of the 10 slides "
        "with highest/lowest loss values across epochs.",
    )
    num_slides_heatmap: int = param.Integer(
        20,
        bounds=(1, None),
        doc="Number of slides to plot in the heatmap plot. Default: 20, It will plot the loss values for the 20 slides "
        "with highest/lowest loss values.",
    )
    save_tile_ids: bool = param.Boolean(
        True,
        doc="If True, will save the tile ids for each bag in the loss cache. Default: True. If False, will only save "
        "the slide ids and their loss values.",
    )
    log_exceptions: bool = param.Boolean(
        True,
        doc="If True, will log exceptions raised during loss values analysis. Default: True. If False, will raise the "
        "intercepted exceptions.",
    )


class LossAnalysisCallback(Callback):
    """Callback to analyse loss values per slide across epochs. It saves the loss values per slide in a csv file every n
    epochs and plots the evolution of the loss values per slide in a heatmap as well as the slides with the
    highest/lowest loss values per epoch in a scatter plot."""

    TILES_JOIN_TOKEN = "$"
    X_LABEL, Y_LABEL = "Epoch", "Slide ids"
    TOP, BOTTOM = "top", "bottom"
    HIGHEST, LOWEST = "highest", "lowest"

    def __init__(
        self,
        outputs_folder: Path,
        max_epochs: int = 30,
        patience: int = 0,
        epochs_interval: int = 1,
        num_slides_scatter: int = 10,
        num_slides_heatmap: int = 20,
        save_tile_ids: bool = False,
        log_exceptions: bool = True,
        create_outputs_folders: bool = True,
    ) -> None:
        """

        :param outputs_folder: Path to the folder where the outputs will be saved.
        :param patience: Number of epochs to wait before starting to cache loss values, defaults to 0.
        :param epochs_interval: Epochs interval to save loss values, defaults to 1.
        :param max_epochs: Maximum number of epochs to train, defaults to 30.
        :param num_slides_scatter: Number of slides to plot in the scatter plot, defaults to 10.
        :param num_slides_heatmap: Number of slides to plot in the heatmap, defaults to 20.
        :param save_tile_ids: If True, will save the tile ids of the tiles in the bag in the loss cache, defaults to
        False. This is useful to analyse the tiles that are contributing to the loss value of a slide.
        :param log_exceptions: If True, will log exceptions raised during loss values analysis, defaults to True. If
        False will raise the intercepted exceptions.
        :param create_outputs_folders: If True, will create the output folders if they don't exist, defaults to True.
        """

        self.patience = patience
        self.epochs_interval = epochs_interval
        self.max_epochs = max_epochs
        self.num_slides_scatter = num_slides_scatter
        self.num_slides_heatmap = num_slides_heatmap
        self.save_tile_ids = save_tile_ids
        self.log_exceptions = log_exceptions

        self.outputs_folder = outputs_folder / "loss_values_callback"
        if create_outputs_folders:
            self.create_outputs_folders()

        self.loss_cache = self.reset_loss_cache()
        self.epochs_range = list(range(self.patience, self.max_epochs, self.epochs_interval))

        self.nan_slides: List[str] = []
        self.anomaly_slides: List[str] = []

    @property
    def cache_folder(self) -> Path:
        return self.outputs_folder / "loss_cache"

    @property
    def scatter_folder(self) -> Path:
        return self.outputs_folder / "loss_scatter"

    @property
    def heatmap_folder(self) -> Path:
        return self.outputs_folder / "loss_heatmap"

    @property
    def stats_folder(self) -> Path:
        return self.outputs_folder / "loss_stats"

    @property
    def anomalies_folder(self) -> Path:
        return self.outputs_folder / "loss_anomalies"

    def create_outputs_folders(self) -> None:
        folders = [
            self.cache_folder,
            self.scatter_folder,
            self.heatmap_folder,
            self.stats_folder,
            self.anomalies_folder,
        ]
        for folder in folders:
            for stage in [ModelKey.TRAIN.value, ModelKey.VAL.value]:
                os.makedirs(folder / stage, exist_ok=True)

    def reset_loss_cache(self) -> LossCacheDictType:
        keys = [ResultsKey.LOSS, ResultsKey.SLIDE_ID]
        if self.save_tile_ids:
            keys.append(ResultsKey.TILE_ID)
        return {key: [] for key in keys}

    def _get_filename(self, filename: str, epoch: int, order: Optional[str] = None) -> str:
        zero_filled_epoch = str(epoch).zfill(len(str(self.max_epochs)))
        filename = filename.format(zero_filled_epoch, order) if order else filename.format(zero_filled_epoch)
        return filename

    def get_loss_cache_file(self, epoch: int, stage: str) -> Path:
        return self.cache_folder / stage / self._get_filename(filename="epoch_{}.csv", epoch=epoch)

    def get_all_epochs_loss_cache_file(self, stage: str) -> Path:
        return self.cache_folder / stage / "all_epochs.csv"

    def get_loss_stats_file(self, stage: str) -> Path:
        return self.stats_folder / stage / "loss_stats.csv"

    def get_loss_ranks_file(self, stage: str) -> Path:
        return self.stats_folder / stage / "loss_ranks.csv"

    def get_loss_ranks_stats_file(self, stage: str) -> Path:
        return self.stats_folder / stage / "loss_ranks_stats.csv"

    def get_nan_slides_file(self, stage: str) -> Path:
        return self.anomalies_folder / stage / "nan_slides.txt"

    def get_anomaly_slides_file(self, stage: str) -> Path:
        return self.anomalies_folder / stage / "anomaly_slides.txt"

    def get_scatter_plot_file(self, order: str, stage: str) -> Path:
        return self.scatter_folder / stage / f"slides_with_{order}_loss_values.png"

    def get_heatmap_plot_file(self, epoch: int, order: str, stage: str) -> Path:
        return self.heatmap_folder / stage / self._get_filename("epoch_{}_{}_slides.png", epoch, order)

    def read_loss_cache(self, epoch: int, stage: str, idx_col: Optional[ResultsKey] = None) -> pd.DataFrame:
        columns = [ResultsKey.SLIDE_ID, ResultsKey.LOSS]
        return pd.read_csv(self.get_loss_cache_file(epoch, stage), index_col=idx_col, usecols=columns)

    def should_cache_loss_values(self, current_epoch: int) -> bool:
        current_epoch = current_epoch + 1
        first_time = (current_epoch - self.patience) == 1
        return first_time or (
            current_epoch > self.patience and (current_epoch - self.patience) % self.epochs_interval == 0
        )

    def merge_loss_caches(self, loss_caches: List[LossCacheDictType]) -> LossCacheDictType:
        """Merges the loss caches from all the workers into a single loss cache"""
        loss_cache = self.reset_loss_cache()
        for loss_cache_per_device in loss_caches:
            for key in loss_cache.keys():
                loss_cache[key].extend(loss_cache_per_device[key])
        return loss_cache

    def gather_loss_cache(self, rank: int) -> None:
        """Gathers the loss cache from all the workers"""
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                loss_caches = [None] * world_size
                torch.distributed.all_gather_object(loss_caches, self.loss_cache)
                if rank == 0:
                    self.loss_cache = self.merge_loss_caches(loss_caches)  # type: ignore

    def save_loss_cache(self, current_epoch: int, stage: str) -> None:
        """Saves the loss cache to a csv file"""
        loss_cache_df = pd.DataFrame(self.loss_cache)
        # Some slides may be appear multiple times in the loss cache in DDP mode. The Distributed Sampler duplicate
        # some slides to even out the number of samples per device, so we only keep the first occurrence.
        loss_cache_df.drop_duplicates(subset=ResultsKey.SLIDE_ID, inplace=True, keep="first")
        loss_cache_df = loss_cache_df.sort_values(by=ResultsKey.LOSS, ascending=False)
        loss_cache_df.to_csv(self.get_loss_cache_file(current_epoch, stage), index=False)

    def _select_values_for_epoch(
        self,
        keys: List[ResultsKey],
        epoch: int,
        stage: str,
        high: Optional[bool] = None,
        num_values: Optional[int] = None
    ) -> List[np.ndarray]:
        loss_cache = self.read_loss_cache(epoch, stage)
        return_values = []
        for key in keys:
            values = loss_cache[key].values
            if high is not None:
                assert num_values is not None, "num_values must be specified if high is specified"
                if high:
                    return_values.append(values[:num_values])
                elif not high:
                    return_values.append(values[-num_values:])
            else:
                return_values.append(values)
        return return_values

    def select_slides_for_epoch(
        self, epoch: int, stage: str, high: Optional[bool] = None, num_slides: Optional[int] = None
    ) -> np.ndarray:
        """Selects slides in ascending/descending order of loss values at a given epoch

        :param epoch: The epoch to select the slides from.
        :param stage: The model's stage (train/val).
        :param high: If True, selects the slides with the highest loss values, else selects the slides with the lowest
            loss values. If None, selects all slides.
        :param num_slides: The number of slides to select. If None, selects all slides.
        """
        return self._select_values_for_epoch([ResultsKey.SLIDE_ID], epoch, stage, high, num_slides)[0]

    def select_slides_losses_for_epoch(
        self, epoch: int, stage: str, high: Optional[bool] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Selects slides and loss values of slides in ascending/descending order of loss at a given epoch

        :param epoch: The epoch to select the slides from.
        :param stage: The model's stage (train/val).
        :param high: If True, selects the slides with the highest loss values, else selects the slides with the lowest
            loss values. If None, selects all slides.
        """
        keys = [ResultsKey.SLIDE_ID, ResultsKey.LOSS]
        return_values = self._select_values_for_epoch(keys, epoch, stage, high, self.num_slides_scatter)
        return return_values[0], return_values[1]

    def select_all_losses_for_selected_slides(self, slides: np.ndarray, stage: str) -> LossDictType:
        """Selects the loss values for a given set of slides"""

        slides_loss_values: LossDictType = {slide_id: [] for slide_id in slides}
        for epoch in self.epochs_range:
            loss_cache = self.read_loss_cache(epoch, stage, idx_col=ResultsKey.SLIDE_ID)
            for slide_id in slides:
                slides_loss_values[slide_id].append(loss_cache.loc[slide_id, ResultsKey.LOSS])
        return slides_loss_values

    def select_slides_and_losses_across_epochs(
        self, stage: str, high: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Selects the slides with the highest/lowest loss values across epochs

        :param stage: The model's stage (train/val).
        :param high: If True, selects the slides with the highest loss values, else selects the slides with the lowest
        loss values.
        """
        slides = []
        slides_loss = []
        for epoch in self.epochs_range:
            epoch_slides, epoch_slides_loss = self.select_slides_losses_for_epoch(epoch, stage, high)
            slides.append(epoch_slides)
            slides_loss.append(epoch_slides_loss)

        return np.array(slides).T, np.array(slides_loss).T

    def save_slide_ids(self, slide_ids: List[str], path: Path) -> None:
        """Dumps the slides ids in a txt file."""
        if slide_ids:
            with open(path, "w") as f:
                for slide_id in slide_ids:
                    f.write(f"{slide_id}\n")

    def sanity_check_loss_values(self, loss_values: LossDictType, stage: str) -> None:
        """Checks if there are any NaNs or any other potential annomalies in the loss values.

        :param loss_values: The loss values for all slides.
        :param stage: The model's stage (train/val).
        """
        # We don't want any of these exceptions to interrupt validation. So we catch them and log them.
        loss_values_copy = loss_values.copy()
        for slide_id, loss in loss_values_copy.items():
            try:
                if np.isnan(loss).any():
                    logging.warning(f"NaNs found in loss values for slide {slide_id}.")
                    self.nan_slides.append(slide_id)
                    loss_values.pop(slide_id)
            except Exception as e:
                logging.warning(f"Error while checking for NaNs in loss values for slide {slide_id} with error {e}.")
                logging.warning(f"Loss values that caused the issue: {loss}")
                self.anomaly_slides.append(slide_id)
                loss_values.pop(slide_id)
        self.save_slide_ids(self.nan_slides, self.get_nan_slides_file(stage))
        self.save_slide_ids(self.anomaly_slides, self.get_anomaly_slides_file(stage))

    def save_loss_ranks(self, slides_loss_values: LossDictType, stage: str) -> None:
        """Saves the loss ranks for each slide across all epochs and their respective statistics in csv files."""

        loss_df = pd.DataFrame(slides_loss_values).T
        loss_df.index.names = [ResultsKey.SLIDE_ID.value]
        loss_df.to_csv(self.get_all_epochs_loss_cache_file(stage))

        loss_stats = loss_df.T.describe().T.sort_values(by="mean", ascending=False)
        loss_stats.to_csv(self.get_loss_stats_file(stage))

        loss_ranks = loss_df.rank(ascending=False)
        loss_ranks.to_csv(self.get_loss_ranks_file(stage))

        loss_ranks_stats = loss_ranks.T.describe().T.sort_values("mean", ascending=True)
        loss_ranks_stats.to_csv(self.get_loss_ranks_stats_file(stage))

    def plot_slides_loss_scatter(
        self,
        slides: np.ndarray,
        slides_loss: np.ndarray,
        stage: str,
        high: bool = True,
        figsize: Tuple[float, float] = (30, 30),
    ) -> None:
        """Plots the slides with the highest/lowest loss values across epochs in a scatter plot

        :param slides: The slides ids.
        :param slides_loss: The loss values for each slide.
        :param stage: The model's stage (train/val).
        :param figsize: The figure size, defaults to (30, 30)
        :param high: If True, plots the slides with the highest loss values, else plots the slides with the lowest loss.
        """
        label = self.TOP if high else self.BOTTOM
        plt.figure(figsize=figsize)
        for i in range(self.num_slides_scatter - 1, -1, -1):
            plt.scatter(self.epochs_range, slides[i], label=f"{label}_{i+1}")
            for loss, epoch, slide in zip(slides_loss[i], self.epochs_range, slides[i]):
                plt.annotate(f"{loss:.3f}", (epoch, slide))
        plt.xlabel(self.X_LABEL)
        plt.ylabel(self.Y_LABEL)
        order = self.HIGHEST if high else self.LOWEST
        plt.title(f"Slides with {order} loss values per epoch.")
        plt.xticks(self.epochs_range)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, linestyle="--")
        plt.savefig(self.get_scatter_plot_file(order, stage), bbox_inches="tight")

    def plot_loss_heatmap_for_slides_of_epoch(
        self,
        slides_loss_values: LossDictType,
        epoch: int,
        stage: str,
        high: bool,
        figsize: Tuple[float, float] = (15, 15)
    ) -> None:
        """Plots the loss values for each slide across all epochs in a heatmap.

        :param slides_loss_values: The loss values for each slide across all epochs.
        :param epoch: The epoch used to select the slides.
        :param stage: The model's stage (train/val).
        :param high: If True, plots the slides with the highest loss values, else plots the slides with the lowest loss.
        :param figsize: The figure size, defaults to (15, 15)
        """
        order = self.HIGHEST if high else self.LOWEST
        loss_values = np.array(list(slides_loss_values.values()))
        slides = list(slides_loss_values.keys())
        plt.figure(figsize=figsize)
        _ = sns.heatmap(loss_values, linewidth=0.5, annot=True, yticklabels=slides)
        plt.xlabel(self.X_LABEL)
        plt.ylabel(self.Y_LABEL)
        plt.title(f"Loss values evolution for {order} slides of epoch {epoch}")
        plt.savefig(self.get_heatmap_plot_file(epoch, order, stage), bbox_inches="tight")

    def _cache_loss_slides(self, trainer: Trainer, outputs: BatchResultsType, batch: Dict) -> None:
        if self.should_cache_loss_values(trainer.current_epoch):
            self.loss_cache[ResultsKey.LOSS].extend(outputs[ResultsKey.LOSS_PER_SAMPLE])
            self.loss_cache[ResultsKey.SLIDE_ID].extend([slides[0] for slides in batch[ResultsKey.SLIDE_ID]])
            if self.save_tile_ids:
                self.loss_cache[ResultsKey.TILE_ID].extend(
                    [self.TILES_JOIN_TOKEN.join(tiles) for tiles in batch[ResultsKey.TILE_ID]]
                )

    def synchronise_processes_and_reset(self, trainer: Trainer, pl_module: BaseDeepMILModule, stage: str) -> None:
        if self.should_cache_loss_values(trainer.current_epoch):
            self.gather_loss_cache(rank=pl_module.global_rank)
            if pl_module.global_rank == 0:
                self.save_loss_cache(trainer.current_epoch, stage)
        self.loss_cache = self.reset_loss_cache()  # reset loss cache for all processes

    def save_loss_outliers_analaysis_results(self, stage: str) -> None:
        all_slides = self.select_slides_for_epoch(epoch=0, stage=stage)
        all_loss_values_per_slides = self.select_all_losses_for_selected_slides(all_slides, stage=stage)

        self.sanity_check_loss_values(all_loss_values_per_slides, stage=stage)
        self.save_loss_ranks(all_loss_values_per_slides, stage=stage)

        top_slides, top_slides_loss = self.select_slides_and_losses_across_epochs(stage, high=True)
        self.plot_slides_loss_scatter(top_slides, top_slides_loss, stage, high=True)

        bottom_slides, bottom_slides_loss = self.select_slides_and_losses_across_epochs(stage, high=False)
        self.plot_slides_loss_scatter(bottom_slides, bottom_slides_loss, stage, high=False)

        for epoch in self.epochs_range:
            epoch_slides = self.select_slides_for_epoch(epoch, stage=stage)

            top_slides = epoch_slides[:self.num_slides_heatmap]
            top_slides_loss_values = self.select_all_losses_for_selected_slides(top_slides, stage=stage)
            self.plot_loss_heatmap_for_slides_of_epoch(top_slides_loss_values, epoch, stage, high=True)

            bottom_slides = epoch_slides[-self.num_slides_heatmap:]
            bottom_slides_loss_values = self.select_all_losses_for_selected_slides(bottom_slides, stage=stage)
            self.plot_loss_heatmap_for_slides_of_epoch(bottom_slides_loss_values, epoch, stage, high=False)

        self.loss_cache = self.reset_loss_cache()

    def handle_loss_exceptions(self, stage: str, exception: Exception) -> None:
        if self.log_exceptions:
            # If something goes wrong, we don't want to crash the training. We just log the error and carry on
            # validation.
            logging.warning(f"Error while detecting {stage} loss values outliers: {exception}")
        else:
            # If we want to debug the error, we raise it. This will crash the training. This is useful when
            # running smoke tests.
            raise Exception(f"Error while detecting {stage} loss values outliers: {exception}")

    def on_train_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: BaseDeepMILModule,
        outputs: BatchResultsType,
        batch: Dict,
        batch_idx: int,
        unused: int = 0
    ) -> None:
        """Caches train loss values per slide at each training step in a local variable self.loss_cache."""
        self._cache_loss_slides(trainer, outputs, batch)

    def on_validation_batch_end(  # type: ignore
        self,
        trainer: Trainer,
        pl_module: BaseDeepMILModule,
        outputs: BatchResultsType,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        """Caches validation loss values per slide at each training step in a local variable self.loss_cache."""
        self._cache_loss_slides(trainer, outputs, batch)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        """Gathers loss values per slide from all processes at the end of each epoch and saves them to a csv file."""
        self.synchronise_processes_and_reset(trainer, pl_module, ModelKey.TRAIN.value)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        """Gathers loss values per slide from all processes at the end of each epoch and saves them to a csv file."""
        self.synchronise_processes_and_reset(trainer, pl_module, ModelKey.VAL.value)

    def on_train_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        """Hook called at the end of training. Plot the loss heatmap and scratter plots after ranking the slides by loss
        values."""
        if pl_module.global_rank == 0:
            try:
                self.save_loss_outliers_analaysis_results(stage=ModelKey.TRAIN.value)
            except Exception as e:
                self.handle_loss_exceptions(stage=ModelKey.TRAIN.value, exception=e)

    def on_validation_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        """Hook called at the end of validation. Plot the loss heatmap and scratter plots after ranking the slides by
        loss values."""
        epoch = trainer.current_epoch
        if pl_module.global_rank == 0 and not pl_module._run_extra_val_epoch and epoch == self.max_epochs:
            try:
                self.save_loss_outliers_analaysis_results(stage=ModelKey.VAL.value)
            except Exception as e:
                self.handle_loss_exceptions(stage=ModelKey.VAL.value, exception=e)
