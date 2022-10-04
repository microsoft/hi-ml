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
from typing import Dict, List, Optional, Tuple
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from health_cpath.models.deepmil import BaseDeepMILModule
from health_cpath.utils.naming import ResultsKey

TILES_JOIN_TOKEN = "$"

LOSS_VALUES_FILENAME = "epoch_{}.csv"
ALL_EPOCHS_FILENAME = "all_epochs.csv"
LOSS_RANKS_FILENAME = "loss_ranks.csv"
LOSS_STATS_FILENAME = "loss_stats.csv"
LOSS_RANKS_STATS_FILENAME = "loss_ranks_stats.csv"

SCATTER_PLOT_FILENAME = "slides_with_{}_loss_values.png"
HEATMAP_PLOT_FILENAME = "epoch_{}_{}_slides.png"

NAN_SLIDES_FILENAME = "nan_slides.txt"
EXCEPTION_SLIDES_FILENAME = "anomaly_slides.txt"

X_LABEL, Y_LABEL = "Epoch", "Slide ids"
TOP, BOTTOM = "top", "bottom"
HIGHEST, LOWEST = "highest", "lowest"

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


class LossAnalysisCallback(Callback):
    """Callback to analyse loss values per slide across epochs. It saves the loss values per slide in a csv file every n
    epochs and plots the evolution of the loss values per slide in a heatmap as well as the slides with the
    highest/lowest loss values per epoch in a scatter plot."""

    def __init__(
        self,
        outputs_folder: Path,
        max_epochs: int = 30,
        patience: int = 0,
        epochs_interval: int = 1,
        num_slides_scatter: int = 10,
        num_slides_heatmap: int = 20,
        save_tile_ids: bool = False,
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
        """

        self.patience = patience
        self.epochs_interval = epochs_interval
        self.max_epochs = max_epochs
        self.num_slides_scatter = num_slides_scatter
        self.num_slides_heatmap = num_slides_heatmap
        self.save_tile_ids = save_tile_ids

        self.outputs_folder = outputs_folder / "loss_values_callback"
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
    def rank_folder(self) -> Path:
        return self.outputs_folder / "loss_ranks"

    @property
    def anomalies_folder(self) -> Path:
        return self.outputs_folder / "loss_anomalies"

    def create_outputs_folders(self) -> None:
        folders = [
            self.cache_folder,
            self.scatter_folder,
            self.heatmap_folder,
            self.rank_folder,
            self.anomalies_folder,
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def reset_loss_cache(self) -> LossCacheDictType:
        keys = [ResultsKey.LOSS, ResultsKey.SLIDE_ID]
        if self.save_tile_ids:
            keys.append(ResultsKey.TILE_ID)
        return {key: [] for key in keys}

    def get_filename(self, root_folder: Path, filename: str, epoch: int, order: Optional[str] = None) -> Path:
        zero_filled_epoch = str(epoch).zfill(len(str(self.max_epochs)))
        filename = filename.format(zero_filled_epoch, order) if order else filename.format(zero_filled_epoch)
        return root_folder / filename

    def read_loss_cache(self, epoch: int, idx_col: Optional[ResultsKey] = None) -> pd.DataFrame:
        return pd.read_csv(self.get_filename(self.cache_folder, LOSS_VALUES_FILENAME, epoch),
                           usecols=[ResultsKey.SLIDE_ID, ResultsKey.LOSS], index_col=idx_col)

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

    def save_loss_cache(self, current_epoch: int) -> None:
        """Saves the loss cache to a csv file"""
        loss_cache_df = pd.DataFrame(self.loss_cache)
        # Some slides may be appear multiple times in the loss cache in DDP mode. The Distributed Sampler duplicate
        # some slides to even out the number of samples per device, so we only keep the first occurrence.
        loss_cache_df.drop_duplicates(subset=ResultsKey.SLIDE_ID, inplace=True, keep="first")
        loss_cache_df = loss_cache_df.sort_values(by=ResultsKey.LOSS, ascending=False)
        filename = self.get_filename(self.cache_folder, LOSS_VALUES_FILENAME, current_epoch)
        loss_cache_df.to_csv(filename, index=False)

    def _select_values_for_epoch(
        self, key: ResultsKey, epoch: int, high: Optional[bool] = None, num_values: Optional[int] = None
    ) -> np.ndarray:
        loss_cache = self.read_loss_cache(epoch)
        values = loss_cache[key].values
        if high is not None:
            assert num_values is not None, "num_values must be specified if high is specified"
            if high:
                return values[:num_values]
            elif not high:
                return values[-num_values:]
        return values

    def select_slides_for_epoch(
        self, epoch: int, high: Optional[bool] = None, num_slides: Optional[int] = None
    ) -> np.ndarray:
        """Selects slides in ascending/descending order of loss values at a given epoch

        :param epoch: The epoch to select the slides from.
        :param high: If True, selects the slides with the highest loss values, else selects the slides with the lowest
            loss values. If None, selects all slides.
        :param num_slides: The number of slides to select. If None, selects all slides.
        """
        return self._select_values_for_epoch(ResultsKey.SLIDE_ID, epoch, high, num_slides)

    def select_slides_losses_for_epoch(
        self, epoch: int, high: Optional[bool] = None, num_slides: Optional[int] = None
    ) -> np.ndarray:
        """Selects loss values of slides in ascending/descending order of loss at a given epoch

        :param epoch: The epoch to select the slides from.
        :param high: If True, selects the slides with the highest loss values, else selects the slides with the lowest
            loss values. If None, selects all slides.
        :param num_slides: The number of slides to select. If None, selects all slides.
        """
        return self._select_values_for_epoch(ResultsKey.LOSS, epoch, high, num_slides)

    def select_all_losses_for_selected_slides(self, slides: np.ndarray) -> LossDictType:
        """Selects the loss values for a given set of slides"""

        slides_loss_values: LossDictType = {slide_id: [] for slide_id in slides}
        for epoch in self.epochs_range:
            loss_cache = self.read_loss_cache(epoch, idx_col=ResultsKey.SLIDE_ID)
            slides_loss_values = {
                slide_id: slides_loss_values[slide_id] + [loss_cache.loc[slide_id, ResultsKey.LOSS]]
                for slide_id in slides
            }
        return slides_loss_values

    def select_slides_and_losses_across_epochs(self, high: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Selects the slides with the highest/lowest loss values across epochs

        :param high: If True, selects the slides with the highest loss values, else selects the slides with the lowest
        loss values.
        """
        slides = []
        slides_loss = []
        for epoch in self.epochs_range:
            epoch_slides = self.select_slides_for_epoch(epoch, high, self.num_slides_scatter)
            epoch_slides_loss = self.select_slides_losses_for_epoch(epoch, high, self.num_slides_scatter)
            slides.append(epoch_slides)
            slides_loss.append(epoch_slides_loss)

        return np.array(slides).T, np.array(slides_loss).T

    def save_slide_ids(self, slide_ids: List[str], filename: str) -> None:
        """Dumps the slides ids in a txt file."""
        if slide_ids:
            with open(self.anomalies_folder / filename, "w") as f:
                for slide_id in slide_ids:
                    f.write(f"{slide_id}\n")

    def sanity_check_loss_values(self, loss_values: LossDictType) -> None:
        """Checks if there are any NaNs or any other potential annomalies in the loss values.

        :param loss_values: The loss values for all slides.
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
        self.save_slide_ids(self.nan_slides, NAN_SLIDES_FILENAME)
        self.save_slide_ids(self.anomaly_slides, EXCEPTION_SLIDES_FILENAME)

    def save_loss_ranks(self, slides_loss_values: LossDictType) -> None:
        """Saves the loss ranks for each slide across all epochs and their respective statistics in csv files."""

        loss_df = pd.DataFrame(slides_loss_values).T
        loss_df.index.names = [ResultsKey.SLIDE_ID.value]
        loss_df.to_csv(self.cache_folder / ALL_EPOCHS_FILENAME)

        loss_stats = loss_df.T.describe().T.sort_values(by="mean", ascending=False)
        loss_stats.to_csv(self.rank_folder / LOSS_STATS_FILENAME)

        loss_ranks = loss_df.rank(ascending=False)
        loss_ranks.to_csv(self.rank_folder / LOSS_RANKS_FILENAME)

        loss_ranks_stats = loss_ranks.T.describe().T.sort_values("mean", ascending=True)
        loss_ranks_stats.to_csv(self.rank_folder / LOSS_RANKS_STATS_FILENAME)

    def plot_slides_loss_scatter(
        self,
        slides: np.ndarray,
        slides_loss: np.ndarray,
        high: bool = True,
        figsize: Tuple[float, float] = (30, 30),
    ) -> None:
        """Plots the slides with the highest/lowest loss values across epochs in a scatter plot

        :param slides: The slides ids.
        :param slides_loss: The loss values for each slide.
        :param figsize: The figure size, defaults to (20, 20)
        :param high: If True, plots the slides with the highest loss values, else plots the slides with the lowest loss.
        """
        label = TOP if high else BOTTOM
        plt.figure(figsize=figsize)
        for i in range(self.num_slides_scatter - 1, -1, -1):
            plt.scatter(self.epochs_range, slides[i], label=f"{label}_{i+1}")
            for loss, epoch, slide in zip(slides_loss[i], self.epochs_range, slides[i]):
                plt.annotate(f"{loss:.3f}", (epoch, slide))
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        order = HIGHEST if high else LOWEST
        plt.title(f"Slides with {order} loss values per epoch.")
        plt.xticks(self.epochs_range)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, linestyle="--")
        plt.savefig(self.scatter_folder / SCATTER_PLOT_FILENAME.format(order), bbox_inches="tight")

    def plot_loss_heatmap_for_slides_of_epoch(
        self, slides_loss_values: LossDictType, epoch: int, high: bool, figsize: Tuple[float, float] = (15, 15)
    ) -> None:
        """Plots the loss values for each slide across all epochs in a heatmap.

        :param slides_loss_values: The loss values for each slide across all epochs.
        :param epoch: The epoch used to select the slides.
        :param high: If True, plots the slides with the highest loss values, else plots the slides with the lowest loss.
        :param figsize: The figure size, defaults to (15, 15)
        """
        order = HIGHEST if high else LOWEST
        loss_values = np.array(list(slides_loss_values.values()))
        slides = list(slides_loss_values.keys())
        plt.figure(figsize=figsize)
        _ = sns.heatmap(loss_values, linewidth=0.5, annot=True, yticklabels=slides)
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        plt.title(f"Loss values evolution for {order} slides of epoch {epoch}")
        plt.savefig(self.get_filename(self.heatmap_folder, HEATMAP_PLOT_FILENAME, epoch, order), bbox_inches="tight")

    @torch.no_grad()
    def on_train_batch_start(  # type: ignore
        self, trainer: Trainer, pl_module: BaseDeepMILModule, batch: Dict, batch_idx: int, unused: int = 0,
    ) -> None:
        """Caches loss values per slide at each training step in a local variable self.loss_cache."""
        if self.should_cache_loss_values(trainer.current_epoch):
            bag_logits, bag_labels, _ = pl_module.compute_bag_labels_logits_and_attn_maps(batch)
            if pl_module.n_classes > 1:
                loss = pl_module.loss_fn_no_reduction(bag_logits, bag_labels.long())
            else:
                loss = pl_module.loss_fn_no_reduction(bag_logits.squeeze(1), bag_labels.float())
            self.loss_cache[ResultsKey.LOSS].extend(loss.tolist())
            self.loss_cache[ResultsKey.SLIDE_ID].extend([slides[0] for slides in batch[ResultsKey.SLIDE_ID]])
            if self.save_tile_ids:
                self.loss_cache[ResultsKey.TILE_ID].extend(
                    [TILES_JOIN_TOKEN.join(tiles) for tiles in batch[ResultsKey.TILE_ID]]
                )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        """Gathers loss values per slide from all processes at the end of each epoch and saves them to a csv file."""
        if self.should_cache_loss_values(trainer.current_epoch):
            self.gather_loss_cache(rank=pl_module.global_rank)
            if pl_module.global_rank == 0:
                self.save_loss_cache(trainer.current_epoch)
        self.loss_cache = self.reset_loss_cache()  # reset loss cache for all processes

    def on_train_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        """Hook called at the end of training. Plot the loss heatmap and scratter plots after ranking the slides by loss
        values."""

        if pl_module.global_rank == 0:
            try:
                all_slides = self.select_slides_for_epoch(epoch=0)
                all_loss_values_per_slides = self.select_all_losses_for_selected_slides(all_slides)

                self.sanity_check_loss_values(all_loss_values_per_slides)
                self.save_loss_ranks(all_loss_values_per_slides)

                top_slides, top_slides_loss = self.select_slides_and_losses_across_epochs(high=True)
                self.plot_slides_loss_scatter(top_slides, top_slides_loss, high=True)

                bottom_slides, bottom_slides_loss = self.select_slides_and_losses_across_epochs(high=False)
                self.plot_slides_loss_scatter(bottom_slides, bottom_slides_loss, high=False)

                for epoch in self.epochs_range:
                    epoch_slides = self.select_slides_for_epoch(epoch)

                    top_slides = epoch_slides[:self.num_slides_heatmap]
                    top_slides_loss_values = self.select_all_losses_for_selected_slides(top_slides)
                    self.plot_loss_heatmap_for_slides_of_epoch(top_slides_loss_values, epoch, high=True)

                    bottom_slides = epoch_slides[-self.num_slides_heatmap:]
                    bottom_slides_loss_values = self.select_all_losses_for_selected_slides(bottom_slides)
                    self.plot_loss_heatmap_for_slides_of_epoch(bottom_slides_loss_values, epoch, high=False)

            except Exception as e:
                # If something goes wrong, we don't want to crash the training. We just log the error and carry on
                # validation.
                logging.warning(f"Error while detecting loss values outliers: {e}")
