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
LOSS_RANKS_STATS_FILENAME = "loss_ranks_stats.csv"

SCATTER_PLOT_FILENAME = "slides_with_{}_loss_values.png"
HEATMAP_PLOT_FILENAME = "epoch_{}_{}_slides.png"

NAN_SLIDES_FILENAME = "nan_slides.txt"
EXCEPTION_SLIDES_FILENAME = "expection_slides.txt"

X_LABEL, Y_LABEL = "Epoch", "Slide ids"
TOP, BOTTOM = "top", "bottom"
HIGHEST, LOWEST = "highest", "lowest"

LossDictType = Dict[ResultsKey, List]


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
        "caching loss values per epoch immediately. Use lv_patience > 0 to wait for a few epochs before starting "
        "the analysis.",
    )
    loss_analysis_epochs_interval: int = param.Integer(
        1,
        bounds=(1, None),
        doc="Epochs interval to save loss values. Default: 1, It will save loss values every epoch. Use "
        "lv_every_n_epochs > 1 to save loss values every n epochs.",
    )
    num_slides_scatter: int = param.Integer(
        10,
        bounds=(1, None),
        doc="Number of slides to plot in the scatter plot. Default: 10, It will plot a scatter of the 10 slides "
        "with highest/lowest loss values across epochs.",
    )
    num_slides_heatmap: int = param.Integer(
        20,
        bounds=(1, None),
        doc="Number of slides to plot in the heatmap plot. Default: 20, It will plot the loss values for the 10 slides "
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
        :param save_every_n_epoch: Epochs interval to save loss values, defaults to 1.
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
        self.exception_slides: List[str] = []

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
    def exception_folder(self) -> Path:
        return self.outputs_folder / "loss_exceptions"

    def create_outputs_folders(self) -> None:
        folders = [
            self.cache_folder,
            self.scatter_folder,
            self.heatmap_folder,
            self.rank_folder,
            self.exception_folder,
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def reset_loss_cache(self) -> LossDictType:
        keys = [ResultsKey.LOSS, ResultsKey.SLIDE_ID]
        if self.save_tile_ids:
            keys.append(ResultsKey.TILE_ID)
        return {key: [] for key in keys}

    def is_time_to_cache_loss_values(self, current_epoch: int) -> bool:
        current_epoch = current_epoch + 1
        first_time = (current_epoch - self.patience) == 1
        return first_time or (
            current_epoch > self.patience and (current_epoch - self.patience) % self.epochs_interval == 0
        )

    def save_loss_cache(self, current_epoch: int) -> None:
        """Saves the loss cache to a csv file"""
        loss_cache_df = pd.DataFrame(self.loss_cache)
        loss_cache_df = loss_cache_df.sort_values(by=ResultsKey.LOSS, ascending=False)
        loss_cache_df.to_csv(self.cache_folder / LOSS_VALUES_FILENAME.format(current_epoch), index=False)

    def merge_loss_caches(self, loss_caches: List[LossDictType]) -> LossDictType:
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

    def select_loss_slides_across_epochs(self, high: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Selects the slides with the highest/lowest loss values across epochs

        :param high: If True, selects the slides with the highest loss values, else selects the slides with the lowest
        loss values.
        """
        slides = []
        slides_loss = []
        for epoch in self.epochs_range:
            loss_cache = pd.read_csv(self.cache_folder / LOSS_VALUES_FILENAME.format(epoch))

            if high:
                slides.append(loss_cache[ResultsKey.SLIDE_ID][: self.num_slides_scatter])
                slides_loss.append(loss_cache[ResultsKey.LOSS][: self.num_slides_scatter])
            else:
                slides.append(loss_cache[ResultsKey.SLIDE_ID][-self.num_slides_scatter:])
                slides_loss.append(loss_cache[ResultsKey.LOSS][-self.num_slides_scatter:])

        return np.array(slides).T, np.array(slides_loss).T

    def plot_slides_loss_scatter(
        self, slides: np.ndarray, slides_loss: np.ndarray, figure_size: Tuple[int, int] = (20, 20), high: bool = True,
    ) -> None:
        """Plots the slides with the highest/lowest loss values across epochs in a scatter plot

        :param slides: The slides ids.
        :param slides_loss: The loss values for each slide.
        :param figure_size: The figure size, defaults to (20, 20)
        :param high: If True, plots the slides with the highest loss values, else plots the slides with the lowest loss.
        """
        label = TOP if high else BOTTOM
        plt.figure(figsize=figure_size)
        for i in range(self.num_slides_scatter - 1, -1, -1):
            plt.scatter(self.epochs_range, slides[i], label=f"{label}_{i+1}")
            coordinates = [
                (loss, epoch, slide) for loss, epoch, slide in zip(slides_loss[i], self.epochs_range, slides[i])
            ]
            for loss, epoch, slide in coordinates:
                plt.annotate(f"{loss:.3f}", (epoch, slide))
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        order = HIGHEST if high else LOWEST
        plt.title(f"Slides with {order} loss values per epoch.")
        plt.xticks(self.epochs_range)
        plt.legend()
        plt.grid(True, linestyle="--")
        plt.savefig(self.scatter_folder / SCATTER_PLOT_FILENAME.format(order), bbox_inches="tight")

    def select_loss_for_slides_of_epoch(self, epoch: int, high: Optional[bool] = None) -> LossDictType:
        """Selects the slides with the highest/lowest loss values for a given epoch and returns the loss values for each
        slide across all epochs.

        :param epoch: The epoch to select the slides from.
        :param high: If True, selects the slides with the highest loss values, else selects the slides with the lowest
            loss values.
        :return: A dictionary containing the loss values for each slide across all epochs.
        """
        loss_cache = pd.read_csv(self.cache_folder / LOSS_VALUES_FILENAME.format(epoch))

        if high:
            slides = loss_cache[ResultsKey.SLIDE_ID][: self.num_slides_heatmap]
        elif not high:
            slides = loss_cache[ResultsKey.SLIDE_ID][-self.num_slides_heatmap:]
        else:
            slides = loss_cache[ResultsKey.SLIDE_ID]

        slides_loss_values: LossDictType = {slide_id: [] for slide_id in slides}
        for epoch in self.epochs_range:
            loss_cache = pd.read_csv(self.cache_folder / LOSS_VALUES_FILENAME.format(epoch))
            loss_cache.set_index(ResultsKey.SLIDE_ID, inplace=True)
            slides_loss_values = {
                slide_id: slides_loss_values[slide_id] + [loss_cache.loc[slide_id, ResultsKey.LOSS]]
                for slide_id in slides
            }
        return slides_loss_values

    def plot_loss_heatmap_for_slides_of_epoch(
        self, slides_loss_values: LossDictType, epoch: int, high: bool, figsize: Tuple[int, int] = (15, 15)
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
        # Loss heatmap plot can go wrong. We need to catch the error and log it otherwise it will interupt validation.
        try:
            plt.figure(figsize=figsize)
            _ = sns.heatmap(loss_values, linewidth=0.5, annot=True, yticklabels=slides)
            plt.xlabel(X_LABEL)
            plt.ylabel(Y_LABEL)
            plt.title(f"Loss values evolution for {order} slides of epoch {epoch}")
            plt.savefig(self.heatmap_folder / HEATMAP_PLOT_FILENAME.format(epoch, order), bbox_inches="tight")
        except Exception as e:
            logging.warning(f"Skipping loss heatmap because of Exception {e}")

    def save_slide_ids(self, slide_ids: List[str], filename: str) -> None:
        """Dumps the slides ids in a txt file."""
        if slide_ids:
            with open(self.exception_folder / filename, "w") as f:
                for slide_id in slide_ids:
                    f.write(f"{slide_id}\n")

    def sanity_check_loss_values(self, loss_values: LossDictType) -> None:
        """Checks if there are any NaNs or any other potential issues in the loss values for a given epoch and slide.

        :param loss_values: The loss values for all slides.
        :param order: If "highest", checks for NaNs in the highest loss values, else checks for NaNs in the lowest loss
        :param epoch: The epoch to check for NaNs.
        """
        for slide_id, loss in loss_values.items():
            try:
                if np.isnan(loss).any():
                    logging.warning(f"NaNs found in loss values for slide {slide_id}.")
                    self.nan_slides.append(slide_id)
            except Exception as e:
                logging.warning(f"Error while checking for NaNs in loss values for slide {slide_id} with error {e}.")
                print("Loos values:", loss)
                self.exception_slides.append(slide_id)
        self.save_slide_ids(self.nan_slides, NAN_SLIDES_FILENAME)
        self.save_slide_ids(self.exception_slides, EXCEPTION_SLIDES_FILENAME)

    def save_loss_ranks(self) -> None:
        """Saves the loss ranks for each slide across all epochs and their respective statistics in csv files."""
        slides_loss_values = self.select_loss_for_slides_of_epoch(epoch=0, high=None)
        self.sanity_check_loss_values(slides_loss_values)
        loss_df = pd.DataFrame(slides_loss_values).T
        loss_df.index.names = [ResultsKey.SLIDE_ID.value]
        loss_df.to_csv(self.cache_folder / ALL_EPOCHS_FILENAME)
        loss_ranks = loss_df.rank(ascending=False)
        loss_ranks.to_csv(self.rank_folder / LOSS_RANKS_FILENAME)
        loss_ranks_stats = loss_ranks.T.describe().T.sort_values("mean", ascending=True)
        loss_ranks_stats.to_csv(self.rank_folder / LOSS_RANKS_STATS_FILENAME)

    @torch.no_grad()
    def on_train_batch_start(  # type: ignore
        self, trainer: Trainer, pl_module: BaseDeepMILModule, batch: Dict, batch_idx: int, unused: int = 0,
    ) -> None:
        """Caches loss values per slide at each training step in a local variable self.loss_cache."""
        if self.is_time_to_cache_loss_values(trainer.current_epoch):
            bag_logits, bag_labels, _ = pl_module.compute_bag_labels_logits_and_attn_maps(batch)
            if pl_module.n_classes > 1:
                loss = pl_module.loss_fn_no_reduction(bag_logits, bag_labels.long())
            else:
                loss = pl_module.loss_fn_no_reduction(bag_logits.squeeze(1), bag_labels.float())
            self.loss_cache[ResultsKey.LOSS].extend(loss.tolist())
            self.loss_cache[ResultsKey.SLIDE_ID].extend(np.array([slides[0] for slides in batch[ResultsKey.SLIDE_ID]]))
            if self.save_tile_ids:
                self.loss_cache[ResultsKey.TILE_ID].extend(
                    [TILES_JOIN_TOKEN.join(tiles) for tiles in batch[ResultsKey.TILE_ID]]
                )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        """Gathers loss values per slide from all processes at the end of each epoch and saves them to a csv file."""
        if self.is_time_to_cache_loss_values(trainer.current_epoch):
            self.gather_loss_cache(rank=pl_module.global_rank)
            if pl_module.global_rank == 0:
                self.save_loss_cache(trainer.current_epoch)
        self.loss_cache = self.reset_loss_cache()

    def on_train_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        """Hook called at the end of training. We use it to plot the loss heatmap and scrater plot."""

        if pl_module.global_rank == 0:
            self.save_loss_ranks()

            slides, slides_loss = self.select_loss_slides_across_epochs(high=True)
            self.plot_slides_loss_scatter(slides, slides_loss, high=True)

            slides, slides_loss = self.select_loss_slides_across_epochs(high=False)
            self.plot_slides_loss_scatter(slides, slides_loss, high=False)

            for epoch in self.epochs_range:
                slides_loss_values = self.select_loss_for_slides_of_epoch(epoch, high=True)
                self.plot_loss_heatmap_for_slides_of_epoch(slides_loss_values, epoch, high=True)

                slides_loss_values = self.select_loss_for_slides_of_epoch(epoch, high=False)
                self.plot_loss_heatmap_for_slides_of_epoch(slides_loss_values, epoch, high=False)
