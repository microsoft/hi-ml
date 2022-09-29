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
from typing import Dict, List, Tuple
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from health_cpath.models.deepmil import BaseDeepMILModule
from health_cpath.utils.naming import ResultsKey

TILES_JOIN_TOKEN = "$"
LOSS_VALUES_FILENAME = "epoch_{}.csv"
X_LABEL, Y_LABEL = "Epoch", "Slide ids"
TOP, BOTTOM = "top", "bottom"
HIGHEST, LOWEST = "highest", "lowest"

LossDictType = Dict[str, List]


class LossCallbackParams(param.Parameterized):
    """Parameters class to group all attributes for loss values analysis callback"""

    analyse_loss_values: bool = param.Boolean(
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
    ) -> None:
        """

        :param outputs_folder: Path to the folder where the outputs will be saved.
        :param patience: Number of epochs to wait before starting to cache loss values, defaults to 0.
        :param save_every_n_epoch: Epochs interval to save loss values, defaults to 1.
        :param max_epochs: Maximum number of epochs to train, defaults to 30.
        :param num_slides_scatter: Number of slides to plot in the scatter plot, defaults to 10.
        :param num_slides_heatmap: Number of slides to plot in the heatmap, defaults to 20.
        """

        self.patience = patience
        self.epochs_interval = epochs_interval
        self.max_epochs = max_epochs
        self.num_slides_scatter = num_slides_scatter
        self.num_slides_heatmap = num_slides_heatmap

        self.outputs_folder = outputs_folder / "loss_values_callback"
        self.create_outputs_folders()

        self.loss_cache = self.reset_loss_cache()
        self.epochs_range = list(range(self.patience, self.max_epochs, self.epochs_interval))

    @property
    def cache_folder(self) -> Path:
        return self.outputs_folder / "loss_cache"

    @property
    def scatter_folder(self) -> Path:
        return self.outputs_folder / "loss_scatter"

    @property
    def evolution_folder(self) -> Path:
        return self.outputs_folder / "loss_values_evolution"

    def create_outputs_folders(self) -> None:
        folders = [self.cache_folder, self.scatter_folder, self.evolution_folder]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def reset_loss_cache() -> LossDictType:
        return {ResultsKey.LOSS: [], ResultsKey.SLIDE_ID: [], ResultsKey.TILE_ID: []}

    def is_time_to_cache_loss_values(self, trainer: Trainer) -> bool:
        return (trainer.current_epoch - self.patience) % self.epochs_interval == 0

    def dump_loss_cache(self, current_epoch: int) -> None:
        loss_cache_df = pd.DataFrame(self.loss_cache)
        loss_cache_df = loss_cache_df.sort_values(by=ResultsKey.LOSS, ascending=False)
        loss_cache_df.to_csv(self.cache_folder / LOSS_VALUES_FILENAME.format(current_epoch), index=False)

    def merge_loss_caches(self, loss_caches: List[LossDictType]) -> LossDictType:
        loss_cache = self.reset_loss_cache()
        for loss_cache_per_device in loss_caches:
            loss_cache[ResultsKey.LOSS].extend(loss_cache_per_device[ResultsKey.LOSS])
            loss_cache[ResultsKey.SLIDE_ID].extend(loss_cache_per_device[ResultsKey.SLIDE_ID])
            loss_cache[ResultsKey.TILE_ID].extend(loss_cache_per_device[ResultsKey.TILE_ID])
        return loss_cache

    def gather_loss_cache(self, rank: int) -> None:
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:
                loss_caches = [None] * world_size
                torch.distributed.all_gather_object(loss_caches, self.loss_cache)
                if rank == 0:
                    self.loss_cache = self.merge_loss_caches(loss_caches)  # type: ignore

    @torch.no_grad()
    def on_train_batch_start(  # type: ignore
        self, trainer: Trainer, pl_module: BaseDeepMILModule, batch: Dict, batch_idx: int, unused: int = 0,
    ) -> None:
        if self.is_time_to_cache_loss_values(trainer):
            bag_logits, bag_labels, _ = pl_module.compute_bag_labels_logits_and_attn_maps(batch)
            if pl_module.n_classes > 1:
                loss = pl_module.loss_fn_no_reduction(bag_logits, bag_labels.long())
            else:
                loss = pl_module.loss_fn_no_reduction(bag_logits.squeeze(1), bag_labels.float())
            self.loss_cache[ResultsKey.LOSS].extend(loss.tolist())
            self.loss_cache[ResultsKey.SLIDE_ID].extend(np.array([slides[0] for slides in batch[ResultsKey.SLIDE_ID]]))
            self.loss_cache[ResultsKey.TILE_ID].extend(
                [TILES_JOIN_TOKEN.join(tiles) for tiles in batch[ResultsKey.TILE_ID]]
            )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        if self.is_time_to_cache_loss_values(trainer):
            self.gather_loss_cache(rank=pl_module.global_rank)
            if pl_module.global_rank == 0:
                self.dump_loss_cache(trainer.current_epoch)
        self.loss_cache = self.reset_loss_cache()

    def select_loss_slides_across_epochs(self, high: bool = True) -> Tuple[np.ndarray, np.ndarray]:

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
        plt.legend()
        plt.grid()
        plt.savefig(self.scatter_folder / f"slides_with_{order}_loss_values.png", bbox_inches="tight")

    def select_loss_for_slides_of_epoch(self, epoch: int, high: bool) -> LossDictType:
        loss_cache = pd.read_csv(self.cache_folder / LOSS_VALUES_FILENAME.format(epoch))

        if high:
            slides = loss_cache[ResultsKey.SLIDE_ID][: self.num_slides_heatmap]
        else:
            slides = loss_cache[ResultsKey.SLIDE_ID][-self.num_slides_heatmap:]

        slides_loss_values: LossDictType = {slide_id: [] for slide_id in slides}
        for epoch in self.epochs_range:
            loss_cache = pd.read_csv(self.cache_folder / LOSS_VALUES_FILENAME.format(epoch))
            loss_cache.set_index(ResultsKey.SLIDE_ID, inplace=True)
            slides_loss_values = {
                slide_id: slides_loss_values[slide_id] + [loss_cache.loc[slide_id, ResultsKey.LOSS]]
                for slide_id in slides
            }
        return slides_loss_values

    def check_for_nans(self, loss_values: LossDictType, order: str, epoch: int) -> None:
        for slide_id, loss in loss_values.items():
            if np.isnan(loss).any():
                logging.warning(f"NaNs found in loss values for slide {slide_id}.")
                logging.warning(f"Skipping {order} loss heatmap for epoch {epoch}.")

    def plot_loss_heatmap_for_slides_of_epoch(
        self, slides_loss_values: LossDictType, epoch: int, high: bool, figsize: Tuple[int, int] = (15, 15)
    ) -> None:
        order = HIGHEST if high else LOWEST
        self.check_for_nans(slides_loss_values, order, epoch)
        loss_values = np.array(list(slides_loss_values.values()))
        slides = list(slides_loss_values.keys())
        # Loss heatmap plot can go wrong. We need to catch the error and log it otherwise it will interupt validation.
        try:
            plt.figure(figsize=figsize)
            _ = sns.heatmap(loss_values, linewidth=0.5, annot=True, yticklabels=slides)
            plt.xlabel(X_LABEL)
            plt.ylabel(Y_LABEL)
            plt.title(f"Loss values evolution for {order} slides of epoch {epoch}")
            plt.savefig(self.evolution_folder / f"{order}_slides_of_epoch_{epoch}.png", bbox_inches="tight")
        except Exception as e:
            logging.warning(f"Skipping loss heatmap because of Exception {e}")

    def on_train_end(self, trainer: Trainer, pl_module: BaseDeepMILModule) -> None:  # type: ignore
        if pl_module.global_rank == 0:
            slides, slides_loss = self.select_loss_slides_across_epochs(high=True)
            self.plot_slides_loss_scatter(slides, slides_loss, high=True)

            slides, slides_loss = self.select_loss_slides_across_epochs(high=False)
            self.plot_slides_loss_scatter(slides, slides_loss, high=False)

            for epoch in self.epochs_range:
                slides_loss_values = self.select_loss_for_slides_of_epoch(epoch, high=True)
                self.plot_loss_heatmap_for_slides_of_epoch(slides_loss_values, epoch, high=True)

                slides_loss_values = self.select_loss_for_slides_of_epoch(epoch, high=False)
                self.plot_loss_heatmap_for_slides_of_epoch(slides_loss_values, epoch, high=False)
