#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
import torch
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


class LossValuesAnalysisCallback(Callback):
    def __init__(
        self,
        outputs_folder: Path,
        patience: int = 0,
        save_every_n_epoch: int = 1,
        max_epochs: int = 30,
        num_slides_scatter: int = 10,
        num_slides_heatmap: int = 20,
    ) -> None:

        self.patience = patience
        self.save_every_n_epoch = save_every_n_epoch
        self.max_epochs = max_epochs
        self.num_slides_scatter = num_slides_scatter
        self.num_slides_heatmap = num_slides_heatmap

        self.outputs_folder = outputs_folder / "loss_values_callback"
        self.create_outputs_folders()

        self.loss_cache = self.reset_loss_cache()
        self.epochs_range = list(range(self.patience, self.max_epochs, self.save_every_n_epoch))

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
        return (trainer.current_epoch - self.patience) % self.save_every_n_epoch == 0

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

    def check_for_nans(self, loss_values: LossDictType) -> None:
        for slide_id, loss in loss_values.items():
            if np.isnan(loss).any():
                raise ValueError(f"NaNs found in loss values for slide {slide_id}.")

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
        plt.savefig(self.scatter_folder / f"slides_with_{order}_loss_values.png", bbox_inches='tight')

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

    def plot_loss_heatmap_for_slides_of_epoch(
        self, slides_loss_values: LossDictType, epoch: int, high: bool, figsize: Tuple[int, int] = (15, 15)
    ) -> None:
        loss_values = np.array(list(slides_loss_values.values()))
        slides = list(slides_loss_values.keys())
        try:
            self.check_for_nans(slides_loss_values)
        except ValueError as e:
            logging.warning(f"Skipping loss heatmap because {e}")
            return
        plt.figure(figsize=figsize)
        _ = sns.heatmap(loss_values, linewidth=0.5, annot=True, yticklabels=slides)
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        order = HIGHEST if high else LOWEST
        plt.title(f"Loss values evolution for {order} slides of epoch {epoch}")
        plt.savefig(self.evolution_folder / f"{order}_slides_of_epoch_{epoch}.png", bbox_inches='tight')

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
