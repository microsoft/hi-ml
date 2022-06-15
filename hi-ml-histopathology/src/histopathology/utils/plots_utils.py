#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from importlib_metadata import Mapping
import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from pathlib import Path
from typing import List, Sequence, Tuple
from prometheus_client import Metric
from torchmetrics.classification.confusion_matrix import ConfusionMatrix

from histopathology.datasets.base_dataset import SlidesDataset
from histopathology.utils.metrics_utils import (
    plot_attention_tiles,
    plot_heatmap_overlay,
    plot_normalized_confusion_matrix,
    plot_scores_hist,
    plot_slide,
)

from histopathology.utils.naming import ModelKey, PlotOptionsKey, SlideKey, MetricsKey
from histopathology.utils.output_utils import ResultsType
from histopathology.utils.tiles_selection_utils import SlideNode, TilesSelector
from histopathology.utils.viz_utils import load_image_dict, save_figure


def save_scores_histogram(results: ResultsType, figures_dir: Path) -> None:
    logging.info("Plotting histogram ...")
    fig = plot_scores_hist(results)
    save_figure(fig=fig, figpath=figures_dir / "hist_scores.png")


def save_confusion_matrix(
    conf_matrix_metric: ConfusionMatrix, class_names: Sequence[str], figures_dir: Path, stage: ModelKey
) -> None:
    logging.info("Computing and saving confusion matrix...")
    cf_matrix = conf_matrix_metric.compute().cpu().numpy()
    #  We can't log tensors in the normal way - just print it to console
    logging.info(f"{stage}/confusion matrix:")
    logging.info(cf_matrix)
    #  Save the normalized confusion matrix as a figure in outputs
    cf_matrix_n = cf_matrix / cf_matrix.sum(axis=1, keepdims=True)
    fig = plot_normalized_confusion_matrix(cm=cf_matrix_n, class_names=(class_names))
    save_figure(fig=fig, figpath=figures_dir / "normalized_confusion_matrix.png")


def save_top_and_bottom_tiles(slide_node: SlideNode, case: str, figures_dir: Path) -> None:
    """Plots and saves the top and bottom attention tiles of a given slide_node

    :param slide_node: the slide_node for which we plot top and bottom tiles.
    :param case: The report case (e.g., TP, FN, ...)
    :param figures_dir: The path to the directory where to save the attention tiles figure.
    """
    top_tiles_fig = plot_attention_tiles(top=True, slide_node=slide_node, case=case)
    save_figure(fig=top_tiles_fig, figpath=figures_dir / f"{slide_node.slide_id}_top.png")

    bottom_tiles_fig = plot_attention_tiles(top=False, slide_node=slide_node, case=case)
    save_figure(fig=bottom_tiles_fig, figpath=figures_dir / f"{slide_node.slide_id}_bottom.png")


def save_slide_thumbnail_and_heatmap(
    results: ResultsType,
    slide_node: SlideNode,
    tile_size: int,
    level: int,
    slides_dataset: SlidesDataset,
    figures_dir: Path,
) -> None:
    slide_index = slides_dataset.dataset_df.index.get_loc(slide_node.slide_id)
    assert isinstance(slide_index, int), f"Got non-unique slide ID: {slide_node.slide_id}"
    slide_dict = slides_dataset[slide_index]
    slide_dict = load_image_dict(slide_dict, level=level, margin=0)
    slide_image = slide_dict[SlideKey.IMAGE]
    location_bbox = slide_dict[SlideKey.LOCATION]

    fig = plot_slide(slide_image=slide_image, scale=1.0)
    save_figure(fig=fig, figpath=figures_dir / f"{slide_node.slide_id}_thumbnail.png")

    fig = plot_heatmap_overlay(
        slide_node=slide_node.slide_id,
        slide_image=slide_image,
        results=results,
        location_bbox=location_bbox,
        tile_size=tile_size,
        level=level,
    )
    save_figure(fig=fig, figpath=figures_dir / f"{slide_node.slide_id}_heatmap.png")


PLOTS_PRINT_STATEMENTS = {
    PlotOptionsKey.TOP_BOTTOM_TILES: "Plotting top and bottom tiles ...",
    PlotOptionsKey.SLIDE_THUMBNAIL_HEATMAP: "Plotting slide thumbnails and heatmaps ...",
    PlotOptionsKey.HISTOGRAM: "Plotting histogram scores ...",
    PlotOptionsKey.CONFUSION_MATRIX: "Plotting confusion matrix ...",
}


class DeepMILPlotsHandler:
    def __init__(
        self,
        plot_options: List[PlotOptionsKey],
        tile_size: int = 224,
        level: int = 1,
        num_columns: int = 4,
        figsize: Tuple[int, int] = (10, 10),
    ) -> None:
        """
        :param tiles_selector: The tiles selector that embeds top and bottom slides and their respective top and bottom
            tiles to be plotted and saved.
        :param num_columns: Number of columns to create the subfigures grid, defaults to 4
        :param figsize: The figure size of tiles attention plots, defaults to (10, 10)
        """
        self.plot_options = plot_options
        self.figsize = figsize
        self.level = level
        self.tile_size = tile_size
        self.num_columns = num_columns

    @staticmethod
    def make_figure_dirs(case: str, figures_dir: Path) -> Path:
        """Create the figure directory"""
        case_dir = figures_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def plot(
        self,
        figures_dir: Path,
        tiles_selector: TilesSelector,
        results: ResultsType,
        slides_dataset: SlidesDataset,
        class_names: Tuple[str, ...],
        metrics_dict: Mapping[MetricsKey, Metric],
    ) -> None:

        if PlotOptionsKey.HISTOGRAM in self.plot_options:
            save_scores_histogram(results, figures_dir)

        if PlotOptionsKey.CONFUSION_MATRIX in self.plot_options:
            # TODO: Re-enable plotting confusion matrix without relying on metrics to avoid DDP deadlocks
            conf_matrix: ConfusionMatrix = metrics_dict[MetricsKey.CONF_MATRIX]  # type: ignore
            save_confusion_matrix(conf_matrix, class_names=class_names, figures_dir=figures_dir)

        for class_id in range(tiles_selector.n_classes):
            for slide_node in tiles_selector.top_slides_heaps[class_id]:
                case = "TN" if class_id == 0 else f"TP_{class_id}"
                case_dir = self.make_figure_dirs(case=case, figures_dir=figures_dir)
                if PlotOptionsKey.TOP_BOTTOM_TILES in self.plot_options:
                    save_top_and_bottom_tiles(slide_node, case, case_dir)
                if PlotOptionsKey.SLIDE_THUMBNAIL_HEATMAP in self.plot_options:
                    save_slide_thumbnail_and_heatmap(
                        results,
                        slide_id=slide_id,
                        tile_size=tile_size,
                        level=level,
                        slides_dataset=slides_dataset,
                        figures_dir=key_dir,
                    )

            for slide_node in tiles_selector.bottom_slides_heaps[class_id]:
                case = "FP" if class_id == 0 else f"FN_{class_id}"
                case_dir = self.make_figure_dirs(case=case, figures_dir=figures_dir)
