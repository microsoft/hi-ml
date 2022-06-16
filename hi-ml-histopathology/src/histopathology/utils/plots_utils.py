#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple, Dict
from torchmetrics.classification.confusion_matrix import ConfusionMatrix

from histopathology.datasets.base_dataset import SlidesDataset
from histopathology.utils.viz_utils import (
    plot_attention_tiles,
    plot_heatmap_overlay,
    plot_normalized_confusion_matrix,
    plot_scores_hist,
    plot_slide,
)
from histopathology.utils.naming import ModelKey, PlotOptionsKey, ResultsKey, SlideKey
from histopathology.utils.tiles_selection_utils import SlideNode, TilesSelector
from histopathology.utils.viz_utils import load_image_dict, save_figure


ResultsType = Dict[ResultsKey, List[Any]]


def validate_plot_options(plot_options: Set[PlotOptionsKey]) -> Set[PlotOptionsKey]:
    for opt in plot_options:
        if opt not in PlotOptionsKey.__members__.values():
            raise ValueError(
                "The selected plot option is not a valid option, choose among the options available in "
                "histopathology.utils.naming.PlotOptionsKey"
            )
    return plot_options


def validate_class_names_for_plot_options(
    class_names: Optional[Sequence[str]], plot_options: Set[PlotOptionsKey]
) -> Optional[Sequence[str]]:
    if PlotOptionsKey.CONFUSION_MATRIX in plot_options and not class_names:
        raise ValueError(
            "No class_names were provided while activating confusion matrix plotting. You need to specify the class "
            "names to use the `PlotOptionsKey.CONFUSION_MATRIX`"
        )
    return class_names


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


def save_top_and_bottom_tiles(
    case: str, slide_node: SlideNode, figures_dir: Path, num_columns: int = 4, figsize: Tuple[int, int] = (10, 10)
) -> None:
    """Plots and saves the top and bottom attention tiles of a given slide_node

    :param case: The report case (e.g., TP, FN, ...)
    :param slide_node: the slide_node for which we plot top and bottom tiles.
    :param figures_dir: The path to the directory where to save the attention tiles figure.
    """
    top_tiles_fig = plot_attention_tiles(
        slide_node=slide_node, case=case, top=True, num_columns=num_columns, figsize=figsize
    )
    save_figure(fig=top_tiles_fig, figpath=figures_dir / f"{slide_node.slide_id}_top.png")

    bottom_tiles_fig = plot_attention_tiles(
        slide_node=slide_node, case=case, top=False, num_columns=num_columns, figsize=figsize
    )
    save_figure(fig=bottom_tiles_fig, figpath=figures_dir / f"{slide_node.slide_id}_bottom.png")


def save_slide_thumbnail_and_heatmap(
    case: str,
    slide_node: SlideNode,
    figures_dir: Path,
    results: ResultsType,
    slides_dataset: SlidesDataset,
    tile_size: int = 224,
    level: int = 1,
) -> None:
    """Plots and saves a slide thumbnail and attention heatmap

    :param case: The report case (e.g., TP, FN, ...)
    :param slide_node: The slide node that encapsulates the slide metadata.
    :param figures_dir: The path to the directory where to save the plots.
    :param results: Dict containing ResultsKey keys (e.g. slide id) and values as lists of output slides.
    :param tile_size: Size of each tile. Default 224.
    :param level: Magnification at which tiles are available (e.g. PANDA levels are 0 for original,
    1 for 4x downsampled, 2 for 16x downsampled). Default 1.
    :param slides_dataset: The slides dataset from where to pick the slide.
    """
    slide_index = slides_dataset.dataset_df.index.get_loc(slide_node.slide_id)
    assert isinstance(slide_index, int), f"Got non-unique slide ID: {slide_node.slide_id}"
    slide_dict = slides_dataset[slide_index]
    slide_dict = load_image_dict(slide_dict, level=level, margin=0)
    slide_image = slide_dict[SlideKey.IMAGE]
    location_bbox = slide_dict[SlideKey.LOCATION]

    fig = plot_slide(case=case, slide_node=slide_node, slide_image=slide_image, scale=1.0)
    save_figure(fig=fig, figpath=figures_dir / f"{slide_node.slide_id}_thumbnail.png")

    fig = plot_heatmap_overlay(
        case=case,
        slide_node=slide_node,
        slide_image=slide_image,
        results=results,
        location_bbox=location_bbox,
        tile_size=tile_size,
        level=level,
    )
    save_figure(fig=fig, figpath=figures_dir / f"{slide_node.slide_id}_heatmap.png")


def make_figure_dirs(subfolder: str, parent_dir: Path) -> Path:
    """Create the figure directory"""
    figures_dir = parent_dir / subfolder
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


class DeepMILPlotsHandler:
    def __init__(
        self,
        plot_options: Set[PlotOptionsKey],
        level: int = 1,
        tile_size: int = 224,
        num_columns: int = 4,
        figsize: Tuple[int, int] = (10, 10),
        conf_matrix: Optional[ConfusionMatrix] = None,
        class_names: Optional[Sequence[str]] = None,
        slides_dataset: Optional[SlidesDataset] = None,
    ) -> None:
        """
        :param num_columns: Number of columns to create the subfigures grid, defaults to 4
        :param figsize: The figure size of tiles attention plots, defaults to (10, 10)
        """
        self.plot_options = validate_plot_options(plot_options)
        self.class_names = validate_class_names_for_plot_options(class_names, plot_options)
        self.level = level
        self.tile_size = tile_size
        self.num_columns = num_columns
        self.figsize = figsize
        self.conf_matrix: ConfusionMatrix = conf_matrix
        self.slides_dataset = slides_dataset

    def save_slide_node_figures(
        self, case: str, slide_node: SlideNode, outputs_dir: Path, results: ResultsType, log: bool
    ) -> None:

        case_dir = make_figure_dirs(subfolder=case, parent_dir=outputs_dir)

        if PlotOptionsKey.TOP_BOTTOM_TILES in self.plot_options:
            save_top_and_bottom_tiles(
                case=case,
                slide_node=slide_node,
                figures_dir=case_dir,
                num_columns=self.num_columns,
                figsize=self.figsize,
            )
        if PlotOptionsKey.SLIDE_THUMBNAIL_HEATMAP in self.plot_options:
            assert self.slides_dataset
            save_slide_thumbnail_and_heatmap(
                case=case,
                slide_node=slide_node,
                figures_dir=case_dir,
                results=results,
                slides_dataset=self.slides_dataset,
                tile_size=self.tile_size,
                level=self.level,
            )

    def log_slide_plot_options(self) -> None:
        if PlotOptionsKey.TOP_BOTTOM_TILES in self.plot_options:
            logging.info("Plotting top and bottom tiles ...")

        if PlotOptionsKey.SLIDE_THUMBNAIL_HEATMAP in self.plot_options:
            logging.info("Plotting slide thumbnails and heatmaps ...")

    def save_all_plot_options(
        self, outputs_dir: Path, tiles_selector: Optional[TilesSelector], results: ResultsType, stage: ModelKey
    ) -> None:

        logging.info(f"Start plotting all figure outputs in {outputs_dir}")
        figures_dir = make_figure_dirs(subfolder="fig", parent_dir=outputs_dir)

        if PlotOptionsKey.HISTOGRAM in self.plot_options:
            logging.info("Plotting histogram scores ...")
            save_scores_histogram(results=results, figures_dir=figures_dir)

        if PlotOptionsKey.CONFUSION_MATRIX in self.plot_options:
            # TODO: Re-enable plotting confusion matrix without relying on metrics to avoid DDP deadlocks
            # will be adressed in a seperate PR
            logging.info("Plotting confusion matrix ...")
            assert self.class_names
            save_confusion_matrix(self.conf_matrix, class_names=self.class_names, figures_dir=figures_dir, stage=stage)

        if tiles_selector:
            self.log_slide_plot_options()
            for class_id in range(tiles_selector.n_classes):

                for slide_node in tiles_selector.top_slides_heaps[class_id]:
                    case = "TN" if class_id == 0 else f"TP_{class_id}"
                    self.save_slide_node_figures(case, slide_node, outputs_dir, results)

                for slide_node in tiles_selector.bottom_slides_heaps[class_id]:
                    case = "FP" if class_id == 0 else f"FN_{class_id}"
                    self.save_slide_node_figures(case, slide_node, outputs_dir, results)
