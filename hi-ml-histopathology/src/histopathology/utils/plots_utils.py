#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Collection, List, Optional, Sequence, Tuple, Dict
from sklearn.metrics import confusion_matrix

from histopathology.datasets.base_dataset import SlidesDataset
from histopathology.utils.viz_utils import (
    plot_attention_tiles,
    plot_heatmap_overlay,
    plot_normalized_confusion_matrix,
    plot_scores_hist,
    plot_slide,
)
from histopathology.utils.naming import ModelKey, PlotOption, ResultsKey, SlideKey
from histopathology.utils.tiles_selection_utils import SlideNode, TilesSelector
from histopathology.utils.viz_utils import load_image_dict, save_figure


ResultsType = Dict[ResultsKey, List[Any]]


def validate_class_names_for_plot_options(
    class_names: Optional[Sequence[str]], plot_options: Collection[PlotOption]
) -> Optional[Sequence[str]]:
    """Make sure that class names are provided if `PlotOption.CONFUSION_MATRIX` is among the chosen plot_options."""
    if PlotOption.CONFUSION_MATRIX in plot_options and not class_names:
        raise ValueError(
            "No class_names were provided while activating confusion matrix plotting. You need to specify the class "
            "names to use the `PlotOption.CONFUSION_MATRIX`"
        )
    return class_names


def save_scores_histogram(results: ResultsType, figures_dir: Path) -> None:
    """Plots and saves histogram scores figure in its dedicated directory.

    :param results: List that contains slide_level dicts
    :param figures_dir: The path to the directory where to save the histogram scores.
    """
    logging.info("Plotting histogram ...")
    fig = plot_scores_hist(results)
    save_figure(fig=fig, figpath=figures_dir / "hist_scores.png")


def save_confusion_matrix(results: ResultsType, class_names: Sequence[str], figures_dir: Path) -> None:
    """Plots and saves confusion matrix figure in its dedicated directory.

    :param class_names: List of class names.
    :param figures_dir: The path to the directory where to save the confusion matrix.
    """
    cf_matrix_n = confusion_matrix(results[ResultsKey.TRUE_LABEL], results[ResultsKey.PRED_LABEL], normalize="pred")
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
    :param slides_dataset: The slides dataset from where to pick the slide.
    :param tile_size: Size of each tile. Default 224.
    :param level: Magnification at which tiles are available (e.g. PANDA levels are 0 for original,
        1 for 4x downsampled, 2 for 16x downsampled). Default 1.
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
        plot_options: Collection[PlotOption],
        level: int = 1,
        tile_size: int = 224,
        num_columns: int = 4,
        figsize: Tuple[int, int] = (10, 10),
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        """_summary_

        :param plot_options: A set of plot options to produce the desired plot outputs.
        :param level: Magnification at which tiles are available (e.g. PANDA levels are 0 for original,
            1 for 4x downsampled, 2 for 16x downsampled). Default 1.
        :param tile_size: _description_, defaults to 224
        :param num_columns: Number of columns to create the subfigures grid, defaults to 4
        :param figsize: The figure size of tiles attention plots, defaults to (10, 10)
        :param class_names: List of class names, defaults to None
        :param slides_dataset: The slides dataset from where to load the whole slide images, defaults to None
        """
        self.plot_options = plot_options
        self.class_names = validate_class_names_for_plot_options(class_names, plot_options)
        self.level = level
        self.tile_size = tile_size
        self.num_columns = num_columns
        self.figsize = figsize
        self.slides_dataset: Optional[SlidesDataset] = None

    def save_slide_node_figures(
        self, case: str, slide_node: SlideNode, outputs_dir: Path, results: ResultsType
    ) -> None:
        """Plots and saves all slide related figures, e.g., `TOP_BOTTOM_TILES` and `SLIDE_THUMBNAIL_HEATMAP`"""

        case_dir = make_figure_dirs(subfolder=case, parent_dir=outputs_dir)

        if PlotOption.TOP_BOTTOM_TILES in self.plot_options:
            save_top_and_bottom_tiles(
                case=case,
                slide_node=slide_node,
                figures_dir=case_dir,
                num_columns=self.num_columns,
                figsize=self.figsize,
            )
        if PlotOption.SLIDE_THUMBNAIL_HEATMAP in self.plot_options:
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

    def save_plots(self, outputs_dir: Path, tiles_selector: Optional[TilesSelector], results: ResultsType) -> None:
        """Plots and saves all selected plot options during inference (validation or test) time.

        :param outputs_dir: The root output directory where to save plots figures.
        :param tiles_selector: The tiles selector used to select top and bottom tiles from top and bottom slides.
        :param results: A dictionary of the validation or tests results.
        :param stage: The model stage: validation or test.
        """

        logging.info(f"Plotting {[opt.value for opt in self.plot_options]}. All figures will be saved in {outputs_dir}")
        figures_dir = make_figure_dirs(subfolder="fig", parent_dir=outputs_dir)

        if PlotOption.HISTOGRAM in self.plot_options:
            save_scores_histogram(results=results, figures_dir=figures_dir)

        if PlotOption.CONFUSION_MATRIX in self.plot_options:
            assert self.class_names
            save_confusion_matrix(results, class_names=self.class_names, figures_dir=figures_dir)

        if tiles_selector:
            for class_id in range(tiles_selector.n_classes):

                for slide_node in tiles_selector.top_slides_heaps[class_id]:
                    case = "TN" if class_id == 0 else f"TP_{class_id}"
                    self.save_slide_node_figures(case, slide_node, outputs_dir, results)

                for slide_node in tiles_selector.bottom_slides_heaps[class_id]:
                    case = "FP" if class_id == 0 else f"FN_{class_id}"
                    self.save_slide_node_figures(case, slide_node, outputs_dir, results)
