#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Collection, List, Optional, Sequence, Tuple, Dict

from sklearn.metrics import confusion_matrix
from torch import Tensor
import matplotlib.pyplot as plt

from health_cpath.datasets.base_dataset import SlidesDataset
from health_cpath.preprocessing.loading import LoadingParams
from health_cpath.utils.viz_utils import (
    plot_attention_tiles,
    plot_heatmap_overlay,
    plot_attention_histogram,
    plot_normalized_confusion_matrix,
    plot_scores_hist,
    plot_slide,
)
from health_cpath.utils.analysis_plot_utils import plot_pr_curve, format_pr_or_roc_axes
from health_cpath.utils.naming import PlotOption, ResultsKey, SlideKey
from health_cpath.utils.tiles_selection_utils import SlideNode, TilesSelector
from health_cpath.utils.viz_utils import save_figure


ResultsType = Dict[ResultsKey, List[Any]]
SlideDictType = Dict[SlideKey, Any]


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


def save_scores_histogram(results: ResultsType, figures_dir: Path, stage: str = '') -> None:
    """Plots and saves histogram scores figure in its dedicated directory.

    :param results: Dict of lists that contains slide_level results
    :param figures_dir: The path to the directory where to save the histogram scores.
    :param stage: Test or validation, used to name the figure. Empty string by default.
    """
    fig = plot_scores_hist(results)
    save_figure(fig=fig, figpath=figures_dir / f"hist_scores_{stage}.png")


def save_pr_curve(results: ResultsType, figures_dir: Path, stage: str = '') -> None:
    """Plots and saves PR curve figure in its dedicated directory. This implementation
    only works for binary classification.
''
    :param results: Dict of lists that contains slide_level results
    :param figures_dir: The path to the directory where to save the histogram scores
    :param stage: Test or validation, used to name the figure. Empty string by default.
    """
    true_labels = [i.item() if isinstance(i, Tensor) else i for i in results[ResultsKey.TRUE_LABEL]]
    if len(set(true_labels)) == 2:
        scores = [i.item() if isinstance(i, Tensor) else i for i in results[ResultsKey.PROB]]
        fig, ax = plt.subplots()
        plot_pr_curve(true_labels, scores, legend_label=stage, ax=ax)
        ax.legend()
        format_pr_or_roc_axes(plot_type='pr', ax=ax)
        save_figure(fig=fig, figpath=figures_dir / f"pr_curve_{stage}.png")
    else:
        logging.warning("The PR curve plot implementation works only for binary cases, this plot will be skipped.")


def save_confusion_matrix(results: ResultsType, class_names: Sequence[str], figures_dir: Path, stage: str = '') -> None:
    """Plots and saves confusion matrix figure in its dedicated directory.

    :param results: Dict of lists that contains slide_level results
    :param class_names: List of class names.
    :param figures_dir: The path to the directory where to save the confusion matrix.
    :param stage: Test or validation, used to name the figure. Empty string by default.
    """
    true_labels = [i.item() if isinstance(i, Tensor) else i for i in results[ResultsKey.TRUE_LABEL]]
    pred_labels = [i.item() if isinstance(i, Tensor) else i for i in results[ResultsKey.PRED_LABEL]]
    all_potential_labels = list(range(len(class_names)))
    true_labels_diff_expected = set(true_labels).difference(set(all_potential_labels))
    pred_labels_diff_expected = set(pred_labels).difference(set(all_potential_labels))

    if true_labels_diff_expected != set():
        raise ValueError("More entries were found in true labels than are available in class names")
    if pred_labels_diff_expected != set():
        raise ValueError("More entries were found in predicted labels than are available in class names")

    cf_matrix_n = confusion_matrix(
        true_labels,
        pred_labels,
        labels=all_potential_labels,
        normalize="true"
    )

    fig = plot_normalized_confusion_matrix(cm=cf_matrix_n, class_names=(class_names))
    save_figure(fig=fig, figpath=figures_dir / f"normalized_confusion_matrix_{stage}.png")


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


def save_slide_thumbnail(case: str, slide_node: SlideNode, slide_dict: SlideDictType, figures_dir: Path) -> None:
    """Plots and saves a slide thumbnail

    :param case: The report case (e.g., TP, FN, ...)
    :param slide_node: The slide node that encapsulates the slide metadata.
    :param slide_dict: The slide dictionary that contains the slide image and other metadata.
    :param figures_dir: The path to the directory where to save the plots.
    """
    fig = plot_slide(case=case, slide_node=slide_node, slide_image=slide_dict[SlideKey.IMAGE], scale=1.0)
    save_figure(fig=fig, figpath=figures_dir / f"{slide_node.slide_id}_thumbnail.png")


def save_attention_heatmap(
    case: str,
    slide_node: SlideNode,
    slide_dict: SlideDictType,
    figures_dir: Path,
    results: ResultsType,
    tile_size: int = 224,
    should_upscale_coords: bool = False,
) -> None:
    """Plots and saves a slide thumbnail and attention heatmap

    :param case: The report case (e.g., TP, FN, ...)
    :param slide_node: The slide node that encapsulates the slide metadata.
    :param slide_dict: The slide dictionary that contains the slide image and other metadata.
    :param figures_dir: The path to the directory where to save the plots.
    :param results: Dict containing ResultsKey keys (e.g. slide id) and values as lists of output slides.
    :param tile_size: Size of each tile. Default 224.
    :param should_upscale_coords: Whether to upscale the coordinates of the attention heatmap. Default False.
    """
    fig = plot_heatmap_overlay(
        case=case,
        slide_node=slide_node,
        slide_dict=slide_dict,
        results=results,
        tile_size=tile_size,
        should_upscale_coords=should_upscale_coords,
    )
    save_figure(fig=fig, figpath=figures_dir / f"{slide_node.slide_id}_heatmap.png")


def save_attention_histogram(case: str, slide_node: SlideNode, results: ResultsType, figures_dir: Path) -> None:
    """Plots a histogram of the attention values of the tiles in a bag.

    :param case: The report case (e.g., TP, FN, ...)
    :param slide_node: The slide node that encapsulates the slide metadata.
    :param results: Dict containing ResultsKey keys (e.g. slide id) and values as lists of output slides.
    :param figures_dir: The path to the directory where to save the plot.
    """
    fig = plot_attention_histogram(case=case, slide_node=slide_node, results=results)
    save_figure(fig=fig, figpath=figures_dir / f"{slide_node.slide_id}_histogram.png")


def make_figure_dirs(subfolder: str, parent_dir: Path) -> Path:
    """Create the figure directory"""
    figures_dir = parent_dir / subfolder
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


class DeepMILPlotsHandler:
    def __init__(
        self,
        plot_options: Collection[PlotOption],
        loading_params: LoadingParams,
        tile_size: int = 224,
        num_columns: int = 4,
        figsize: Tuple[int, int] = (10, 10),
        stage: str = '',
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Class that handles the plotting of DeepMIL results.

        :param plot_options: A set of plot options to produce the desired plot outputs.
        :param loading_params: The loading parameters to use when loading the whole slide images.
        :param tile_size: The size of the tiles to use when plotting the attention tiles, defaults to 224
        :param num_columns: Number of columns to create the subfigures grid, defaults to 4
        :param figsize: The figure size of tiles attention plots, defaults to (10, 10)
        :param stage: Test or Validation, used to name the plots
        :param class_names: List of class names, defaults to None
        """

        self.plot_options = plot_options
        self.class_names = validate_class_names_for_plot_options(class_names, plot_options)
        self.tile_size = tile_size
        self.num_columns = num_columns
        self.figsize = figsize
        self.stage = stage
        self.loading_params = loading_params
        self.should_upscale_coords = loading_params.should_upscale_coordinates()
        self.loading_params.set_roi_type_to_foreground()
        self.slides_dataset: Optional[SlidesDataset] = None

    def get_slide_dict(self, slide_node: SlideNode) -> SlideDictType:
        """Returns the slide dictionary for a given slide node"""
        assert self.slides_dataset is not None, "Cannot plot attention heatmap or wsi without slides dataset"
        slide_index = self.slides_dataset.dataset_df.index.get_loc(slide_node.slide_id)
        assert isinstance(slide_index, int), f"Got non-unique slide ID: {slide_node.slide_id}"
        slide_dict = self.slides_dataset[slide_index]
        loader = self.loading_params.get_load_roid_transform()
        slide_dict = loader(slide_dict)
        return slide_dict

    def save_slide_node_figures(
        self, case: str, slide_node: SlideNode, outputs_dir: Path, results: ResultsType
    ) -> None:
        """Plots and saves all slide related figures: `TOP_BOTTOM_TILES`, `SLIDE_THUMBNAIL` and `ATTENTION_HEATMAP`."""
        case_dir = make_figure_dirs(subfolder=case, parent_dir=outputs_dir)

        if PlotOption.TOP_BOTTOM_TILES in self.plot_options:
            save_top_and_bottom_tiles(case, slide_node, case_dir, self.num_columns, self.figsize)

        if PlotOption.ATTENTION_HISTOGRAM in self.plot_options:
            save_attention_histogram(case, slide_node, results, case_dir)

        if PlotOption.ATTENTION_HEATMAP in self.plot_options or PlotOption.SLIDE_THUMBNAIL in self.plot_options:
            slide_dict = self.get_slide_dict(slide_node=slide_node)

            if PlotOption.SLIDE_THUMBNAIL in self.plot_options:
                save_slide_thumbnail(case=case, slide_node=slide_node, slide_dict=slide_dict, figures_dir=case_dir)

            if PlotOption.ATTENTION_HEATMAP in self.plot_options:
                save_attention_heatmap(
                    case, slide_node, slide_dict, case_dir, results, self.tile_size, self.should_upscale_coords
                )

    def save_plots(self, outputs_dir: Path, tiles_selector: Optional[TilesSelector], results: ResultsType) -> None:
        """Plots and saves all selected plot options during inference (validation or test) time.

        :param outputs_dir: The root output directory where to save plots figures.
        :param tiles_selector: The tiles selector used to select top and bottom tiles from top and bottom slides.
        :param results: A dictionary of the validation or tests results.
        :param stage: The model stage: validation or test.
        """
        if self.plot_options:
            logging.info(
                f"Plotting {[opt.value for opt in self.plot_options]}..."
            )
            figures_dir = make_figure_dirs(subfolder="fig", parent_dir=outputs_dir)

            if PlotOption.PR_CURVE in self.plot_options:
                save_pr_curve(results=results, figures_dir=figures_dir, stage=self.stage)

            if PlotOption.HISTOGRAM in self.plot_options:
                save_scores_histogram(results=results, figures_dir=figures_dir, stage=self.stage,)

            if PlotOption.CONFUSION_MATRIX in self.plot_options:
                assert self.class_names
                save_confusion_matrix(results, class_names=self.class_names, figures_dir=figures_dir, stage=self.stage)

            if tiles_selector:
                for class_id in range(tiles_selector.n_classes):

                    for slide_node in tiles_selector.top_slides_heaps[class_id]:
                        case = "TN" if class_id == 0 else f"TP_{class_id}"
                        self.save_slide_node_figures(case, slide_node, outputs_dir, results)

                    for slide_node in tiles_selector.bottom_slides_heaps[class_id]:
                        case = "FP" if class_id == 0 else f"FN_{class_id}"
                        self.save_slide_node_figures(case, slide_node, outputs_dir, results)
