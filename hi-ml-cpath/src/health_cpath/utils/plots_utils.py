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
import numpy as np

from health_cpath.datasets.base_dataset import SlidesDataset
from health_cpath.preprocessing.loading import LoadingParams
from health_cpath.utils.viz_utils import (
    plot_attention_tiles,
    plot_heatmap_overlay,
    plot_attention_histogram,
    plot_normalized_and_non_normalized_confusion_matrices,
    plot_scores_hist,
    plot_slide,
)
from health_cpath.utils.analysis_plot_utils import plot_pr_curve, format_pr_or_roc_axes, plot_roc_curve
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


def save_pr_curve(
    results: ResultsType, figures_dir: Path, stage: str = '', stratify_metadata: Optional[List[Any]] = None
) -> None:
    """Plots and saves PR curve figure in its dedicated directory. This implementation
        only works for binary classification.
    ''
        :param results: Dict of lists that contains slide_level results
        :param figures_dir: The path to the directory where to save the figure
        :param stage: Test or validation, used to name the figure. Empty string by default.
    """
    true_labels = get_list_from_results_dict(results=results, results_key=ResultsKey.TRUE_LABEL)
    if len(set(true_labels)) == 2:
        scores = get_list_from_results_dict(results=results, results_key=ResultsKey.PROB)
        fig, ax = plt.subplots()
        plot_pr_curve(true_labels, scores, legend_label=stage, ax=ax)
        if stratify_metadata is not None:
            stratified_outputs = get_stratified_outputs(
                true_labels=true_labels, scores=scores, stratify_metadata=stratify_metadata
            )
            for key in stratified_outputs.keys():
                true_stratified = stratified_outputs[key][0]
                pred_stratified = stratified_outputs[key][1]
                plot_pr_curve(true_stratified, pred_stratified, legend_label=f"{stage}_{key}", ax=ax)
        ax.legend()
        format_pr_or_roc_axes(plot_type='pr', ax=ax)
        save_figure(fig=fig, figpath=figures_dir / f"pr_curve_{stage}.png")
    else:
        logging.warning("The PR curve plot implementation works only for binary cases, this plot will be skipped.")


def save_roc_curve(
    results: ResultsType, figures_dir: Path, stage: str = '', stratify_metadata: Optional[List[Any]] = None
) -> None:
    """Plots and saves ROC curve figure in its dedicated directory. This implementation
    only works for binary classification.

    :param results: Dict of lists that contains slide_level results
    :param figures_dir: The path to the directory where to save the figure
    :param stage: Test or validation, used to name the figure. Empty string by default.
    :param stratify_metadata: A list containing metadata values to plot stratified curves.
    """
    true_labels = get_list_from_results_dict(results=results, results_key=ResultsKey.TRUE_LABEL)
    if len(set(true_labels)) == 2:
        scores = get_list_from_results_dict(results=results, results_key=ResultsKey.PROB)
        fig, ax = plt.subplots()
        plot_roc_curve(true_labels, scores, legend_label=stage, ax=ax)
        if stratify_metadata is not None:
            stratified_outputs = get_stratified_outputs(
                true_labels=true_labels, scores=scores, stratify_metadata=stratify_metadata
            )
            for key in stratified_outputs.keys():
                true_stratified = stratified_outputs[key][0]
                pred_stratified = stratified_outputs[key][1]
                plot_roc_curve(true_stratified, pred_stratified, legend_label=f"{stage}_{key}", ax=ax)
        ax.legend()
        format_pr_or_roc_axes(plot_type='roc', ax=ax)
        save_figure(fig=fig, figpath=figures_dir / f"roc_curve_{stage}.png")
    else:
        logging.warning("The ROC curve plot implementation works only for binary cases, this plot will be skipped.")


def save_confusion_matrix(results: ResultsType, class_names: Sequence[str], figures_dir: Path, stage: str = '') -> None:
    """Plots and saves confusion matrix figure in its dedicated directory.

    :param results: Dict of lists that contains slide_level results
    :param class_names: List of class names.
    :param figures_dir: The path to the directory where to save the confusion matrix.
    :param stage: Test or validation, used to name the figure. Empty string by default.
    """
    true_labels = get_list_from_results_dict(results=results, results_key=ResultsKey.TRUE_LABEL)
    pred_labels = get_list_from_results_dict(results=results, results_key=ResultsKey.PRED_LABEL)
    all_potential_labels = list(range(len(class_names)))
    true_labels_diff_expected = set(true_labels).difference(set(all_potential_labels))
    pred_labels_diff_expected = set(pred_labels).difference(set(all_potential_labels))

    if true_labels_diff_expected != set():
        raise ValueError("More entries were found in true labels than are available in class names")
    if pred_labels_diff_expected != set():
        raise ValueError("More entries were found in predicted labels than are available in class names")

    cf_matrix = confusion_matrix(true_labels, pred_labels, labels=all_potential_labels)
    cf_matrix_n = confusion_matrix(true_labels, pred_labels, labels=all_potential_labels, normalize="true")

    fig = plot_normalized_and_non_normalized_confusion_matrices(cm=cf_matrix, cm_n=cf_matrix_n, class_names=class_names)
    save_figure(fig=fig, figpath=figures_dir / f"confusion_matrices_{stage}.png")


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
    extra_slide_dict: Optional[Dict[SlideKey, Any]] = None,
) -> None:
    """Plots and saves a slide thumbnail and attention heatmap

    :param case: The report case (e.g., TP, FN, ...)
    :param slide_node: The slide node that encapsulates the slide metadata.
    :param slide_dict: The slide dictionary that contains the slide image and other metadata.
    :param figures_dir: The path to the directory where to save the plots.
    :param results: Dict containing ResultsKey keys (e.g. slide id) and values as lists of output slides.
    :param tile_size: Size of each tile. Default 224.
    :param should_upscale_coords: Whether to upscale the coordinates of the attention heatmap. Default False.
    :param extra_slide_dict: An optional dictionary containing an extra slide image and metadata. Default None.
    """
    fig = plot_heatmap_overlay(
        case=case,
        slide_node=slide_node,
        slide_dict=slide_dict,
        results=results,
        tile_size=tile_size,
        should_upscale_coords=should_upscale_coords,
        extra_slide_dict=extra_slide_dict,
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


def get_list_from_results_dict(results: ResultsType, results_key: ResultsKey) -> List[Any]:
    """Get a specific results list from the slide_level results dictionary, we extract items from tensors
    here so that it's compatible with inputs formats of scikit learn functions.

    :param results: Dict of lists that contains slide_level results
    :param results_key: ResultsKey key for the list to be retrieved
    """
    return [i.item() if isinstance(i, Tensor) else i for i in results[results_key]]


def get_stratified_outputs(
    true_labels: List[Any], scores: List[Any], stratify_metadata: List[Any]
) -> Dict[str, List[Any]]:
    """
    Get stratified true labels and predictions given metadata, from all true and prediction labels
    to plot stratified curves.

    :param true_labels: list of true labels.
    :param scores: list of prediction scores.
    :param stratify_metadata: list containing the corresponding metadata values on which to stratify results.
    :return: A dictionary of stratified outputs, where a key is a unique value from the metadata,
    and value contains a list of two lists - true labels in first list, predicted labels in second list.
    """
    unique_vals, unique_counts = np.unique(stratify_metadata, return_counts=True)  # metadata should not contain nans
    stratified_outputs = {}
    for val, count in zip(unique_vals, unique_counts):
        idxs = [i for i, x in enumerate(stratify_metadata) if x == val]
        assert len(idxs) == count
        true_stratified = [true_labels[i] for i in idxs]
        pred_stratified = [scores[i] for i in idxs]
        stratified_outputs[val] = [true_stratified, pred_stratified]
    return stratified_outputs


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
        stratify_plots_by: Optional[str] = None,
    ) -> None:
        """Class that handles the plotting of DeepMIL results.

        :param plot_options: A set of plot options to produce the desired plot outputs.
        :param loading_params: The loading parameters to use when loading the whole slide images.
        :param tile_size: The size of the tiles to use when plotting the attention tiles, defaults to 224
        :param num_columns: Number of columns to create the subfigures grid, defaults to 4
        :param figsize: The figure size of tiles attention plots, defaults to (10, 10)
        :param stage: Test or Validation, used to name the plots
        :param class_names: List of class names, defaults to None
        :param stratify_plots_by: Name of metadata field to stratify output plots (PR curve, ROC curve).
        `None` by default (no stratification).
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
        self.extra_slides_dataset: Optional[SlidesDataset] = None
        self.stratify_plots_by = stratify_plots_by

    def get_slide_dict(self, slide_node: SlideNode, slides_dataset: SlidesDataset) -> Optional[SlideDictType]:
        """Returns the slide dictionary for a given slide node from a slides dataset.

        :param slide_node: The slide node that encapsulates the slide metadata.
        :param slides_dataset: The slides dataset that contains the slide image and other metadata.
        """
        try:
            slide_index = slides_dataset.dataset_df.index.get_loc(slide_node.slide_id)
        except KeyError:
            logging.warning(f"Could not find slide {slide_node.slide_id} in the dataset. Skipping extra slide...")
            return None
        assert isinstance(slide_index, int), f"Got non-unique slide ID: {slide_node.slide_id}"
        slide_dict = slides_dataset[slide_index]
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
            assert self.slides_dataset is not None, "Cannot plot attention heatmap or thumbnail without slides dataset"
            slide_dict = self.get_slide_dict(slide_node=slide_node, slides_dataset=self.slides_dataset)
            assert slide_dict is not None, "Slide dict is None. Cannot plot attention heatmap or thumbnail."

            if PlotOption.SLIDE_THUMBNAIL in self.plot_options:
                save_slide_thumbnail(case=case, slide_node=slide_node, slide_dict=slide_dict, figures_dir=case_dir)

            if PlotOption.ATTENTION_HEATMAP in self.plot_options:
                if self.extra_slides_dataset is not None:
                    extra_slide_dict = self.get_slide_dict(slide_node, self.extra_slides_dataset)
                else:
                    extra_slide_dict = None
                save_attention_heatmap(
                    case,
                    slide_node,
                    slide_dict,
                    case_dir,
                    results,
                    self.tile_size,
                    self.should_upscale_coords,
                    extra_slide_dict,
                )

    def get_metadata(self, results: ResultsType) -> Optional[List[Any]]:
        """
        Get metadata of outputs (validation or test) from slides dataset to stratify plots (e.g PR curve, ROC curve).
        Returns metadata values of the results slides specified in `stratify_plots by` from slides dataset.
        Returns `None` if slides dataset is `None` or if `stratify_plots_by` is set to `None` (no stratification).

        :param results: Dict containing ResultsKey keys (e.g. slide id) and values as lists of output slides.
        """
        if self.slides_dataset is None or self.stratify_plots_by is None:
            stratify_metadata = None
            if self.stratify_plots_by is not None:
                logging.warning("Slides dataset is missing so stratified plots will be skipped.")
        else:
            slides_df = self.slides_dataset.dataset_df
            all_slide_ids = slides_df.index.to_list()
            output_slide_ids = [x[0] for x in results[ResultsKey.SLIDE_ID]]  # get unique slide ID from bags
            stratify_metadata = []
            for slide in output_slide_ids:
                idx = all_slide_ids.index(slide)
                sample = self.slides_dataset[idx]
                if self.stratify_plots_by not in sample[SlideKey.METADATA]:
                    logging.warning(
                        f"{self.stratify_plots_by} not available in the slides dataset metadata, \
                        make sure the dataset includes stratify_plots_by, so stratified plots will be skipped."
                    )
                    return None
                stratify_metadata.append(sample[SlideKey.METADATA][self.stratify_plots_by])
        return stratify_metadata

    def save_plots(self, outputs_dir: Path, tiles_selector: Optional[TilesSelector], results: ResultsType) -> None:
        """Plots and saves all selected plot options during inference (validation or test) time.

        :param outputs_dir: The root output directory where to save plots figures.
        :param tiles_selector: The tiles selector used to select top and bottom tiles from top and bottom slides.
        :param results: A dictionary of the validation or tests results.
        :param stage: The model stage: validation or test.
        """
        if self.plot_options:
            logging.info(f"Plotting {[opt.value for opt in self.plot_options]}...")
            figures_dir = make_figure_dirs(subfolder="fig", parent_dir=outputs_dir)

            stratify_metadata = self.get_metadata(results=results)

            if PlotOption.PR_CURVE in self.plot_options:
                save_pr_curve(
                    results=results, figures_dir=figures_dir, stage=self.stage, stratify_metadata=stratify_metadata
                )

            if PlotOption.ROC_CURVE in self.plot_options:
                save_roc_curve(
                    results=results, figures_dir=figures_dir, stage=self.stage, stratify_metadata=stratify_metadata
                )

            if PlotOption.HISTOGRAM in self.plot_options:
                save_scores_histogram(
                    results=results,
                    figures_dir=figures_dir,
                    stage=self.stage,
                )

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
