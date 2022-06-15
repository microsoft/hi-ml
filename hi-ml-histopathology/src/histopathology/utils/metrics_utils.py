#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collection

from math import ceil
from pathlib import Path
from typing import Sequence, List, Any, Dict, Union, Tuple

from histopathology.utils.naming import ResultsKey
from histopathology.utils.heatmap_utils import location_selected_tiles
from histopathology.utils.tiles_selection_utils import SlideNode


def plot_scores_hist(
    results: Dict, prob_col: str = ResultsKey.CLASS_PROBS, gt_col: str = ResultsKey.TRUE_LABEL
) -> plt.Figure:
    """
    :param results: List that contains slide_level dicts
    :param prob_col: column name that contains the scores
    :param gt_col: column name that contains the true label
    :return: matplotlib figure of the scores histogram by class
    """
    n_classes = len(results[prob_col][0])
    scores_class = []
    for j in range(n_classes):
        scores = [results[prob_col][i][j].cpu().item() for i, gt in enumerate(results[gt_col]) if gt == j]
        scores_class.append(scores)
    fig, ax = plt.subplots()
    ax.hist(scores_class, label=[str(i) for i in range(n_classes)], alpha=0.5)
    ax.set_xlabel("Predicted Score")
    ax.legend()
    return fig


def plot_attention_tiles(
    slide_node: SlideNode, top: bool, case: str, num_columns: int, figsize: Tuple[int, int]
) -> plt.Figure:
    """Plot top or bottom tiles figures along with their attention scores.

    :param slide_node: The slide node for which we would like to plot attention tiles.
    :param top: Decides which tiles to plot. If True, plots top tile nodes of the slide_node. Otherwise, plots bottom
        tile nodes.
    :param case: The report case (e.g., TP, FN, ...)
    :param num_columns: Number of columns to create the subfigures grid, defaults to 4
    :param figsize: The figure size of tiles attention plots, defaults to (10, 10)
    """
    tile_nodes = slide_node.top_tiles if top else slide_node.bottom_tiles
    num_rows = int(ceil(len(tile_nodes) / num_columns))
    assert (
        num_rows > 0
    ), "The number of selected top and bottom tiles is too low. Try debugging with a higher num_top_tiles and/or a "
    "higher number of batches."

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=figsize)
    fig.suptitle(
        f"{case}: {slide_node.slide_id} P={abs(slide_node.prob_score):.2f} \n True label: {slide_node.true_label}"
    )

    for ax, tile_node in zip(axs.flat, tile_nodes):
        ax.imshow(np.transpose(tile_node.data.numpy(), (1, 2, 0)), clim=(0, 255), cmap="gray")
        ax.set_title("%.6f" % tile_node.attn)

    for ax in axs.flat:
        ax.set_axis_off()
    return fig


def plot_slide(slide_image: np.ndarray, scale: float) -> plt.Figure:
    """Plots a slide thumbnail from a given slide image and scale.
    :param slide_image: Numpy array of the slide image (shape: [3, H, W]).
    :return: matplotlib figure of the slide thumbnail.
    """
    fig, ax = plt.subplots()
    slide_image = slide_image.transpose(1, 2, 0)
    ax.imshow(slide_image)
    ax.set_axis_off()
    original_size = fig.get_size_inches()
    fig.set_size_inches((original_size[0] * scale, original_size[1] * scale))
    return fig


def plot_heatmap_overlay(
    slide_node: SlideNode,
    slide_image: np.ndarray,
    results: Dict[ResultsKey, List[Any]],
    location_bbox: List[int],
    tile_size: int = 224,
    level: int = 1,
) -> plt.Figure:
    """Plots heatmap of selected tiles (e.g. tiles in a bag) overlay on the corresponding slide.
    :param slide_node: The slide node that encapsulates the slide metadata.
    :param slide_image: Numpy array of the slide image (shape: [3, H, W]).
    :param results: Dict containing ResultsKey keys (e.g. slide id) and values as lists of output slides.
    :param tile_size: Size of each tile. Default 224.
    :param level: Magnification at which tiles are available (e.g. PANDA levels are 0 for original,
    1 for 4x downsampled, 2 for 16x downsampled). Default 1.
    :param location_bbox: Location of the bounding box of the slide.
    :return: matplotlib figure of the heatmap of the given tiles on slide.
    """
    fig, ax = plt.subplots()
    slide_image = slide_image.transpose(1, 2, 0)
    ax.imshow(slide_image)
    ax.set_xlim(0, slide_image.shape[1])
    ax.set_ylim(slide_image.shape[0], 0)

    coords_list = []
    slide_ids = [item[0] for item in results[ResultsKey.SLIDE_ID]]
    slide_idx = slide_ids.index(slide_node.slide_id)
    attentions = results[ResultsKey.BAG_ATTN][slide_idx]

    # for each tile in the bag
    for tile_idx in range(len(results[ResultsKey.IMAGE_PATH][slide_idx])):
        tile_coords = np.transpose(
            np.array(
                [
                    results[ResultsKey.TILE_LEFT][slide_idx][tile_idx].cpu().numpy(),
                    results[ResultsKey.TILE_TOP][slide_idx][tile_idx].cpu().numpy(),
                ]
            )
        )
        coords_list.append(tile_coords)

    coords = np.array(coords_list)
    attentions = np.array(attentions.cpu()).reshape(-1)

    sel_coords = location_selected_tiles(tile_coords=coords, location_bbox=location_bbox, level=level)
    cmap = plt.cm.get_cmap("Reds")

    tile_xs, tile_ys = sel_coords.T
    rects = [patches.Rectangle(xy, tile_size, tile_size) for xy in zip(tile_xs, tile_ys)]

    pc = collection.PatchCollection(rects, match_original=True, cmap=cmap, alpha=0.5, edgecolor="black")
    pc.set_array(np.array(attentions))
    pc.set_clim([0, 1])
    ax.add_collection(pc)
    plt.colorbar(pc, ax=ax)
    return fig


def plot_normalized_confusion_matrix(cm: np.ndarray, class_names: Sequence[str]) -> plt.Figure:
    """Plots a normalized confusion matrix and returns the figure.
    param cm: Normalized confusion matrix to be plotted.
    param class_names: List of class names.
    """
    import seaborn as sns

    fig, ax = plt.subplots()
    ax = sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    return fig


def resize_and_save(width_inch: int, height_inch: int, filename: Union[Path, str], dpi: int = 150) -> None:
    """
    Resizes the present figure to the given (width, height) in inches, and saves it to the given filename.
    :param width_inch: The width of the figure in inches.
    :param height_inch: The height of the figure in inches.
    :param filename: The filename to save to.
    :param dpi: Image resolution in dots per inch
    """
    fig = plt.gcf()
    fig.set_size_inches(width_inch, height_inch)
    # Workaround for Exception in Tkinter callback
    fig.canvas.start_event_loop(sys.float_info.min)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
