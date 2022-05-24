#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import sys
from pathlib import Path
from typing import Sequence, Tuple, List, Any, Dict, Union

import torch
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import matplotlib.patches as patches
import matplotlib.collections as collection

from histopathology.models.transforms import load_pil_image
from histopathology.utils.naming import ResultsKey
from histopathology.utils.heatmap_utils import location_selected_tiles


def select_k_tiles(results: Dict, n_tiles: int = 5, n_slides: int = 5, label: int = 1,
                   highest_pred: bool = False, highest_att: bool = True,
                   slide_col: str = ResultsKey.SLIDE_ID, gt_col: str = ResultsKey.TRUE_LABEL,
                   attn_col: str = ResultsKey.BAG_ATTN, prob_col: str = ResultsKey.CLASS_PROBS,
                   return_col: str = ResultsKey.IMAGE_PATH) -> List[Tuple[Any, Any, List[Any], List[Any]]]:
    """
    :param results: List that contains slide_level dicts
    :param n_tiles: number of tiles to be selected for each slide
    :param n_slides: number of slides to be selected
    :param label: which label to use to select slides
    :param highest_pred: criteria to be used to sort the slides by highest prediction score
    :param highest_att: criteria to be used to sort the tiles by highest attention score
    :param slide_col: column name that contains slide identifiers
    :param gt_col: column name that contains labels
    :param attn_col: column name that contains scores used to sort tiles
    :param prob_col: column name that contains scores used to sort slides
    :param return_col: column name of the values we want to return for each tile
    :return: tuple containing the slides id, the slide score, the tile ids, the tiles scores
    """
    # TODO: Refactor into separate functions to select slides by probabilities and tiles by attentions
    tmp_s = [(results[prob_col][i][label], i) for i, gt in enumerate(results[gt_col]) if gt == label]  # type ignore

    if len(tmp_s) == 0:
        return []

    tmp_s.sort(reverse=highest_pred)
    _, sorted_idx = zip(*tmp_s)

    k_idx = []
    for _, slide_idx in enumerate(sorted_idx[:n_slides]):
        tmp = results[attn_col][slide_idx]
        _, t_indices = torch.sort(tmp, descending=highest_att)
        k_tiles = []
        scores = []
        for t_idx in t_indices[0][:n_tiles]:
            k_tiles.append(results[return_col][slide_idx][t_idx])
            scores.append(results[attn_col][slide_idx][0][t_idx])
        # slide_ids are duplicated
        k_idx.append((results[slide_col][slide_idx][0],
                      results[prob_col][slide_idx],
                      k_tiles, scores))
    return k_idx


def plot_scores_hist(results: Dict, prob_col: str = ResultsKey.CLASS_PROBS,
                     gt_col: str = ResultsKey.TRUE_LABEL) -> plt.Figure:
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


def plot_attention_tiles(slide: str, scores: List[float], paths: List, attn: List, case: str, ncols: int = 5,
                         size: Tuple = (10, 10)) -> plt.Figure:
    """
    :param slide: slide identifier
    :param scores: predicted scores of each class for the slide
    :param paths: list of paths to tiles belonging to the slide
    :param attn: list of scores belonging to the tiles in paths. paths and attn are expected to have the same shape
    :param case: string used to define the title of the plot e.g. TP
    :param ncols: number of cols the produced figure should have
    :param size: size of the plot
    :return: matplotlib figure of each tile in paths with attn score
    """
    nrows = int(ceil(len(paths) / ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)
    fig.suptitle(f"{case}: {slide} P=%.2f" % max(scores))
    for i in range(len(paths)):
        img = load_pil_image(paths[i])
        axs.ravel()[i].imshow(img, clim=(0, 255), cmap='gray')
        axs.ravel()[i].set_title("%.6f" % attn[i].cpu().item())
    for i in range(len(axs.ravel())):
        axs.ravel()[i].set_axis_off()
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


def plot_heatmap_overlay(slide: str,
                         slide_image: np.ndarray,
                         results: Dict[ResultsKey, List[Any]],
                         location_bbox: List[int],
                         tile_size: int = 224,
                         level: int = 1) -> plt.Figure:
    """Plots heatmap of selected tiles (e.g. tiles in a bag) overlay on the corresponding slide.
    :param slide: slide identifier.
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
    slide_idx = slide_ids.index(slide)
    attentions = results[ResultsKey.BAG_ATTN][slide_idx]

    # for each tile in the bag
    for tile_idx in range(len(results[ResultsKey.IMAGE_PATH][slide_idx])):
        tile_coords = np.transpose(np.array([results[ResultsKey.TILE_LEFT][slide_idx][tile_idx].cpu().numpy(),
                                             results[ResultsKey.TILE_TOP][slide_idx][tile_idx].cpu().numpy()]))
        coords_list.append(tile_coords)

    coords = np.array(coords_list)
    attentions = np.array(attentions.cpu()).reshape(-1)

    sel_coords = location_selected_tiles(tile_coords=coords, location_bbox=location_bbox, level=level)
    cmap = plt.cm.get_cmap('Reds')

    tile_xs, tile_ys = sel_coords.T
    rects = [patches.Rectangle(xy, tile_size, tile_size) for xy in zip(tile_xs, tile_ys)]

    pc = collection.PatchCollection(rects, match_original=True, cmap=cmap, alpha=.5, edgecolor='black')
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
    ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt=".2%")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
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
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
