#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collection

from math import ceil
from pathlib import Path
from typing import Sequence, List, Any, Dict, Optional, Union, Tuple

from monai.data.dataset import Dataset
from monai.data.image_reader import WSIReader
from monai.transforms.io.dictionary import LoadImaged
from torch.utils.data import DataLoader

from health_cpath.utils.naming import SlideKey
from health_cpath.utils.naming import ResultsKey
from health_cpath.utils.heatmap_utils import location_selected_tiles
from health_cpath.utils.tiles_selection_utils import SlideNode
from health_cpath.datasets.panda_dataset import PandaDataset, LoadPandaROId


def load_image_dict(sample: dict, level: int, margin: int, wsi_has_mask: bool = True) -> Dict[SlideKey, Any]:
    """
    Load image from metadata dictionary
    :param sample: dict describing image metadata. Example:
        {'image_id': ['1ca999adbbc948e69783686e5b5414e4'],
        'image': ['/tmp/datasets/PANDA/train_images/1ca999adbbc948e69783686e5b5414e4.tiff'],
         'mask': ['/tmp/datasets/PANDA/train_label_masks/1ca999adbbc948e69783686e5b5414e4_mask.tiff'],
         'data_provider': ['karolinska'],
         'isup_grade': tensor([0]),
         'gleason_score': ['0+0']}
    :param level: level of resolution to be loaded
    :param margin: margin to be included
    :return: a dict containing the image data and metadata
    """
    loader: Union[LoadImaged, LoadPandaROId]
    if wsi_has_mask:
        loader = LoadPandaROId(WSIReader("cuCIM"), level=level, margin=margin)
    else:
        loader = LoadImaged(keys=[SlideKey.IMAGE], reader=WSIReader("cuCIM"), dtype=np.uint8, level=level,
                            image_only=True)
    img = loader(sample)
    return img


def plot_panda_data_sample(
    panda_dir: str, nsamples: int, ncols: int, level: int, margin: int, title_key: str = "data_provider"
) -> None:
    """
    :param panda_dir: path to the dataset, it's expected a file called "train.csv" exists at the path.
        Look at the PandaDataset for more detail
    :param nsamples: number of random samples to be visualized
    :param ncols: number of columns in the figure grid. Nrows is automatically inferred
    :param level: level of resolution to be loaded
    :param margin: margin to be included
    :param title_key: metadata key in image_dict used to label each subplot
    """
    panda_dataset = Dataset(PandaDataset(root=panda_dir))[:nsamples]  # type: ignore
    loader = DataLoader(panda_dataset, batch_size=1)

    nrows = math.ceil(nsamples / ncols)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(9, 9))

    for dict_images, ax in zip(loader, axes.flat):
        slide_id = dict_images[SlideKey.SLIDE_ID]
        title = dict_images[SlideKey.METADATA][title_key]
        print(f">>> Slide {slide_id}")
        img = load_image_dict(dict_images, level=level, margin=margin)
        ax.imshow(img[SlideKey.IMAGE].transpose(1, 2, 0))
        ax.set_title(title)
    fig.tight_layout()


def plot_scores_hist(
    results: Dict, prob_col: str = ResultsKey.CLASS_PROBS, gt_col: str = ResultsKey.TRUE_LABEL
) -> plt.Figure:
    """Plot scores as a histogram.

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


def _get_histo_plot_title(case: str, slide_node: SlideNode) -> str:
    """Return the standard title for histopathology plots.

    :param case: case id e.g., TP, FN, FP, TN
    :param slide_node: SlideNode object that encapsulates the slide information
    """
    return (
        f"{case}: {slide_node.slide_id} P={slide_node.pred_prob_score:.2f} \n Predicted label: {slide_node.pred_label} "
        f"True label: {slide_node.true_label}"
    )


def plot_attention_tiles(
    case: str, slide_node: SlideNode, top: bool, num_columns: int, figsize: Tuple[int, int]
) -> Optional[plt.Figure]:
    """Plot top or bottom tiles figures along with their attention scores.

    :param case: The report case (e.g., TP, FN, ...)
    :param slide_node: The slide node for which we would like to plot attention tiles.
    :param top: Decides which tiles to plot. If True, plots top tile nodes of the slide_node. Otherwise, plots bottom
        tile nodes.
    :param num_columns: Number of columns to create the subfigures grid, defaults to 4
    :param figsize: The figure size of tiles attention plots, defaults to (10, 10)
    """
    tile_nodes = slide_node.top_tiles if top else slide_node.bottom_tiles
    num_rows = int(ceil(len(tile_nodes) / num_columns))
    if num_rows == 0:
        logging.warning(
            "The number of selected top and bottom tiles is too low, plotting will be skipped."
            "Try debugging with a higher num_top_tiles and/or a higher number of batches.")
        return None

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=figsize)
    fig.suptitle(_get_histo_plot_title(case, slide_node))
    for ax, tile_node in zip(axs.flat, tile_nodes):
        ax.imshow(np.transpose(tile_node.data.numpy(), (1, 2, 0)), clim=(0, 255), cmap="gray")
        ax.set_title("%.6f" % tile_node.attn)

    for ax in axs.flat:
        ax.set_axis_off()
    return fig


def plot_slide(case: str, slide_node: SlideNode, slide_image: np.ndarray, scale: float) -> plt.Figure:
    """Plots a slide thumbnail from a given slide image and scale.

    :param case: The report case (e.g., TP, FN, ...)
    :param slide_node: The slide node for which we would like to plot attention tiles.
    :param slide_image: Numpy array of the slide image (shape: [3, H, W]).
    :return: matplotlib figure of the slide thumbnail.
    """
    fig, ax = plt.subplots()
    slide_image = slide_image.transpose(1, 2, 0)
    ax.imshow(slide_image)
    fig.suptitle(_get_histo_plot_title(case, slide_node))
    ax.set_axis_off()
    original_size = fig.get_size_inches()
    fig.set_size_inches((original_size[0] * scale, original_size[1] * scale))
    return fig


def plot_heatmap_overlay(
    case: str,
    slide_node: SlideNode,
    slide_image: np.ndarray,
    results: Dict[ResultsKey, List[Any]],
    location_bbox: List[int],
    tile_size: int = 224,
    level: int = 1,
) -> plt.Figure:
    """Plots heatmap of selected tiles (e.g. tiles in a bag) overlay on the corresponding slide.

    :param case: The report case (e.g., TP, FN, ...)
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
    fig.suptitle(_get_histo_plot_title(case, slide_node))

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


def save_figure(fig: Optional[plt.figure], figpath: Path) -> None:
    """Save a matplotlib figure in a given figpath.

    :param fig: The figure to be saved.
    :param figpath: The filename where to save the figure.
    """
    if fig is None:
        return
    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)
