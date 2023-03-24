#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from health_multimodal.image.data.io import load_image


TypeArrayImage = Union[np.ndarray, Image.Image]


def _plot_image(
    image: TypeArrayImage,
    axis: plt.Axes,
    title: Optional[str] = None,
) -> None:
    """Plot an image on a given axis, deleting the axis ticks and axis labels.

    :param image: Input image.
    :param axis: Axis to plot the image on.
    :param title: Title used for the axis.
    """
    axis.imshow(image)
    axis.axis("off")
    if title is not None:
        axis.set_title(title)


def _get_isolines_levels(step_size: float) -> np.ndarray:
    num_steps = np.floor(round(1 / step_size)).astype(int)
    levels = np.linspace(step_size, 1, num_steps)
    return levels


def _plot_isolines(
    image: TypeArrayImage,
    heatmap: np.ndarray,
    axis: plt.Axes,
    title: Optional[str] = None,
    colormap: str = "RdBu_r",
    step: float = 0.25,
) -> None:
    """Plot an image and overlay heatmap isolines on it.

    :param image: Input image.
    :param heatmap: Heatmap of the same size as the image.
    :param axis: Axis to plot the image on.
    :param title: Title used for the axis.
    :param colormap: Name of the Matplotlib colormap used for the isolines.
    :param step: Step size between the isolines levels. The levels are in :math:`(0, 1]`.
        For example, a step size of 0.25 will result in isolines levels of 0.25, 0.5, 0.75 and 1.
    """
    axis.imshow(image)
    levels = _get_isolines_levels(step)
    contours = axis.contour(
        heatmap,
        cmap=colormap,
        vmin=-1,
        vmax=1,
        levels=levels,
    )
    axis.clabel(contours, inline=True, fontsize=10)
    axis.axis("off")
    if title is not None:
        axis.set_title(title)


def _plot_heatmap(
    image: TypeArrayImage,
    heatmap: np.ndarray,
    figure: plt.Figure,
    axis: plt.Axes,
    colormap: str = "RdBu_r",
    title: Optional[str] = None,
    alpha: float = 0.5,
) -> None:
    """Plot a heatmap overlaid on an image.

    :param image: Input image.
    :param heatmap: Input heatmap of the same size as the image.
    :param figure: Figure to plot the images on.
    :param axis: Axis to plot the images on.
    :param colormap: Name of the Matplotlib colormap for the heatmap.
    :param title: Title used for the axis.
    :param alpha: Heatmap opacity. Must be in :math:`[0, 1]`.
    """
    axis.imshow(image)
    axes_image = axis.matshow(heatmap, alpha=alpha, cmap=colormap, vmin=-1, vmax=1)
    # https://www.geeksforgeeks.org/how-to-change-matplotlib-color-bar-size-in-python/
    divider = make_axes_locatable(axis)
    colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)
    colorbar = figure.colorbar(axes_image, cax=colorbar_axes)
    # https://stackoverflow.com/a/50671487/3956024
    colorbar.ax.tick_params(pad=35)
    plt.setp(colorbar.ax.get_yticklabels(), ha="right")
    axis.axis("off")
    if title is not None:
        axis.set_title(title)


def plot_phrase_grounding_similarity_map(
    image_path: Path, similarity_map: np.ndarray, bboxes: Optional[List[Tuple[float, float, float, float]]] = None
) -> plt.Figure:
    """Plot visualization of the input image, the similarity heatmap and the heatmap isolines.

    :param image_path: Path to the input image.
    :param similarity_map: Phase grounding similarity map of the same size as the image.
    :param bboxes: Optional list of bounding boxes to plot on the image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    image = load_image(image_path).convert("RGB")
    _plot_image(image, axis=axes[0], title="Input image")
    _plot_isolines(image, similarity_map, axis=axes[1], title="Similarity isolines")
    _plot_heatmap(image, similarity_map, figure=fig, axis=axes[2], title="Similarity heatmap")
    if bboxes is not None:
        _plot_bounding_boxes(ax=axes[1], bboxes=bboxes)
    return fig


def _plot_bounding_boxes(
    ax: plt.Axes, bboxes: List[Tuple[float, float, float, float]], linewidth: float = 1.5, alpha: float = 0.45
) -> None:
    """
    Plot bounding boxes on an existing axes object.

    :param ax: The axes object to plot the bounding boxes on.
    :param bboxes: A list of bounding box coordinates as (x, y, width, height) tuples.
    :param linewidth: Optional line width for the bounding box edges (default is 2).
    :param alpha: Optional opacity for the bounding box edges (default is 1.0).
    """
    for bbox in bboxes:
        x, y, width, height = bbox
        rect = patches.Rectangle(
            (x, y), width, height, linewidth=linewidth, edgecolor='k', facecolor='none', linestyle='--', alpha=alpha
        )
        ax.add_patch(rect)
