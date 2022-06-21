from pathlib import Path
from typing import Union, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from health_multimodal.image.data.io import load_image


TypeArrayImage = Union[np.ndarray, Image.Image]
TypeOptionalAxis = Optional[plt.Axes]
TypeOptionalString = Optional[str]


def plot_image(
    image: TypeArrayImage,
    axis: plt.Axes,
    title: Optional[str] = None,
) -> None:
    """
    """
    axis.imshow(image)
    axis.axis('off')
    if title is not None:
        axis.set_title(title)


def plot_isolines(
        image: TypeArrayImage,
        similarity: np.ndarray,
        axis: plt.Axes,
        title: Optional[str] = None,
        colormap: str = 'RdBu_r',
        step: float = 0.25,
) -> None:
    """
    """
    axis.imshow(image)
    num_steps = int(round(1 / step))
    levels = np.linspace(step, 1, num_steps)
    contours = axis.contour(
        similarity,
        cmap=colormap,
        vmin=-1,
        vmax=1,
        levels=levels,
    )
    axis.clabel(contours, inline=True, fontsize=10)
    axis.axis('off')
    if title is not None:
        axis.set_title(title)


def plot_heatmap(
        image: TypeArrayImage,
        similarity: np.ndarray,
        figure: plt.Figure,
        axis: plt.Axes,
        colormap: str = 'RdBu_r',
        title: Optional[str] = None,
        alpha: float = 0.5,
) -> None:
    """
    """
    axis.imshow(image)
    axes_image = axis.matshow(similarity, alpha=alpha, cmap=colormap, vmin=-1, vmax=1)
    # https://www.geeksforgeeks.org/how-to-change-matplotlib-color-bar-size-in-python/
    divider = make_axes_locatable(axis)
    colorbar_axes = divider.append_axes('right', size='10%', pad=0.1)
    colorbar = figure.colorbar(axes_image, cax=colorbar_axes)
    # https://stackoverflow.com/a/50671487/3956024
    colorbar.ax.tick_params(pad=35)
    plt.setp(colorbar.ax.get_yticklabels(), ha='right')
    axis.axis('off')
    if title is not None:
        axis.set_title(title)


def plot_phrase_grounding_similarity_map(image_path: Path, similarity_map: np.ndarray) -> None:
    """
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    image = load_image(image_path).convert('RGB')
    plot_image(image, axis=axes[0], title='Input image')
    plot_isolines(image, similarity_map, axis=axes[1], title='Similarity isolines')
    plot_heatmap(image, similarity_map, figure=fig, axis=axes[2], title='Similarity heatmap')
