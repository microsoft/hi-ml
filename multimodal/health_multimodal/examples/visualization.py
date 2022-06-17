import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_image(image, axis=None, title=None):
    axis.imshow(image)
    axis.axis('off')
    if title is not None:
        axis.set_title(title)


def plot_isolines(image, similarity, axis=None, title=None, colormap='RdBu_r', step=0.25):
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


def plot_heatmap(image, similarity, figure=None, axis=None, colormap='RdBu_r', title=None, alpha=0.5):
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
