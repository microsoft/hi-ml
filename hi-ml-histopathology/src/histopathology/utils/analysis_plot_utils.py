#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import umap
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, roc_curve, confusion_matrix

from histopathology.utils.naming import ResultsKey

TRAIN_STYLE = dict(ls='-')
VAL_STYLE = dict(ls='--')
BEST_EPOCH_LINE_STYLE = dict(ls=':', lw=1)
BEST_TRAIN_MARKER_STYLE = dict(marker='o', markeredgecolor='w', markersize=6)
BEST_VAL_MARKER_STYLE = dict(marker='*', markeredgecolor='w', markersize=11)


def get_tsne_projection(features: List[Any], n_components: int = 2, n_jobs: int = -1, **kwargs: Any) -> List[Any]:
    """
    Get the t-sne projection of high dimensional data in a lower dimensional space
    :param features: list of features in higher dimensional space (n x f for n samples and f features per sample)
    :param **kwargs: keyword arguments to be passed to TSNE()
    :return: list of features in lower dimensional space (n x c for n samples and c components)
    """
    tsne_2d = TSNE(n_components=n_components, n_jobs=n_jobs, **kwargs)
    tsne_proj = tsne_2d.fit_transform(features)
    return tsne_proj


def get_umap_projection(features: List[Any], n_components: int = 2, n_jobs: int = -1, **kwargs: Any) -> List[Any]:
    """
    Get the umap projection of high dimensional data in a lower dimensional space
    :param features: list of features in higher dimensional space (n x f for n samples and f features per sample)
    :param **kwargs: keyword arguments to be passed to UMAP()
    :return: list of features in lower dimensional space (n x c for n samples and c components)
    """
    umap_2d = umap.UMAP(n_components=n_components, n_jobs=n_jobs, **kwargs)
    umap_proj = umap_2d.fit_transform(features)
    return umap_proj


def normalize_array_minmax(arr: List[float]) -> List[float]:
    """
    Normalize an array in range 0 to 1
    :param arr: array to be normalized
    :return: normalized array
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def normalize_array_mean(arr: List[float]) -> List[float]:
    """
    Normalize an array with zero mean and unit variance
    :param arr: array to be normalized
    :return: normalized array
    """
    return (arr - np.mean(arr)) / np.std(arr)


def plot_projected_features_2d(data: Any, labels: List[int], classes: List[str], title: str = "") -> None:
    """
    Plot a scatter plot of projected features in two dimensions
    :param data: features projected in 2d space (nx2)
    :param labels: corresponding labels of the data (nx1)
    :param classes: list of classes in the dataset
    :param title: plot title string
    """
    plt.figure()
    scatter = plt.scatter(data[:, 0], data[:, 1], 20, labels)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.title(title)


def plot_box_whisker(data_list: List[Any], column_names: List[str], show_outliers: bool, title: str = "") -> None:
    """
    Plot a box whisker plot of column data
    :param columns: data to be plotted in columns
    :param column_names: names of the columns
    :param show_outliers: whether outliers need to be shown
    :param title: plot title string
    """
    plt.figure()
    _, ax = plt.subplots()
    ax.boxplot(data_list, showfliers=show_outliers)
    positions = range(1, len(column_names) + 1)
    means = []
    for i in range(len(data_list)):
        means.append(np.mean(data_list[i]))
    ax.plot(positions, means, 'rs')
    plt.xticks(positions, column_names)
    plt.title(title)


def plot_histogram(data: List[Any], title: str = "") -> None:
    """
    Plot a histogram given some data
    :param data: data to be plotted
    :param title: plot title string
    """
    plt.figure()
    plt.hist(data, bins=50)
    plt.gca().set(title=title, xlabel='Values', ylabel='Frequency')


def plot_roc_curve(labels: Sequence, scores: Sequence, label: str, ax: Axes) -> None:
    """Plot ROC curve for the given labels and scores, with AUROC in the line legend.

    :param labels: The true binary labels.
    :param scores: Scores predicted by the model.
    :param label: An line identifier to be displayed in the legend.
    :param ax: `Axes` object onto which to plot.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    label = f"{label} (AUROC: {auroc:.3f})"
    ax.plot(fpr, tpr, label=label)


def plot_pr_curve(labels: Sequence, scores: Sequence, label: str, ax: Axes) -> None:
    """Plot precision-recall curve for the given labels and scores, with AUROC in the line legend.

    :param labels: The true binary labels.
    :param scores: Scores predicted by the model.
    :param label: An line identifier to be displayed in the legend.
    :param ax: `Axes` object onto which to plot.
    """
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)
    label = f"{label} (AUPR: {aupr:.3f})"
    ax.plot(recall, precision, label=label)


def format_pr_or_roc_axes(plot_type: str, ax: Axes) -> None:
    """Format PR or ROC plot with appropriate axis labels, limits, and grid.

    :param plot_type: Either 'pr' or 'roc'.
    :param ax: `Axes` object to format.
    """
    if plot_type == 'pr':
        xlabel, ylabel = "Recall", "Precision"
    elif plot_type == 'roc':
        xlabel, ylabel = "False positive rate", "True positive rate"
    else:
        raise ValueError(f"Plot type must be either 'pr' or 'roc' (received '{plot_type}')")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect(1)
    ax.set_xlim(-.05, 1.05)
    ax.set_ylim(-.05, 1.05)
    ax.grid(color='0.9')


def _plot_crossval_roc_and_pr_curves(crossval_dfs: Dict[int, pd.DataFrame], roc_ax: Axes, pr_ax: Axes,
                                     scores_column: str = ResultsKey.PROB) -> None:
    """Plot ROC and precision-recall curves for multiple cross-validation runs onto provided axes.

    This is called by :py:func:`plot_crossval_roc_and_pr_curves()`, which additionally creates a figure and the axes.

    :param crossval_dfs: Dictionary of dataframes with cross-validation indices as keys,
        as returned by :py:func:`collect_crossval_outputs()`.
    :param roc_ax: `Axes` object onto which to plot ROC curves.
    :param pr_ax: `Axes` object onto which to plot precision-recall curves.
    """
    for k, tiles_df in crossval_dfs.items():
        slides_groupby = tiles_df.groupby(ResultsKey.SLIDE_ID)

        tile_labels = slides_groupby[ResultsKey.TRUE_LABEL]
        # True slide label is guaranteed unique
        assert all(len(unique_slide_label) == 1 for unique_slide_label in tile_labels.unique())
        labels = tile_labels.first()

        tile_scores = slides_groupby[scores_column]
        non_unique_slides = [slide_id for slide_id, unique_slide_score in tile_scores.unique().items()
                             if len(unique_slide_score) > 1]
        if non_unique_slides:
            warnings.warn(f"Found {len(non_unique_slides)}/{len(slides_groupby)} non-unique slides in fold {k}: "
                          f"{sorted(non_unique_slides)}")
        # TODO: Re-enable assertion once we can guarantee uniqueness of slides during validation
        # assert len(non_unique_slides) == 0
        scores = tile_scores.first()

        plot_roc_curve(labels, scores, label=f"Fold {k}", ax=roc_ax)
        plot_pr_curve(labels, scores, label=f"Fold {k}", ax=pr_ax)
    legend_kwargs = dict(edgecolor='none', fontsize='small')
    roc_ax.legend(**legend_kwargs)
    pr_ax.legend(**legend_kwargs)
    format_pr_or_roc_axes('roc', roc_ax)
    format_pr_or_roc_axes('pr', pr_ax)


def plot_crossval_roc_and_pr_curves(crossval_dfs: Dict[int, pd.DataFrame],
                                    scores_column: str = ResultsKey.PROB) -> Figure:
    """Plot ROC and precision-recall curves for multiple cross-validation runs.

    This will create a new figure with two subplots (left: ROC, right: PR).

    :param crossval_dfs: Dictionary of dataframes with cross-validation indices as keys,
        as returned by :py:func:`collect_crossval_outputs()`.
    :return: The created `Figure` object.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    _plot_crossval_roc_and_pr_curves(crossval_dfs, scores_column=scores_column, roc_ax=axs[0], pr_ax=axs[1])
    return fig


def plot_crossval_training_curves(metrics_df: pd.DataFrame, train_metric: str, val_metric: str, ax: Axes,
                                  best_epochs: Optional[Dict[int, int]] = None, ylabel: Optional[str] = None) -> None:
    """Plot paired training and validation metrics for every training epoch of cross-validation runs.

    :param metrics_df: Metrics dataframe, as returned by :py:func:`collect_crossval_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :param train_metric: Name of the training metric to plot.
    :param val_metric: Name of the validation metric to plot.
    :param ax: `Axes` object onto which to plot.
    :param best_epochs: If provided, adds visual indicators of the best epoch for each run.
    :param ylabel: If provided, adds a label to the Y-axis.
    """
    for k in sorted(metrics_df.columns):
        train_values = metrics_df.loc[train_metric, k]
        val_values = metrics_df.loc[val_metric, k]
        line, = ax.plot(train_values, **TRAIN_STYLE, label=f"Fold {k}")
        color = line.get_color()
        ax.plot(val_values, color=color, **VAL_STYLE)
        if best_epochs is not None:
            best_epoch = best_epochs[k]
            ax.plot(best_epoch, train_values[best_epoch], color=color, zorder=1000, **BEST_TRAIN_MARKER_STYLE)
            ax.plot(best_epoch, val_values[best_epoch], color=color, zorder=1000, **BEST_VAL_MARKER_STYLE)
            ax.axvline(best_epoch, color=color, **BEST_EPOCH_LINE_STYLE)
    ax.grid(color='0.9')
    ax.set_xlabel("Epoch")
    if ylabel:
        ax.set_ylabel(ylabel)


def add_training_curves_legend(fig: Figure, include_best_epoch: bool = False) -> None:
    """Add a legend to a training curves figure, indicating cross-validation indices and train/val.

    :param fig: `Figure` object onto which to add the legend.
    :param include_best_epoch: If `True`, adds legend items for the best epoch indicators from
        :py:func:`plot_crossval_training_curves()`.
    """
    legend_kwargs = dict(edgecolor='none', fontsize='small', borderpad=.2)

    # Add primary legend for main lines (crossval folds)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), **legend_kwargs, loc='lower center',
               bbox_to_anchor=(0.5, -0.06), ncol=len(by_label))

    # Add secondary legend for line styles
    legend_handles = [Line2D([], [], **TRAIN_STYLE, color='k', label="Training"),
                      Line2D([], [], **VAL_STYLE, color='k', label="Validation")]
    if include_best_epoch:
        legend_handles.append(Line2D([], [], **BEST_EPOCH_LINE_STYLE, **BEST_TRAIN_MARKER_STYLE,
                                     color='k', label="Best epoch (train)"),)
        legend_handles.append(Line2D([], [], **BEST_EPOCH_LINE_STYLE, **BEST_VAL_MARKER_STYLE,
                                     color='k', label="Best epoch (val.)"),)
    fig.legend(handles=legend_handles, **legend_kwargs, loc='lower center',
               bbox_to_anchor=(0.5, -0.1), ncol=len(legend_handles))


def plot_confusion_matrices(crossval_dfs: Dict[int, pd.DataFrame], class_names: List[str]) -> Figure:
    import seaborn as sns
    fig, axs = plt.subplots(1, len(crossval_dfs), figsize=(len(crossval_dfs)*6, 5))
    for k, tiles_df in crossval_dfs.items():
        slides_groupby = tiles_df.groupby(ResultsKey.SLIDE_ID)
        tile_labels_true = slides_groupby[ResultsKey.TRUE_LABEL]
        # True slide label is guaranteed unique
        assert all(len(unique_slide_label) == 1 for unique_slide_label in tile_labels_true.unique())
        labels_true = tile_labels_true.first()

        tile_labels_pred = slides_groupby[ResultsKey.PRED_LABEL]
        labels_pred = tile_labels_pred.first()

        cf_matrix = confusion_matrix(y_true=labels_true, y_pred=labels_pred)
        cf_matrix_n = cf_matrix / cf_matrix.sum(axis=1, keepdims=True)
        sns.heatmap(cf_matrix_n, annot=True, cmap='Blues', fmt=".2%", ax=axs[k])
        axs[k].set_xlabel('Predicted')
        axs[k].set_ylabel('True')
        axs[k].xaxis.set_ticklabels(class_names)
        axs[k].yaxis.set_ticklabels(class_names)
        axs[k].set_title(f'Fold {k}')
    return fig
