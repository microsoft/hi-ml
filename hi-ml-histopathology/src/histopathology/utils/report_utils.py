#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from azureml.core import Experiment, Run, Workspace
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from health_azure.utils import aggregate_hyperdrive_metrics, get_aml_run_from_run_id, get_tags_from_hyperdrive_run
from histopathology.utils.naming import ResultsKey

TRAIN_STYLE = dict(ls='-')
VAL_STYLE = dict(ls='--')
BEST_EPOCH_LINE_STYLE = dict(ls=':', lw=1)
BEST_TRAIN_MARKER_STYLE = dict(marker='o', markeredgecolor='w', markersize=6)
BEST_VAL_MARKER_STYLE = dict(marker='*', markeredgecolor='w', markersize=11)


def download_from_run_if_necessary(run: Run, remote_dir: Path, download_dir: Path, filename: str) -> Path:
    """Download any file from an AML run if it doesn't exist locally.

    :param run: AML Run object.
    :param remote_dir: Remote directory from where the file is downloaded.
    :param download_dir: Local directory where to save the downloaded file.
    :param filename: Name of the file to be downloaded (e.g. `"test_output.csv"`).
    :return: Local path to the downloaded file.
    """
    local_path = download_dir / filename
    remote_path = remote_dir / filename
    if local_path.exists():
        print("File already exists at", local_path)
    else:
        local_path.parent.mkdir(exist_ok=True, parents=True)
        run.download_file(str(remote_path), str(local_path), _validate_checksum=True)
        assert local_path.exists()
        print("File is downloaded at", local_path)
    return local_path


def collect_crossval_outputs(parent_run_id: str, download_dir: Path, aml_workspace: Workspace,
                             crossval_arg_name: str = "cross_validation_split_index",
                             output_filename: str = "test_output.csv") -> Dict[int, pd.DataFrame]:
    """Fetch output CSV files from cross-validation runs as dataframes.

    Will only download the CSV files if they do not already exist locally.

    :param parent_run_id: Azure ML run ID for the parent Hyperdrive run.
    :param download_dir: Base directory where to download the CSV files. A new sub-directory will
        be created for each child run (e.g. `<download_dir>/<crossval index>/*.csv`).
    :param aml_workspace: Azure ML workspace in which the runs were executed.
    :param crossval_arg_name: Name of the Hyperdrive argument used for indexing the child runs.
    :param output_filename: Filename of the output CSVs to download.
    :return: A dictionary of dataframes with the sorted cross-validation indices as keys.
    """
    parent_run = get_aml_run_from_run_id(parent_run_id, aml_workspace)

    all_outputs_dfs = {}
    for child_run in parent_run.get_children():
        child_run_index = get_tags_from_hyperdrive_run(child_run, crossval_arg_name)
        if child_run_index is None:
            raise ValueError(f"Child run expected to have the tag '{crossval_arg_name}'")
        child_dir = download_dir / str(child_run_index)
        try:
            child_csv = download_from_run_if_necessary(child_run,
                                                       remote_dir=Path("outputs"),
                                                       download_dir=child_dir,
                                                       filename=output_filename)
            all_outputs_dfs[child_run_index] = pd.read_csv(child_csv)
        except Exception as e:
            print(f"Failed to download {output_filename} for run {child_run.id}: {e}")
    return dict(sorted(all_outputs_dfs.items()))


def collect_crossval_metrics(parent_run_id: str, download_dir: Path, aml_workspace: Workspace,
                             crossval_arg_name: str = "cross_validation_split_index") -> pd.DataFrame:
    """Fetch metrics logged to Azure ML from cross-validation runs as a dataframe.

    Will only download the metrics if they do not already exist locally, as this can take several
    seconds for each child run.

    :param parent_run_id: Azure ML run ID for the parent Hyperdrive run.
    :param download_dir: Directory where to save the downloaded metrics as `aml_metrics.pickle`.
    :param aml_workspace: Azure ML workspace in which the runs were executed.
    :param crossval_arg_name: Name of the Hyperdrive argument used for indexing the child runs.
    :return: A dataframe in the format returned by :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    """
    # Save metrics as a pickle because complex dataframe structure is lost in CSV
    metrics_pickle = download_dir / "aml_metrics.pickle"
    if metrics_pickle.is_file():
        print(f"AML metrics file already exists at {metrics_pickle}")
        with open(metrics_pickle, 'rb') as f:
            metrics_df = pickle.load(f)
    else:
        metrics_df = aggregate_hyperdrive_metrics(run_id=parent_run_id,
                                                  child_run_arg_name=crossval_arg_name,
                                                  aml_workspace=aml_workspace)
        metrics_pickle.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing AML metrics file to {metrics_pickle}")
        with open(metrics_pickle, 'wb') as f:
            pickle.dump(metrics_df, f)
    return metrics_df.sort_index(axis='columns')


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


def _plot_crossval_roc_and_pr_curves(crossval_dfs: Dict[int, pd.DataFrame],
                                     roc_ax: Axes, pr_ax: Axes) -> None:
    for k, tiles_df in crossval_dfs.items():
        slides_groupby = tiles_df.groupby(ResultsKey.SLIDE_ID)
        labels = slides_groupby[ResultsKey.TRUE_LABEL].agg(pd.Series.mode)
        scores = slides_groupby[ResultsKey.PROB].agg(pd.Series.mode)
        plot_roc_curve(labels, scores, label=f"Fold {k}", ax=roc_ax)
        plot_pr_curve(labels, scores, label=f"Fold {k}", ax=pr_ax)
    legend_kwargs = dict(edgecolor='none', fontsize='small')
    roc_ax.legend(**legend_kwargs)
    pr_ax.legend(**legend_kwargs)
    format_pr_or_roc_axes('roc', roc_ax)
    format_pr_or_roc_axes('pr', pr_ax)


def plot_crossval_roc_and_pr_curves(crossval_dfs: Dict[int, pd.DataFrame]) -> Figure:
    """Plot ROC and precision-recall curves for multiple cross-validation runs.

    This will create a new figure with two subplots (left: ROC, right: PR).

    :param crossval_dfs: Dictionary of dataframes with cross-validation indices as keys,
        as returned by :py:func:`collect_crossval_outputs()`.
    :return: The created `Figure` object.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    _plot_crossval_roc_and_pr_curves(crossval_dfs, roc_ax=axs[0], pr_ax=axs[1])
    return fig


def get_crossval_metrics_table(metrics_df: pd.DataFrame,
                               metrics_list: Sequence[str]) -> pd.DataFrame:
    """Format raw cross-validation metrics into a table with a summary "Mean ± Std" column.

    Note that this function only supports scalar metrics. To format metrics that are logged
    hroughout training, you should call :py:func:`get_best_epoch_metrics()` first.

    :param metrics_df: Metrics dataframe, as returned by :py:func:`collect_crossval_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :param metrics_list: The list of metrics to include in the table.
    :return: A dataframe with the values of the selected metrics formatted as strings, including a
        header and a summary column.
    """
    header = ["Metric"] + [f"Split {k}" for k in metrics_df.columns] + ["Mean ± Std"]
    metrics_rows = []
    for metric in metrics_list:
        values: pd.Series = metrics_df.loc[metric]
        mean = values.mean()
        std = values.std()
        row = [metric] + [f"{v:.3f}" for v in values] + [f"{mean:.3f} ± {std:.3f}"]
        metrics_rows.append(row)
    table = pd.DataFrame(metrics_rows, columns=header).set_index(header[0])
    return table


def get_best_epochs(metrics_df: pd.DataFrame, primary_metric: str, maximise: bool = True) -> Dict[int, int]:
    """Determine the best epoch for each cross-validation run based on a given metric.

    The returned epoch indices are relative to the logging frequency of the chosen metric, i.e.
    should not be mixed between training and validation metrics if the latter are not logged at
    every training epoch.

    :param metrics_df: Metrics dataframe, as returned by :py:func:`collect_crossval_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :param primary_metric: Name of the reference metric to optimise.
    :param maximise: Whether the given metric should be maximised (minimised if `False`).
    :return: Dictionary mapping each cross-validation index to its best epoch.
    """
    best_fn = np.argmax if maximise else np.argmin
    best_epochs = metrics_df.loc[primary_metric].apply(best_fn)
    return best_epochs.to_dict()


def get_best_epoch_metrics(metrics_df: pd.DataFrame, metrics_list: Sequence[str],
                           best_epochs: Dict[int, int]) -> pd.DataFrame:
    """Extract the values of the selected cross-validation metrics at the given best epochs.

    :param metrics_df: Metrics dataframe, as returned by :py:func:`collect_crossval_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :param metrics_list: Names of the metrics to index by the best epoch indices provided. Their
        values in `metrics_df` should be lists.
    :param best_epochs: Dictionary of cross-validation indices to best epochs, as returned by
        :py:func:`get_best_epochs()`.
    :return: Dataframe with the same columns as `metrics_df` and rows specified by `metrics_list`,
        containing only scalar values.
    """
    best_metrics = [metrics_df.loc[metrics_list, k].apply(lambda values: values[epoch])
                    for k, epoch in best_epochs.items()]
    best_metrics_df = pd.DataFrame(best_metrics).T
    return best_metrics_df


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


def get_formatted_run_info(parent_run: Run) -> str:
    """Format Azure ML cross-validation run information as HTML.

    Includes details of the parent and child runs, as well as submission information.

    :param parent_run: Parent Hyperdrive Azure ML run object.
    :return: Formatted HTML string.
    """
    def format_experiment(experiment: Experiment) -> str:
        return f"<a href={experiment.get_portal_url()}>{experiment.name}</a>"

    def format_run(run: Run) -> str:
        return f"<a href={run.get_portal_url()}>{run.display_name}</a> ({run.id}, {run.get_status()})"

    def format_submission_info(run: Run) -> str:
        details = run.get_details()
        start_time = dateutil.parser.parse(details['startTimeUtc'])
        return f"Started on {start_time.strftime('%d %b %Y %H:%M %Z')} by {details['submittedBy']}"

    html = f"<p>Experiment: {format_experiment(parent_run.experiment)}"
    html += f"\n<br>Parent run: {format_run(parent_run)}"

    html += "\n<ul>"
    for k, child_run in enumerate(sorted(parent_run.get_children(), key=lambda r: r.id)):
        html += f"\n<li>Child {k}: {format_run(child_run)}</li>"
    html += "\n</ul>"

    html += f"\n<p>{format_submission_info(parent_run)}"
    html += f"\n<p>Command-line arguments: <code>{parent_run.get_tags()['commandline_args']}</code>"
    return html
