from pathlib import Path
from typing import List, Sequence

import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from azureml.core import Experiment, Run, Workspace
from health_azure import download_files_from_run_id, get_workspace
from health_azure.utils import get_aml_run_from_run_id
from histopathology.utils.naming import ResultsKey
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def download_file_if_necessary(run_id: str, remote_dir: Path, download_dir: Path, filename: str) -> None:
    """
    Function to download any file from an AML run if it doesn't exist locally
    :param run_id: run ID of the AML run
    :param remote_dir: remote directory from where the file is downloaded
    :param download_dir: local directory where to save the downloaded file
    :param filename: name of the file to be downloaded (e.g. `"test_output.csv"`).
    """
    local_path = download_dir / run_id / "outputs" / filename
    remote_path = remote_dir / filename
    if local_path.exists():
        print("File already exists at", local_path)
    else:
        aml_workspace = get_workspace()
        local_dir = local_path.parent.parent
        local_dir.mkdir(exist_ok=True, parents=True)
        download_files_from_run_id(run_id=run_id,
                                   output_folder=local_dir,
                                   prefix=str(remote_path),
                                   workspace=aml_workspace,
                                   validate_checksum=True)
        assert local_path.exists()
        print("File is downloaded at", local_path)


def collect_crossval_outputs(parent_run_id: str, download_dir: Path,
                             num_splits: int) -> List[pd.DataFrame]:
    output_filename = "test_output.csv"

    all_outputs_dfs = []
    for i in range(num_splits):
        child_run_id = f"{parent_run_id}_{i}"
        download_file_if_necessary(run_id=child_run_id,
                                   remote_dir=Path("outputs"),
                                   download_dir=download_dir,
                                   filename=output_filename)

        child_outputs_df = pd.read_csv(download_dir / child_run_id / "outputs" / output_filename)
        all_outputs_dfs.append(child_outputs_df)

    return all_outputs_dfs


def plot_roc_curve(labels: Sequence, scores: Sequence, label: str, ax: Axes) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    label = f"{label} (auROC: {auroc:.3f})"
    ax.plot(fpr, tpr, label=label)


def plot_pr_curve(labels: Sequence, scores: Sequence, label: str, ax: Axes) -> None:
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)
    label = f"{label} (auPR: {aupr:.3f})"
    ax.plot(recall, precision, label=label)


def format_pr_or_roc_axes(plot_type: str, ax: Axes) -> None:
    """
    Format PR or ROC plot with appropriate title, axis labels, limits, and grid.
    :param plot_type: Either 'pr' or 'roc'.
    :param ax: Axes object to format.
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
    ax.grid(lw=1, color='lightgray')


def _plot_crossval_roc_and_pr_curves(crossval_dfs: Sequence[pd.DataFrame],
                                     roc_ax: Axes, pr_ax: Axes) -> None:
    for k, tiles_df in enumerate(crossval_dfs):
        slides_groupby = tiles_df.groupby(ResultsKey.SLIDE_ID)
        labels = slides_groupby[ResultsKey.TRUE_LABEL].agg(pd.Series.mode)
        scores = slides_groupby[ResultsKey.PROB].agg(pd.Series.mode)
        plot_roc_curve(labels, scores, label=f"Fold {k}", ax=roc_ax)
        plot_pr_curve(labels, scores, label=f"Fold {k}", ax=pr_ax)
    legend_kwargs = dict(edgecolor='none')
    roc_ax.legend(**legend_kwargs)
    pr_ax.legend(**legend_kwargs)
    format_pr_or_roc_axes('roc', roc_ax)
    format_pr_or_roc_axes('pr', pr_ax)


def plot_crossval_roc_and_pr_curves(crossval_dfs: Sequence[pd.DataFrame]) -> Figure:
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    _plot_crossval_roc_and_pr_curves(crossval_dfs, roc_ax=axs[0], pr_ax=axs[1])
    return fig


def get_crossval_metrics_table(metrics_df: pd.DataFrame,
                               metrics_list: Sequence[str]) -> pd.DataFrame:
    num_splits = len(metrics_df.columns)
    header = ["Metric"] + [f"Split {k}" for k in range(num_splits)] + ["Mean ± Std"]
    metrics_rows = []
    for metric in metrics_list:
        values: pd.Series = metrics_df.loc[metric]
        mean = values.mean()
        std = values.std()
        row = [metric] + [f"{v:.3f}" for v in values] + [f"{mean:.3f} ± {std:.3f}"]
        metrics_rows.append(row)
    table = pd.DataFrame(metrics_rows, columns=header)
    return table


def get_best_epoch_metrics(metrics_df: pd.DataFrame, primary_metric: str,
                           metrics_list: Sequence[str], maximise: bool = True) -> pd.DataFrame:
    best_fn = np.argmax if maximise else np.argmin
    best_epochs = metrics_df.loc[primary_metric].apply(best_fn)
    best_metrics = [metrics_df.loc[metrics_list, k].apply(lambda values: values[epoch])
                    for k, epoch in enumerate(best_epochs)]
    return pd.DataFrame(best_metrics).T


def plot_crossval_training_curves(metrics_df: pd.DataFrame, metric: str, ax: Axes) -> None:
    for k in sorted(metrics_df.columns):
        values = metrics_df.loc[metric, k]
        ax.plot(values)


def get_formatted_run_info(parent_run_id: str, aml_workspace: Workspace) -> str:
    run = get_aml_run_from_run_id(parent_run_id, aml_workspace=aml_workspace)

    def format_experiment(experiment: Experiment) -> str:
        return f"<a href={experiment.get_portal_url()}>{experiment.name}</a>"

    def format_run(run: Run) -> str:
        return f"<a href={run.get_portal_url()}>{run.display_name}</a> ({run.id})"

    def format_submission_info(run: Run) -> str:
        details = run.get_details()
        start_time = dateutil.parser.parse(details['startTimeUtc'])
        return f"Started on {start_time.strftime('%d %b %Y %H:%M %Z')} by {details['submittedBy']}"

    html = f"<p>Experiment: {format_experiment(run.experiment)}"
    html += f"\n<br>Parent run: {format_run(run)}"

    html += "\n<ul>"
    for k, child_run in enumerate(sorted(run.get_children(), key=lambda r: r.id)):
        html += f"\n<li>Child {k}: {format_run(child_run)}</li>"
    html += "\n</ul>"

    html += f"\n<p>{format_submission_info(run)}"
    html += f"\n<p>Command-line arguments: <code>{run.get_tags()['commandline_args']}</code>"
    return html
