#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import dateutil.parser
import numpy as np
import pandas as pd
from azureml.core import Experiment, Run, Workspace

from health_ml.utils.common_utils import df_to_json
from health_azure.utils import (aggregate_hyperdrive_metrics, download_file_if_necessary, get_aml_run_from_run_id,
                                get_tags_from_hyperdrive_run)
from health_cpath.utils.output_utils import (AML_LEGACY_TEST_OUTPUTS_CSV, AML_TEST_OUTPUTS_CSV,
                                             AML_VAL_OUTPUTS_CSV, validate_class_names)
from health_cpath.utils.naming import AMLMetricsJsonKey


def run_has_val_and_test_outputs(run: Run) -> bool:
    """Checks whether the given run has both validation and test outputs files.

    :param parent_run: The run whose outputs to check.
    :raises ValueError: If the run does not have the expected output file(s).
    :return: `True` if the run has validation and test outputs, `False` if it is a legacy run with
        only test outputs.
    """
    available_files: List[str] = run.get_file_names()

    if AML_VAL_OUTPUTS_CSV in available_files and AML_TEST_OUTPUTS_CSV in available_files:
        return True
    elif AML_LEGACY_TEST_OUTPUTS_CSV in available_files:
        return False
    else:
        raise ValueError(f"Run {run.display_name} ({run.id}) does not have the expected files "
                         f"({AML_LEGACY_TEST_OUTPUTS_CSV} or both {AML_VAL_OUTPUTS_CSV} and "
                         f"{AML_TEST_OUTPUTS_CSV}): {available_files}")


def child_runs_have_val_and_test_outputs(parent_run: Run) -> bool:
    """Checks whether all child hyperdrive runs have both validation and test outputs files.

    :param parent_run: The parent Hyperdrive run.
    :raises ValueError: If any of the child runs does not have the expected output files, or if
        some of the child runs have both outputs and some have only test outputs.
    :return: `True` if all children have validation and test outputs, `False` if all children are
        legacy runs with only test outputs.
    """
    have_val_and_test_outputs = [run_has_val_and_test_outputs(child_run) for child_run in parent_run.get_children()]
    if all(have_val_and_test_outputs):
        return True
    elif not any(have_val_and_test_outputs):
        return False
    else:
        raise ValueError(f"Parent run {parent_run.display_name} ({parent_run.id}) has mixed children with legacy "
                         "test-only outputs and with both validation and test outputs")


def collect_hyperdrive_outputs(parent_run_id: str, download_dir: Path, aml_workspace: Workspace,
                               hyperdrive_arg_name: str = "crossval_index",
                               output_filename: str = "test_output.csv",
                               overwrite: bool = False) -> Dict[int, pd.DataFrame]:
    """Fetch output CSV files from Hyperdrive child runs as dataframes.

    Will only download the CSV files if they do not already exist locally.

    :param parent_run_id: Azure ML run ID for the parent Hyperdrive run.
    :param download_dir: Base directory where to download the CSV files. A new sub-directory will
        be created for each child run (e.g. `<download_dir>/<hyperdrive_arg_name>/*.csv`).
    :param aml_workspace: Azure ML workspace in which the runs were executed.
    :param hyperdrive_arg_name: Name of the Hyperdrive argument used for indexing the child runs.
    :param output_filename: Filename of the output CSVs to download.
    :param overwrite: Whether to force the download even if each file already exists locally.
    :return: A dictionary of dataframes with the sorted hyperdrive_arg_name indices as keys.
    """
    parent_run = get_aml_run_from_run_id(parent_run_id, aml_workspace)

    all_outputs_dfs = {}
    for child_run in parent_run.get_children():
        child_run_index = get_tags_from_hyperdrive_run(child_run, hyperdrive_arg_name)
        if child_run_index is None:
            raise ValueError(f"Child run expected to have the tag '{hyperdrive_arg_name}'")
        child_dir = download_dir / str(child_run_index)
        try:
            child_csv = download_file_if_necessary(child_run, output_filename, child_dir / output_filename,
                                                   overwrite=overwrite)
            all_outputs_dfs[child_run_index] = pd.read_csv(child_csv)
        except Exception as e:
            print(f"Failed to download {output_filename} for run {child_run.id}: {e}")
    return dict(sorted(all_outputs_dfs.items()))  # type: ignore


def download_hyperdrive_metrics_if_required(parent_run_id: str, download_dir: Path, aml_workspace: Workspace,
                                            hyperdrive_arg_name: str = "crossval_index",
                                            overwrite: bool = False) -> Path:
    """Fetch metrics logged to Azure ML from hyperdrive runs.

    Will only download the metrics if they do not already exist locally, as this can take several
    seconds for each child run.

    :param parent_run_id: Azure ML run ID for the parent Hyperdrive run.
    :param download_dir: Directory where to save the downloaded metrics as `aml_metrics.json`.
    :param aml_workspace: Azure ML workspace in which the runs were executed.
    :param hyperdrive_arg_name: Name of the Hyperdrive argument used for indexing the child runs.
    :param overwrite: Whether to force the download even if metrics are already saved locally.
    :return: The path of the downloaded json file.
    """
    metrics_json = download_dir / "aml_metrics.json"
    if not overwrite and metrics_json.is_file():
        print(f"AML metrics file already exists at {metrics_json}")
    else:
        metrics_df = aggregate_hyperdrive_metrics(run_id=parent_run_id,
                                                  child_run_arg_name=hyperdrive_arg_name,
                                                  aml_workspace=aml_workspace)
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing AML metrics file to {metrics_json}")
        df_to_json(metrics_df, metrics_json)
    return metrics_json


def collect_hyperdrive_metrics(metrics_json: Path) -> pd.DataFrame:
    """
    Collect the hyperdrive metrics from the downloaded metrics json file in a dataframe.
    :param metrics_json: Path of the downloaded metrics file `aml_metrics.json`.
    :return: A dataframe in the format returned by :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    """
    metrics_df = pd.read_json(metrics_json).sort_index(axis='columns')
    return metrics_df


def get_hyperdrive_metrics_table(metrics_df: pd.DataFrame, metrics_list: Sequence[str]) -> pd.DataFrame:
    """Format raw hyperdrive metrics into a table with a summary "Mean ± Std" column.

    Note that this function only supports scalar metrics. To format metrics that are logged
    throughout training, you should call :py:func:`get_best_epoch_metrics()` first.

    :param metrics_df: Metrics dataframe, as returned by :py:func:`collect_hyperdrive_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :param metrics_list: The list of metrics to include in the table.
    :return: A dataframe with the values of the selected metrics formatted as strings, including a
        header and a summary column.
    """
    header = ["Metric"] + [f"Child {k}" for k in metrics_df.columns] + ["Mean ± Std"]
    metrics_rows = []
    for metric in metrics_list:
        values: pd.Series = metrics_df.loc[metric]
        mean = values.mean()
        std = values.std()
        round_values: List[str] = [f"{v:.3f}" if v is not None else str(np.nan) for v in values]
        agg_values: List[str] = [f"{mean:.3f} ± {std:.3f}"]
        row = [metric] + round_values + agg_values
        metrics_rows.append(row)
    table = pd.DataFrame(metrics_rows, columns=header).set_index(header[0])
    return table


def get_best_epochs(metrics_df: pd.DataFrame, primary_metric: str, max_epochs_dict: Dict[int, int],
                    maximise: bool = True) -> Dict[int, Any]:
    """Determine the best epoch for each hyperdrive child run based on a given metric.

    The returned epoch indices are relative to the logging frequency of the chosen metric, i.e.
    should not be mixed between pipeline stages that log metrics at different epoch intervals.

    :param metrics_df: Metrics dataframe, as returned by :py:func:`collect_hyperdrive_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :param primary_metric: Name of the reference metric to optimise.
    :max_epochs_dict: A dictionary of the maximum number of epochs in each cross-validation round.
    :param maximise: Whether the given metric should be maximised (minimised if `False`).
    :return: Dictionary mapping each hyperdrive child index to its best epoch.
    """
    best_epochs: Dict[int, Any] = {}
    for child_index in metrics_df.columns:
        primary_metric_list = metrics_df[child_index][primary_metric]
        if primary_metric_list is not None:
            # If extra validation epoch was logged (N+1), return only the first N elements
            primary_metric_list = primary_metric_list[:-1] \
                if (len(primary_metric_list) == max_epochs_dict[child_index] + 1) else primary_metric_list
            best_epochs[child_index] = int(np.argmax(primary_metric_list)
                                           if maximise else np.argmin(primary_metric_list))
        else:
            best_epochs[child_index] = None
    return best_epochs


def get_best_epoch_metrics(metrics_df: pd.DataFrame, metrics_list: Sequence[str],
                           best_epochs: Dict[int, Any]) -> pd.DataFrame:
    """Extract the values of the selected hyperdrive metrics at the given best epochs.

    The `best_epoch` indices are relative to the logging frequency of the chosen primary metric,
    i.e. the metrics in `metrics_list` must have been logged at the same epoch intervals.

    :param metrics_df: Metrics dataframe, as returned by :py:func:`collect_hyperdrive_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :param metrics_list: Names of the metrics to index by the best epoch indices provided. Their
        values in `metrics_df` should be lists.
    :param best_epochs: Dictionary of hyperdrive child runs indices to best epochs, as returned by
        :py:func:`get_best_epochs()`.
    :return: Dataframe with the same columns as `metrics_df` and rows specified by `metrics_list`,
        containing only scalar values.
    """
    best_metrics = [metrics_df.loc[metrics_list, k].apply(lambda values: values[epoch])
                    if epoch is not None else metrics_df.loc[metrics_list, k] for k, epoch in best_epochs.items()]
    best_metrics_df = pd.DataFrame(best_metrics).T
    return best_metrics_df


def get_formatted_run_info(parent_run: Run) -> str:
    """Format Azure ML hyperdrive run information as HTML.

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


def get_child_runs_hyperparams(metrics_df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Get the hyperparameters of each child run from the metrics dataframe.
    :param: metrics_df: Metrics dataframe, as returned by :py:func:`collect_hyperdrive_metrics()` and
        :py:func:`~health_azure.aggregate_hyperdrive_metrics()`.
    :return: A dictionary of hyperparameter dictionaries for the child runs.
    """
    hyperparams_children = {}
    for child_index in metrics_df.columns:
        hyperparams = metrics_df[child_index][AMLMetricsJsonKey.HYPERPARAMS]
        hyperparams_dict = dict(zip(hyperparams[AMLMetricsJsonKey.NAME], hyperparams[AMLMetricsJsonKey.VALUE]))
        hyperparams_children[child_index] = hyperparams_dict
    return hyperparams_children


def collect_class_info(hyperparams_children: Dict[int, Dict]) -> Tuple[int, List[str]]:
    """
    Get the class names from the hyperparameters of child runs.
    :param hyperparams_children: Dict of hyperparameter dicts, as returned by :py:func:`get_child_runs_hyperparams()`.
    :return: Number of classes and list of class names.
    """
    hyperparams_single_run = list(hyperparams_children.values())[0]
    num_classes = int(hyperparams_single_run[AMLMetricsJsonKey.N_CLASSES])
    class_names = hyperparams_single_run[AMLMetricsJsonKey.CLASS_NAMES]
    if class_names == "None":
        class_names = None
    else:
        # Remove [,], and quotation marks from the string of class names
        class_names = [name.lstrip() for name in class_names[1:-1].replace("'", "").split(',')]
    class_names = validate_class_names(class_names=class_names, n_classes=num_classes)
    return (num_classes, list(class_names))


def get_max_epochs(hyperparams_children: Dict[int, Dict]) -> Dict[int, int]:
    """
    Get the maximum number of epochs for each round from the metrics dataframe.
    :param hyperparams_children: Dict of hyperparameter dicts, as returned by :py:func:`get_child_runs_hyperparams()`.
    :return: Dictionary with the number of epochs in each hyperdrive run.
    """
    max_epochs_dict = {}
    for child_index in hyperparams_children.keys():
        max_epochs_dict[child_index] = int(hyperparams_children[child_index][AMLMetricsJsonKey.MAX_EPOCHS])
    return max_epochs_dict
