#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd
from azureml.core import Run
from azureml.core.run import _OfflineRun

HIML_ROOT = Path(__file__).parent.parent.parent.parent.parent.absolute()
health_ml_root = HIML_ROOT / "hi-ml" / "src"
health_azure_root = HIML_ROOT / "hi-ml-azure" / "src"
sys.path.insert(0, str(health_ml_root))
sys.path.insert(0, str(health_azure_root))

from health_ml.utils.common_utils import df_to_json  # noqa: E402
from health_azure import aggregate_hyperdrive_metrics  # NOQA: E402
from health_azure.utils import get_aml_run_from_run_id, get_metrics_for_childless_run  # NOQA: E402


def print_metrics(metrics_list: List[str], metrics_df: pd.DataFrame) -> None:
    """
    Given a DataFrame of metric names and corresponding values, and a list of the names metrics we
    want to print, attempts to locate each metric in the table and prints some summary statistics for it.

    :param metrics_list: The list of metrics to print values for.
    :param metrics_df: A pandas DataFrame representing a table of metric names and corresponding values.
    """
    for metric in metrics_list:
        if metric in metrics_df.index.values:
            mean = metrics_df.loc[[metric]].mean(axis=1)[metric]
            std = metrics_df.loc[[metric]].std(axis=1)[metric]
            print(f"{metric}: {round(mean,4)} ± {round(std,4)}")
        else:
            print(f"Metric {metric} not found in the Hyperdrive run metrics for run id {run_id}.")


def get_metrics_from_run(run: Run, metrics_list: List[str]) -> pd.DataFrame:
    """
    Gets a list of metrics from an Azure ML Run and returns them as a Pandas DataFrame.
    If the Run has children, then the DataFrame will contain the aggregated metrics across
    each of these children.

    :param run: An AzureML Run (which could be a HyperDriveRun, or a plain Run with no children)
    :param metrics_list: A list of metrics names to include.
    :return: A Pandas DataFrame containing metric names and corresponding values.
    """
    print(f"Getting metrics for run {run.id}")
    if isinstance(run, _OfflineRun):
        raise ValueError("Can't get metrics for an OfflineRun")
    if len(list(run.get_children())) > 0:
        metrics_df = aggregate_hyperdrive_metrics(
            child_run_arg_name="crossval_index",
            run=run,
            keep_metrics=metrics_list)
    else:
        metrics_df = get_metrics_for_childless_run(
            run=run,
            keep_metrics=metrics_list)
    print_metrics(metrics_list, metrics_df)
    return metrics_df


def upload_regression_metrics_file_to_run(metrics_df: pd.DataFrame, run: Run) -> None:
    """
    For a given metrics DataFrame, creates a temporary local csv file and uploads its to the provided
    Azure ML Run. Note that this assumes there is an 'outputs' folder in your Run.

    :param metrics_df: the DataFrame of metrics that should be stored in a csv file and uploaded to the run.
    :param run: The AML Run to upload the metrics file to.
    """
    regression_results_dir = Path('/tmp') / run.id
    regression_results_dir.mkdir(exist_ok=True)
    metrics_json_output = regression_results_dir / "metrics.json"

    df_to_json(metrics_df, metrics_json_output)
    print("Uploading metrics file to AML Run")
    run.upload_file("outputs/regression_metrics.json", str(metrics_json_output))
    metrics_json_output.unlink()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=str, default='', help="The run id to retrieve metrics from")
    parser.add_argument("--metrics_list", type=str,
                        help="A comma-separated list of metrics names to retrieve from the AML Run")
    parser.add_argument("--upload_metrics_file", type=bool, default=True,
                        help="If True, saves a json file of the metrics dataframe and uploads this to the AML Run")
    args = parser.parse_args(sys.argv[1:])
    run_id = args.run
    metrics_list = args.metrics_list.split(",")

    run = get_aml_run_from_run_id(run_id)
    print(f"Run: {run}. Run id: {run.id}")
    metrics_df = get_metrics_from_run(run, metrics_list)
    if args.upload_metrics_file:
        upload_regression_metrics_file_to_run(metrics_df, run)
