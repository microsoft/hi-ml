#!/usr/bin/env python3
import json
import os

from argparse import ArgumentParser, Namespace
from itertools import islice
from pathlib import Path
from typing import List, Optional, Tuple

from azureml.core import Experiment, Run, Workspace
from azureml.tensorboard import Tensorboard

from health.azure.azure_util import fetch_run, get_most_recent_run


# TODO: replace root dir with util to find
ROOT_DIR = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
TENSORBOARD_LOG_DIR = OUTPUT_DIR / "tensorboard"


def get_azure_secrets(config_path: Optional[Path] = None) -> Tuple[str, str, str]:
    """

    Retrieve secrets for connecting to Azure ML, either from config (if file exists)
    or else from environment variables

    :param config_path: Optional path to config.json file containing AML secrets
    :type config_path: Optional[Path]
    :return: subscription_id, resource_group and workspace_name as read from file or env vars
    :rtype: Tuple[str, str, str]
    """
    # Load config and retrieve AML Workspace
    if config_path is not None:
        with open(config_path, "r") as f_path:
            config = json.load(f_path)
            subscription_id = config.get("subscription_id")
            resource_group = config.get("resource_group")
            workspace_name = config.get("workspace_name")
    else:
        subscription_id = os.getenv("subscription_id")
        resource_group = os.getenv("resource_group")
        workspace_name = os.getenv("workspace_name")
    return subscription_id, resource_group, workspace_name


def get_aml_runs(args: Namespace, workspace: Workspace) -> List[Optional[Run]]:
    """
    Download runs from Azure ML. Runs are specified either in file specified in latest_run_path,
    by run_recovery_ids, or else the latest 'num_runs' runs from experiment 'experiment_name' as
    specified in args.

    :param args: Arguments containing either path to most_recent_run.txt,
    experiment name or run recovery id
    :type args: Namespace
    :param workspace: Azure ML Workspace
    :type workspace: Workspace
    :raises ValueError: If experiment_name in args does not exist in the Workspace
    :return: List of Azure ML Runs, or an empty list if none are retrieved
    :rtype: List[Optional[Run]]
    """
    if args.latest_run_path is not None:
        latest_run_path = Path(args.latest_run_path)
        runs = [get_most_recent_run(latest_run_path, workspace)]  # list of length 1 (most recent Run)
    elif args.experiment_name is not None:
        experiment_name = args.experiment_name
        tags = args.tags if len(args.tags) > 0 else None
        num_runs = args.num_runs

        if experiment_name not in workspace.experiments:
            raise ValueError(f"No such experiment {experiment_name} in workspace")

        experiment: Experiment = workspace.experiments[experiment_name]
        runs = list(islice(experiment.get_runs(tags=tags), num_runs))
    elif len(args.run_recovery_ids) > 0:
        runs = [fetch_run(workspace, run_id) for run_id in args.run_recovery_ids]
    return runs


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.json",
        required=False,
        help="Path to config.json where Workspace name is defined"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        required=False,
        help="The port to run Tensorboard on"
    )
    parser.add_argument(
        "--run_logs_dir",
        type=str,
        default="tensorboard_logs",
        required=False,
        help="Path to directory in which to store Tensorboard logs"
    )
    parser.add_argument(
        "--latest_run_path",
        type=str,
        required=False,
        help="Optional path to most_recent_run.txt where details on latest run are stored"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=False,
        help="The name of the AML Experiment that you wish to view Runs from"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        required=False,
        help="The number of most recent runs that you wish to view in Tensorboard"
    )
    parser.add_argument(
        "--tags",
        action="append",
        default=[],
        required=False,
        help="Optional experiment tags to restrict the AML Runs that are returned"
    )
    parser.add_argument(
        "--run_recovery_ids",
        default=[],
        action='append',
        required=False,
        help="Optional run recovery ids of the runs to plot"
    )

    args = parser.parse_args()

    config_path = Path(args.config_path)
    subscription_id, resource_group, workspace_name = get_azure_secrets(config_path)

    workspace = Workspace(subscription_id, resource_group, workspace_name)

    runs = get_aml_runs(args, workspace)
    if len(runs) == 0:
        raise ValueError("No runs were found")

    # start Tensorboard
    print(f"runs: {runs}")

    run_logs_dir = TENSORBOARD_LOG_DIR / args.run_logs_dir
    ts = Tensorboard(runs=runs, local_root=str(run_logs_dir), port=args.port)

    ts.start()
    print("==============================================================================\n\n")
    input("Press Enter to close TensorBoard...")
    ts.stop()


if __name__ == "__main__":
    main()
