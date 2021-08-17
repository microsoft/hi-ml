#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from itertools import islice
from pathlib import Path
from typing import List

from azureml.core import Experiment, Run, Workspace
from azureml.tensorboard import Tensorboard

from health.azure.azure_util import fetch_run, get_most_recent_run
from health.azure.himl import get_workspace


ROOT_DIR = Path.cwd()
OUTPUT_DIR = ROOT_DIR / "outputs"
TENSORBOARD_LOG_DIR = OUTPUT_DIR / "tensorboard"


def get_aml_runs(args: Namespace, workspace: Workspace) -> List[Run]:
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
    :rtype: List[Run]
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
    return [run for run in runs if run is not None]


def main() -> None:  # pragma: no cover
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
    if not config_path.is_file():
        raise ValueError(
            "You must provide a config.json file in the root folder to connect"
            "to an AML workspace. This can be downloaded from your AML workspace (see README.md)"
            )

    workspace = get_workspace(aml_workspace=None, workspace_config_path=config_path)

    runs = get_aml_runs(args, workspace)
    if len(runs) == 0:
        raise ValueError("No runs were found")

    # start Tensorboard
    print(f"runs: {runs}")

    run_logs_dir = TENSORBOARD_LOG_DIR / args.run_logs_dir
    ts = Tensorboard(runs=runs, local_root=str(run_logs_dir), port=args.port)

    ts.start()
    print("=============================================================================\n\n")
    input("Press Enter to close TensorBoard...")
    ts.stop()


if __name__ == "__main__":
    main()
