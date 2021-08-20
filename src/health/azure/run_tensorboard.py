#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from pathlib import Path

from azureml.tensorboard import Tensorboard

from health.azure.azure_util import AzureRunIdSource, get_aml_runs, get_most_recent_run
from health.azure.himl import get_workspace


ROOT_DIR = Path.cwd()
OUTPUT_DIR = ROOT_DIR / "outputs"
TENSORBOARD_LOG_DIR = OUTPUT_DIR / "tensorboard"


def determine_run_id_source(args: Namespace):
    """
    From the args inputted, determine what is the source of Runs to be downloaded and plotted
    (e.g. extract id from latest run path, or take most recent run of an Experiment etc. )

    :param args: Arguments for determining the source of AML Runs to be retrieved
    :return: The source from which to extract the latest Run id(s)
    """
    if args.latest_run_path is not None:
        return AzureRunIdSource.LATEST_RUN_FILE
    elif args.experiment_name is not None:
        return AzureRunIdSource.EXPERIMENT_LATEST
    elif args.run_recovery_ids is not None:
        return AzureRunIdSource.RUN_RECOVERY_ID


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

    run_id_source = determine_run_id_source(args)
    runs = get_aml_runs(args, workspace, run_id_source)
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
