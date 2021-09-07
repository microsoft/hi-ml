#!/usr/bin/env python3
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from argparse import ArgumentParser
from pathlib import Path

from azureml.tensorboard import Tensorboard

from health.azure.azure_util import get_aml_runs, determine_run_id_source
from health.azure.himl import get_workspace


ROOT_DIR = Path.cwd()
OUTPUT_DIR = ROOT_DIR / "outputs"


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
        "--tags",
        action="append",
        default=None,
        required=False,
        help="Optional experiment tags to restrict the AML Runs that are returned"
    )
    parser.add_argument(
        "--run_recovery_ids",
        default=None,
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

    run_logs_dir = OUTPUT_DIR / args.run_logs_dir
    run_logs_dir.mkdir(exist_ok=True)
    ts = Tensorboard(runs=runs, local_root=str(run_logs_dir), port=args.port)

    ts.start()
    print("=============================================================================\n\n")
    input("Press Enter to close TensorBoard...")
    ts.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
