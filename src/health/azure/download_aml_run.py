#!/usr/bin/env python3
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from argparse import ArgumentParser
from pathlib import Path

from health.azure.azure_util import get_aml_runs

from health.azure.himl import get_workspace
from health.azure.run_tensorboard import determine_run_id_source


def main() -> None:  # pragma: no cover
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        required=False,
        help="Path to directory to store  files downloaded from Run"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.json",
        required=False,
        help="Path to config.json where Workspace name is defined"
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
        help="The number of most recent runs that you wish to download"
    )
    parser.add_argument(
        "--tags",
        action="append",
        default=None,
        required=False,
        help="Optional experiment tags to restrict the AML Runs that are returned"
    )
    parser.add_argument(
        "--run_ids",
        action="append",
        type=str,
        default=None,
        required=False,
        help="Optional Run ID that you wish to download files for"
    )
    parser.add_argument(
        "--run_recovery_ids",
        default=None,
        action='append',
        required=False,
        help="Optional run recovery ids of the runs to plot"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    config_path = Path(args.config_path)
    if not config_path.is_file():
        raise ValueError(
            "You must provide a config.json file in the root folder to connect"
            "to an AML workspace. This can be downloaded from your AML workspace (see README.md)"
            )

    workspace = get_workspace(aml_workspace=None, workspace_config_path=config_path)

    run_id_source = determine_run_id_source(args)
    run = get_aml_runs(args, workspace, run_id_source)[0]

    # TODO: extend to multiple runs?
    try:  # pragma: no cover
        run.download_files(output_directory=str(output_dir))
        print(f"Downloading files to {args.output_dir} ")
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Couldn't download files from run {args.run_id}: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
