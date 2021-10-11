#!/usr/bin/env python3
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from argparse import ArgumentParser, Namespace
from pathlib import Path

from health_azure.utils import AzureRunIdSource, _download_files_from_run, get_aml_runs

from health_azure.himl import get_workspace
from health_azure.himl_tensorboard import determine_run_id_source


def determine_output_dir_name(args: Namespace, run_id_source: AzureRunIdSource, output_dir: Path) -> Path:
    """
    Determine the name of the directory in which to store downloaded AML Run files

    :param args: Arguments for determining the source of the AML Runs
    :param run_id_source: The source from which to download AML Runs
    :param output_dir: The path to the outputs directory in which to create this new directory
    :return: The path in which to store the AML Run files
    """
    if run_id_source == AzureRunIdSource.EXPERIMENT_LATEST:
        output_path = output_dir / args.experiment
    elif run_id_source == AzureRunIdSource.LATEST_RUN_FILE:
        output_path = output_dir / Path(args.latest_run_file).stem
    elif run_id_source == AzureRunIdSource.RUN_RECOVERY_ID:
        output_path = output_dir / args.run_recovery_id.replace(":", "")
    else:  # run_id_source == AzureRunIdSource.RUN_ID:
        output_path = output_dir / args.run_id

    output_path.mkdir(exist_ok=True)
    return output_path


def main() -> None:  # pragma: no cover
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        required=False,
        help="Path to directory to store files downloaded from the AML Run"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.json",
        required=False,
        help="Path to config.json where Workspace name is defined"
    )
    parser.add_argument(
        "--latest_run_file",
        type=str,
        required=False,
        help="Optional path to most_recent_run.txt where the ID of the latest run is stored"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=False,
        help="The name of the AML Experiment that you wish to download Run files from"
    )
    parser.add_argument(
        "--tags",
        action="append",
        default=None,
        required=False,
        help="Optional experiment tags to restrict the AML Runs that are returned"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        required=False,
        help="Optional Run ID of the run that you wish to download files from"
    )
    parser.add_argument(
        "--run_recovery_id",
        type=str,
        default=None,
        required=False,
        help="Optional run recovery ID of the run to download files from"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        required=False,
        help="Optional prefix to filter Run files by"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    config_path = Path(args.config_file)

    workspace = get_workspace(aml_workspace=None, workspace_config_path=config_path)

    run_id_source = determine_run_id_source(args)
    output_path = determine_output_dir_name(args, run_id_source, output_dir)
    prefix = args.prefix

    run = get_aml_runs(args, workspace, run_id_source)[0]

    # TODO: extend to multiple runs?
    try:  # pragma: no cover
        _download_files_from_run(run, output_dir=output_path, prefix=prefix)
        print(f"Downloaded file(s) to '{output_path}'")
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Couldn't download files from run {args.run_id}: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
