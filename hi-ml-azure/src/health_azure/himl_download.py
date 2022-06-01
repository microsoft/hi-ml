#!/usr/bin/env python3
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import param
import sys
from pathlib import Path
from typing import List

from azureml.core import Run

import health_azure.utils as azure_util
from health_azure.himl import RUN_RECOVERY_FILE


class HimlDownloadConfig(azure_util.AmlRunScriptConfig):
    output_dir: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                           doc="Path to directory to store files downloaded from the AML Run")
    config_file: Path = param.ClassSelector(class_=Path, default=None, instantiate=False,
                                            doc="Path to config.json where Workspace name is defined. If not provided, "
                                                "the code will try to locate a config.json file in any of the parent "
                                                "folders of the current working directory")

    prefix: str = param.String(default=None, allow_None=True, doc="Optional prefix to filter Run files by")


def retrieve_runs(download_config: HimlDownloadConfig) -> List[Run]:
    """
    Retrieve a list of AML Run objects, given a HimlDownloadConfig object which contains values for either run
    (one or more run ids), experiment (experiment name) or latest_run_file. If none of these are provided,
    the parent directories of this script will be searched for a "most_recent_run.txt" file, and the run id will
    be extracted from there, to retrieve the run object(s). If no Runs are found, a ValueError will be raised.

    :param download_config: A HimlDownloadConfig object containing run information (e.g. run ids or experiment name)
    :return: List of AML Run objects
    """
    if download_config.run is not None:
        run_ids: List[str] = download_config.run
        runs = [azure_util.get_aml_run_from_run_id(r_id) for r_id in run_ids]
        if len(runs) == 0:
            raise ValueError(f"Did not find any runs with the given run id(s): {download_config.run}")
    elif download_config.experiment is not None:
        runs = azure_util.get_latest_aml_runs_from_experiment(download_config.experiment,
                                                              download_config.num_runs,
                                                              download_config.tags,
                                                              workspace_config_path=download_config.config_file)
        if len(runs) == 0:
            raise ValueError(f"Did not find any runs under the given experiment name: {download_config.experiment}")
    else:
        most_recent_run_path = download_config.latest_run_file or Path(RUN_RECOVERY_FILE)
        run_or_recovery_id = azure_util.get_most_recent_run_id(most_recent_run_path)
        runs = [azure_util.get_aml_run_from_run_id(run_or_recovery_id,
                                                   workspace_config_path=download_config.config_file)]
        if len(runs) == 0:
            raise ValueError(f"Did not find any runs with run id {run_or_recovery_id} as found in"
                             f" {download_config.latest_run_file}")
    return runs


def main() -> None:  # pragma: no cover

    download_config = HimlDownloadConfig()
    download_config = azure_util.parse_args_and_update_config(download_config, sys.argv[1:])

    output_dir = download_config.output_dir
    output_dir.mkdir(exist_ok=True)

    runs = retrieve_runs(download_config)

    for run in runs:
        output_folder = output_dir / run.id

        try:  # pragma: no cover
            azure_util.download_files_from_run_id(run.id, output_folder=output_folder, prefix=download_config.prefix,
                                                  workspace_config_path=download_config.config_file)
            print(f"Downloaded file(s) to '{output_folder}'")
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Failed to download files from run {run.id}: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
