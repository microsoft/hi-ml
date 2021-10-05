#!/usr/bin/env python3
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import param
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Union

import health.azure.azure_util as util

from health.azure.himl import get_workspace
# from health.azure.himl_tensorboard import determine_run_id_source


class ScriptConfig(util.GenericConfig):
    output_dir: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                           doc="Path to directory to store files downloaded from the AML Run")
    config_file: Path = param.ClassSelector(class_=Path, default=None, instantiate=False,
                                            doc="Path to config.json where Workspace name is defined")
    latest_run_file: Path = param.ClassSelector(class_=Path, default=None, instantiate=False,
                                                doc="Optional path to most_recent_run.txt where the ID of the"
                                                    "latest run is stored")
    experiment_name: str = param.String(default=None, allow_None=True,
                                        doc="The name of the AML Experiment that you wish to "
                                            "download Run files from")
    num_runs: int = param.Integer(default=1, allow_None=True, doc="The number of runs to download from the "
                                                                  "named experiment")
    tags: Dict[str, Any] = param.Dict()
    run: Union[List[str], str] = util.RunIdOrListParam(default=None,
                                                               doc="Either single or multiple run id(s). Also "
                                                                   "supports run_recovery_ids but this is not "
                                                                   "recommended")
    prefix: str = param.String(default=None, allow_None=True, doc="Optional prefix to filter Run files by")


def main() -> None:  # pragma: no cover

    script_config = ScriptConfig.parse_args()
    config_path = script_config.config_file
    output_dir = script_config.output_dir
    prefix = script_config.prefix

    output_dir.mkdir(exist_ok=True)
    workspace = get_workspace(aml_workspace=None, workspace_config_path=config_path)

    if script_config.run is None:
        if script_config.experiment_name is None:
            # default to latest run file
            latest_run_file = util._find_file("most_recent_run.txt")
            runs = [util.get_most_recent_run(latest_run_file, aml_workspace=workspace)]
        else:
            # get latest runs from experiment
            runs = util.get_latest_aml_runs_from_experiment(script_config.experiment_name, tags=script_config.tags,
                                                            num_runs=script_config.num_runs, aml_workspace=workspace)
    else:
        run_ids = script_config.run if isinstance(script_config.run, list) else [script_config.run]
        runs = [util.get_aml_run(run_id.val, aml_workspace=workspace) for run_id in run_ids]

    for run in runs:
        output_folder = output_dir / run.id

        try:  # pragma: no cover
            util.download_run_files(run, output_dir=output_folder, prefix=prefix)
            print(f"Downloaded file(s) to '{output_folder}'")
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Failed to download files from run {run.id}: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
