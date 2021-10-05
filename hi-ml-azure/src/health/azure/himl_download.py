#!/usr/bin/env python3
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import param
from pathlib import Path

import health.azure.azure_util as util
from health.azure.himl import get_workspace


class HimlDownloadConfig(util.ScriptConfig):
    output_dir: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                           doc="Path to directory to store files downloaded from the AML Run")
    config_file: Path = param.ClassSelector(class_=Path, default=None, instantiate=False,
                                            doc="Path to config.json where Workspace name is defined")

    prefix: str = param.String(default=None, allow_None=True, doc="Optional prefix to filter Run files by")


def main() -> None:  # pragma: no cover

    download_config = HimlDownloadConfig.parse_args()
    config_path = download_config.config_file
    output_dir = download_config.output_dir
    prefix = download_config.prefix

    output_dir.mkdir(exist_ok=True)
    workspace = get_workspace(aml_workspace=None, workspace_config_path=config_path)

    runs = util.get_runs_from_script_config(download_config, workspace)
    for run in runs:
        output_folder = output_dir / run.id

        try:  # pragma: no cover
            util.download_run_files(run, output_dir=output_folder, prefix=prefix)
            print(f"Downloaded file(s) to '{output_folder}'")
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Failed to download files from run {run.id}: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
