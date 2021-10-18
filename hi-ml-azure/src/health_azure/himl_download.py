#!/usr/bin/env python3
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import param
from pathlib import Path

import health_azure.utils as azure_util


class HimlDownloadConfig(azure_util.AmlRunScriptConfig):
    output_dir: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                           doc="Path to directory to store files downloaded from the AML Run")
    config_file: Path = param.ClassSelector(class_=Path, default=None, instantiate=False,
                                            doc="Path to config.json where Workspace name is defined")

    prefix: str = param.String(default=None, allow_None=True, doc="Optional prefix to filter Run files by")


def main() -> None:  # pragma: no cover

    download_config = HimlDownloadConfig.parse_args()
    output_dir = download_config.output_dir
    output_dir.mkdir(exist_ok=True)

    if download_config.run is not None:
        runs = download_config.run if isinstance(download_config.run, list) else [download_config.run]
    elif download_config.experiment is not None:
        r

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
