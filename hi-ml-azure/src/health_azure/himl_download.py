#!/usr/bin/env python3
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import param
import sys
from pathlib import Path
from health_azure.argparsing import parse_args_and_update_config
from health_azure.himl import download_job_outputs_logs
from health_azure.utils import AmlRunScriptConfig, get_ml_client, get_workspace


class HimlDownloadConfig(AmlRunScriptConfig):
    output_dir: Path = param.ClassSelector(
        class_=Path,
        default=Path("outputs"),
        instantiate=False,
        doc="Path to directory to store files downloaded from the AML Run",
    )
    config_file: Path = param.ClassSelector(
        class_=Path,
        default=None,
        instantiate=False,
        doc="Path to config.json where Workspace name is defined. If not provided, "
        "the code will try to locate a config.json file in any of the parent "
        "folders of the current working directory",
    )

    files_to_download: str = param.String(default=None, allow_None=True, doc="Path to the file to download")


def main() -> None:  # pragma: no cover
    download_config = HimlDownloadConfig()
    download_config = parse_args_and_update_config(download_config, sys.argv[1:])

    output_dir = download_config.output_dir
    output_dir.mkdir(exist_ok=True)

    files_to_download = download_config.files_to_download

    ml_client = get_ml_client()
    for run_id in download_config.run:
        download_job_outputs_logs(ml_client, run_id, file_to_download_path=files_to_download, download_dir=output_dir)
        print("Successfully downloaded output and log files")


if __name__ == "__main__":  # pragma: no cover
    main()
