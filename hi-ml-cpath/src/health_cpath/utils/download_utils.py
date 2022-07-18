#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from pathlib import Path

from health_azure import download_files_from_run_id, get_workspace
from health_azure.utils import CheckpointDownloader
from health_ml.utils import fixed_paths
from health_ml.utils.common_utils import CHECKPOINT_FOLDER, DEFAULT_AML_UPLOAD_DIR
from health_ml.utils.checkpoint_utils import LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX


def download_file_if_necessary(run_id: str, remote_dir: Path, download_dir: Path, filename: str) -> None:
    """
    Function to download any file from an AML run if it doesn't exist locally
    :param run_id: run ID of the AML run
    :param remote_dir: remote directory from where the file is downloaded
    :param download_dir: local directory where to save the downloaded file
    :param filename: name of the file to be downloaded (e.g. `"test_output.csv"`).
    """
    aml_workspace = get_workspace()
    os.chdir(fixed_paths.repository_root_directory())
    local_path = download_dir / run_id.split(":")[1] / "outputs" / filename
    remote_path = remote_dir / filename
    if local_path.exists():
        print("File already exists at", local_path)
    else:
        local_dir = local_path.parent.parent
        local_dir.mkdir(exist_ok=True, parents=True)
        download_files_from_run_id(
            run_id=run_id,
            output_folder=local_dir,
            prefix=str(remote_path),
            workspace=aml_workspace,
            validate_checksum=True,
        )
        assert local_path.exists()
        print("File is downloaded at", local_path)


def get_checkpoint_downloader(ckpt_run_id: str, outputs_folder: Path) -> CheckpointDownloader:
    downloader = CheckpointDownloader(
        aml_workspace=get_workspace(),
        run_id=ckpt_run_id,
        checkpoint_filename=LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX,
        download_dir=outputs_folder,
        remote_checkpoint_dir=Path(f"{DEFAULT_AML_UPLOAD_DIR}/{CHECKPOINT_FOLDER}/"),
    )
    downloader.download_checkpoint_if_necessary()
    return downloader
