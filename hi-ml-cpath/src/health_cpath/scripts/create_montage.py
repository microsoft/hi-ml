#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
"""
This script can be used to create a montage of slides, given a slides dataset or a folder with images.

For a full documentation of the parameters, run `python create_montage.py --help`
"""
import logging
from pathlib import Path
import sys
from typing import Optional


current_file = Path(__file__).absolute()
repository_root = current_file.parent.parent.parent.parent.parent
folders_to_add = [
    repository_root / "hi-ml" / "src",
    repository_root / "hi-ml-azure" / "src",
    repository_root / "hi-ml-cpath" / "src",
]
for folder in folders_to_add:
    assert folder.is_dir()
    sys.path.insert(0, str(folder))

from health_azure.himl import submit_to_azure_if_needed, DatasetConfig  # noqa
from health_azure.logging import logging_to_stdout  # noqa
from health_cpath.utils.montage import create_config_from_args  # noqa


def main() -> None:
    config = create_config_from_args()
    logging_to_stdout()
    submit_to_azureml = config.cluster != ""
    if config.dataset.strip() == "":
        raise ValueError("Please provide a dataset name via --dataset")
    elif config.dataset.startswith("/"):
        if submit_to_azureml:
            raise ValueError("Cannot submit to AzureML if dataset is a local folder")
        input_folder: Optional[Path] = Path(config.dataset)
    else:
        logging.info(f"In AzureML use mounted dataset '{config.dataset}' in datastore {config.datastore}")
        input_dataset = DatasetConfig(name=config.dataset, datastore=config.datastore, use_mounting=True)
        logging.info(f"Submitting to AzureML, running on cluster {config.cluster}")
        run_info = submit_to_azure_if_needed(
            entry_script=current_file,
            snapshot_root_directory=repository_root,
            compute_cluster_name=config.cluster,
            conda_environment_file=config.conda_env,
            submit_to_azureml=submit_to_azureml,
            input_datasets=[input_dataset],
            strictly_aml_v1=True,
            docker_shm_size=config.docker_shm_size,
            wait_for_completion=config.wait_for_completion,
            workspace_config_file=config.workspace_config_path,
            display_name=config.display_name,
        )
        input_folder = run_info.input_datasets[0]

    assert input_folder is not None
    config.create_montage(input_folder)


if __name__ == "__main__":
    main()
