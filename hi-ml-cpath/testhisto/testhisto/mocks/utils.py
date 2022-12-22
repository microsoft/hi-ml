#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path

from health_azure.datasets import DatasetConfig

from health_azure.utils import WORKSPACE_CONFIG_JSON, check_config_json, get_workspace
from testazure.utils_testazure import get_shared_config_json


def download_azure_dataset(tmp_path: Path, dataset_id: str) -> None:
    logging.info("Trying retrieve AML workspace via get_workspace")
    try:
        # For local dev machines: when config.json is specified at the root of repository
        ws = get_workspace()
    except ValueError:
        # For github agents: config.json dumped from environement variables
        with check_config_json(script_folder=tmp_path, shared_config_json=get_shared_config_json()):
            ws = get_workspace(workspace_config_path=tmp_path / WORKSPACE_CONFIG_JSON)
    dataset = DatasetConfig(name=dataset_id, target_folder=tmp_path, use_mounting=False)
    dataset_dl_folder = dataset.to_input_dataset_local(strictly_aml_v1=True, workspace=ws)
    logging.info(f"Dataset saved in {dataset_dl_folder}")
