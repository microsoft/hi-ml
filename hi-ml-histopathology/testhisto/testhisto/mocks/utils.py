#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path

import pytest
from health_azure.datasets import DatasetConfig

from health_azure.utils import WORKSPACE_CONFIG_JSON, create_config_json, get_workspace
from testazure.utils_testazure import get_shared_config_json
from testhisto.mocks.base_data_generator import MockHistoDataType


def download_azure_dataset(tmp_path: Path, dataset_id: str) -> None:
    logging.info("get_workspace")
    try:
        # For local dev machines: when config.json is specified at the root of repository
        ws = get_workspace()
    except ValueError:
        # For github agents: config.json dumped from environement variables
        create_config_json(script_folder=tmp_path, shared_config_json=get_shared_config_json())
        ws = get_workspace(workspace_config_path=tmp_path / WORKSPACE_CONFIG_JSON)
    dataset = DatasetConfig(name=dataset_id, target_folder=tmp_path, use_mounting=False)
    dataset_dl_folder = dataset.to_input_dataset_local(ws)
    logging.info(f"Dataset saved in {dataset_dl_folder}")


@pytest.fixture(scope="session")
def tmp_path_to_pathmnist_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_dir = tmp_path_factory.mktemp(MockHistoDataType.PATHMNIST.value)
    download_azure_dataset(tmp_dir, dataset_id=MockHistoDataType.PATHMNIST.value)
    return tmp_dir
