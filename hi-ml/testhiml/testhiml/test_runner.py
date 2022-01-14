#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import shutil
import sys
from pathlib import Path
from typing import List, Any
from unittest.mock import Mock, patch

import pytest
from azureml.core.run import Run

from health_ml.runner import Runner
from health_ml.utils.common_utils import RUN_RECOVERY_ID_KEY
from health_ml.utils.fixed_paths import repository_root_directory

from testhiml.utils_testhiml import create_dataset_df


def create_mock_run(mock_upload_path: Path, config: Any) -> Run:
    """
    Create a mock AzureML Run object.

    :param mock_upload_path: Path to folder to store uploaded folders.
    :param config: Deep learning config.
    :return: Mock Run.
    """

    def mock_upload_folder(name: str, path: str, datastore_name: str = None) -> None:
        """
        Mock AzureML function Run.upload_folder.
        https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py#
        upload-folder-name--path--datastore-name-none-
        """
        shutil.copytree(src=str(path), dst=str(mock_upload_path / name))

    def mock_download_file(name: str, output_file_path: str = None, _validate_checksum: bool = False) -> None:
        """
        Mock AzureML function Run.download_file.
        https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py#
        download-file-name--output-file-path-none---validate-checksum-false-
        """
        if output_file_path is not None:
            dataset_df = create_dataset_df()
            dataset_df.to_csv(output_file_path)

    child_runs: List[Run] = []
    for i in range(config.number_of_cross_validation_splits):
        child_run = Mock(name=f'mock_child_run{i}')
        child_run.__class__ = Run
        child_run.download_file = Mock(name='mock_download_file', side_effect=mock_download_file)
        child_run.id = f'child_id:{i}'
        child_run.get_tags = lambda: {"cross_validation_split_index": i, RUN_RECOVERY_ID_KEY: 'rec_id'}
        child_runs.append(child_run)

    run = Mock(name='mock_run')
    run.__class__ = Run
    run.download_file = Mock(name='mock_download_file', side_effect=mock_download_file)
    run.get_children.return_value = child_runs
    run.get_tags = lambda: {RUN_RECOVERY_ID_KEY: 'rec_id'}
    run.id = 'run_id:1'
    run.tags = {"run_recovery_id": 'id'}
    run.upload_folder = Mock(name='mock_upload_folder', side_effect=mock_upload_folder)

    return run


@pytest.fixture(scope="module")
def runner() -> Runner:
    project_root = repository_root_directory()
    return Runner(project_root=project_root)


def test_run(runner: Runner) -> None:
    model_name = "HelloContainer"
    arguments = ["", f"--model={model_name}"]
    with patch("health_ml.runner.Runner.run_in_situ") as mock_run_in_situ:
        with patch("health_ml.runner.get_workspace"):
            with patch.object(sys, "argv", arguments):
                model_config, azure_run_info = runner.run()
        mock_run_in_situ.assert_called_once()

    assert model_config is not None  # for pyright
    assert model_config.model_name == model_name
    assert azure_run_info.run is None
    assert len(azure_run_info.input_datasets) == len(azure_run_info.output_datasets) == 0
