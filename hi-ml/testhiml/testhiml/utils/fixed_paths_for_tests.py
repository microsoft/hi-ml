#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Optional
from functools import lru_cache

from health_azure.himl import effective_experiment_name
from health_azure.utils import PathOrString
from health_azure.utils import create_aml_run_object
from testazure.utils_testazure import DEFAULT_WORKSPACE


def tests_root_directory(path: Optional[PathOrString] = None) -> Path:
    """
    Gets the full path to the root directory that holds the tests.
    If a relative path is provided then concatenate it with the absolute path
    to the repository root.

    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    root = Path(__file__).resolve().parent.parent.parent
    return root / path if path else root


def full_test_data_path(prefix: str = "", suffix: str = "") -> Path:
    """
    Takes a relative path inside the testhiml/test_data folder, and returns its absolute path.

    :param prefix: An optional prefix to the path "test_data" that comes after the root directory
    :param suffix: An optional suffix to the path "test_data"
    :return: The absolute path
    """
    data_path = tests_root_directory()
    if prefix:
        data_path = data_path / prefix

    data_path = data_path / "test_data"
    if suffix:
        data_path = data_path / suffix

    return data_path


@lru_cache(maxsize=1)
def mock_run_id(id: int = 0) -> str:
    """Create a mock aml run that contains a checkpoint for hello_world container.

    :param id: A dummy argument to be used for caching.
    :return: The run id of the created run that contains the checkpoint.
    """

    experiment_name = effective_experiment_name("himl-tests")
    run_to_download_from = create_aml_run_object(experiment_name=experiment_name, workspace=DEFAULT_WORKSPACE.workspace)
    full_file_path = full_test_data_path(suffix="hello_world_checkpoint.ckpt")

    run_to_download_from.upload_file("outputs/checkpoints/last.ckpt", str(full_file_path))
    run_to_download_from.upload_file("outputs/checkpoints/best_val_loss.ckpt", str(full_file_path))
    run_to_download_from.upload_file("custom/path/model.ckpt", str(full_file_path))

    run_to_download_from.complete()
    return run_to_download_from.id
