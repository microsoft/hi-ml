#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from argparse import ArgumentParser
from pathlib import Path

import pytest
import subprocess

from health.azure import download_aml_run
from health.azure.azure_util import AzureRunIdSource

DOWNLOAD_SCRIPT_PATH = download_aml_run.__file__


def test_download_aml_run_args(tmp_path: Path) -> None:
    # if no required args are passed, will fail
    # TODO: Create a Run and afterwards check that expected dirs have been downloaded
    tmp_output_dir = tmp_path / "tmp_dir"

    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", DOWNLOAD_SCRIPT_PATH, "--output_dir", str(tmp_output_dir)])
        assert 'One of latest_run_file, experiment_name, run_recovery_id ' \
               'or run_id must be provided' in str(e)


def test_no_config_path() -> None:
    # if no config path exists, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", DOWNLOAD_SCRIPT_PATH, "--config_path", "idontexist"])
        assert "You must provide a config.json file in the root folder to connect" in str(e)


def test_download_aml_run_no_runs() -> None:
    # if no such run exists, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", DOWNLOAD_SCRIPT_PATH, "--run_id", "madeuprun"])
        assert "was not found" in str(e)


def test_determine_output_dir_name(tmp_path: Path) -> None:
    mock_output_dir = tmp_path / "outputs"
    mock_output_dir.mkdir(exist_ok=True)

    parser = ArgumentParser()
    parser.add_argument("--latest_run_file", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--run_recovery_id", type=str)
    parser.add_argument("--run_id", type=str)

    # if experiment name is provided, expect that to be included in the directory
    mock_experiment_name = "fake-experiment"
    mock_args = parser.parse_args(["--experiment_name", mock_experiment_name])
    run_id_source = AzureRunIdSource.EXPERIMENT_LATEST
    output_dir = download_aml_run.determine_output_dir_name(mock_args, run_id_source, mock_output_dir)
    assert output_dir == mock_output_dir / mock_experiment_name

    # if latest run path is provided, expect that to be included in the directory path
    mock_args = parser.parse_args(["--latest_run_file", "most_recent_run.txt"])
    run_id_source = AzureRunIdSource.LATEST_RUN_FILE
    output_dir = download_aml_run.determine_output_dir_name(mock_args, run_id_source, mock_output_dir)
    assert output_dir == mock_output_dir / "most_recent_run"

    # if run ID is provided, expect that to be included in the directory path
    mock_args = parser.parse_args(["--run_id", "run123abc"])
    run_id_source = AzureRunIdSource.RUN_ID
    output_dir = download_aml_run.determine_output_dir_name(mock_args, run_id_source, mock_output_dir)
    assert output_dir == mock_output_dir / "run123abc"

    # if run recovery ID is provided, expect that to be included in the directory path
    mock_args = parser.parse_args(["--run_recovery_id", "experiment:run123abc"])
    run_id_source = AzureRunIdSource.RUN_RECOVERY_ID
    output_dir = download_aml_run.determine_output_dir_name(mock_args, run_id_source, mock_output_dir)
    assert output_dir == mock_output_dir / "experimentrun123abc"
