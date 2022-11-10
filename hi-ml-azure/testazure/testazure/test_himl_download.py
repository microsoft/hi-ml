#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
import pytest
import subprocess

from health_azure import himl_download

DOWNLOAD_SCRIPT_PATH = himl_download.__file__


def test_download_aml_run_args(tmp_path: Path) -> None:
    # if no required args are passed, will fail
    # TODO: Create a Run and afterwards check that expected dirs have been downloaded
    tmp_output_dir = tmp_path / "tmp_dir"

    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", DOWNLOAD_SCRIPT_PATH, "--output_dir", str(tmp_output_dir)])
        assert 'One of latest_run_file, experiment, run_recovery_id ' \
               'or run_id must be provided' in str(e)


def test_no_config_path(tmp_path: Path) -> None:
    # if no config path exists, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", DOWNLOAD_SCRIPT_PATH, "--config_path", "idontexist", "--output_dir", str(tmp_path)])
        assert "You must provide a config.json file in the root folder to connect" in str(e)


def test_download_aml_run_no_runs(tmp_path: Path) -> None:
    # if no such run exists, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", DOWNLOAD_SCRIPT_PATH, "--run_id", "madeuprun", "--output_dir", str(tmp_path)])
        assert "was not found" in str(e)
