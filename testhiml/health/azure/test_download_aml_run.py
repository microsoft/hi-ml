#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import pytest
import subprocess

from health.azure import download_aml_run

DOWNLOAD_SCRIPT_PATH = download_aml_run.__file__


def test_download_aml_run_args(tmp_path: Path) -> None:
    # if no required args are passed, will fail
    # TODO: Create a Run and afterwards check that expected dirs have been downloaded
    tmp_output_dir = tmp_path / "tmp_dir"

    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", DOWNLOAD_SCRIPT_PATH, "--output_dir", str(tmp_output_dir)])
        assert 'One of latest_run_path, experiment_name, run_recovery_ids ' \
               'or run_ids must be provided' in str(e)


def test_no_config_path() -> None:
    # if no config path exists, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", DOWNLOAD_SCRIPT_PATH, "--config_path", "idontexist"])
        assert "You must provide a config.json file in the root folder to connect" in str(e)


def test_download_aml_run_no_runs() -> None:
    # if no such run exists, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", DOWNLOAD_SCRIPT_PATH, "--run_ids", "madeuprun"])
        assert "was not found" in str(e)
