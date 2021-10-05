#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import List, Union
from unittest.mock import patch

import pytest
import subprocess
import sys

from health.azure import himl_download
from health.azure import azure_util as util

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


@pytest.mark.parametrize("arguments, run_id", [
    (["", "--run", "run_abc_123"], util.RunId("run_abc_123")),
    (["", "--run", "run_abc_123,run_def_456"], [util.RunId("run_abc_123"), util.RunId("run_def_456")]),
    (["", "--run", "expt_name:run_abc_123"], util.RunRecoveryId("expt_name:run_abc_123")),
])
def test_script_config_run_src(arguments: List[str], run_id: Union[List[util.RunId], util.RunId]) -> None:
    with patch.object(sys, "argv", arguments):
        script_config = himl_download.ScriptConfig.parse_args()

        if isinstance(run_id, list):
            for script_config_run, expected_run in zip(script_config.run, run_id):
                assert script_config_run.val == expected_run.val
        else:
            assert script_config.run.val == run_id.val
