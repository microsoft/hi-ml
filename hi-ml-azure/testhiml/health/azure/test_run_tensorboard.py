#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
import subprocess

from health.azure import run_tensorboard

TENSORBOARD_SCRIPT_PATH = run_tensorboard.__file__


class MockRun:
    def __init__(self) -> None:
        self.id = 'run1234'


def test_run_tensorboard_args() -> None:
    # if no required args are passed, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", TENSORBOARD_SCRIPT_PATH])
        assert "One of latest_run_file, experiment_name, run_recovery_ids" \
               " or run_ids must be provided" in str(e)


def test_no_config_path() -> None:
    # if no config path exists, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", TENSORBOARD_SCRIPT_PATH, "--config_path", "idontexist"])
        assert "You must provide a config.json file in the root folder to connect" in str(e)


def test_run_tensorboard_no_runs() -> None:
    # if no such run exists, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", TENSORBOARD_SCRIPT_PATH, "--run_recovery_ids", "madeuprun"])
        assert "No runs were found" in str(e)
