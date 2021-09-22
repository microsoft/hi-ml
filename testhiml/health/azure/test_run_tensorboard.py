#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
import subprocess

from pathlib import Path
from unittest import mock

from health.azure import himl_tensorboard
from health.azure.himl_tensorboard import WrappedTensorboard, ROOT_DIR

from azureml.core import Experiment, Workspace


TENSORBOARD_SCRIPT_PATH = himl_tensorboard.__file__


class MockRun:
    def __init__(self) -> None:
        self.id = 'run1234'


def test_run_tensorboard_args() -> None:
    # if no required args are passed, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", TENSORBOARD_SCRIPT_PATH])
        assert "One of latest_run_file, experiment, run_recovery_ids" \
               " or run_ids must be provided" in str(e)


def test_no_config_path() -> None:
    # if no config path exists, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", TENSORBOARD_SCRIPT_PATH, "--config_path", "idontexist"])
        assert "You must provide a config.json file in the root folder to connect" in str(e)


def test_run_tensorboard_no_runs(tmp_path: Path) -> None:
    # if no such run exists, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", TENSORBOARD_SCRIPT_PATH, "--run_recovery_ids", "madeuprun",
                          "--log_dir", str(tmp_path)])
        assert "No runs were found" in str(e)


def test_wrapped_tensorboard_local_logs(tmp_path: Path) -> None:
    mock_run = mock.MagicMock()
    mock_run.id = "id123"
    local_root = Path("test_data") / "dummy_summarywriter_logs"
    remote_root = tmp_path / "tensorboard_logs"
    ts = WrappedTensorboard(remote_root=str(remote_root), local_root=str(local_root), runs=[mock_run])
    url = ts.start()
    assert url is not None
    assert ts.remote_root == str(remote_root)
    assert ts._local_root == str(local_root)
    ts.stop()


@pytest.mark.skip
def test_wrapped_tensorboard_remote_logs(tmp_path: Path) -> None:
    """
    This test expects an experiment called 'tensorboard_test' in your workspace, with at least 1 associated run
    See the scripts in test_tensorboard to create this Experiment & Run.
    :param tmp_path:
    :return:
    """
    # get the latest run in this experiment
    ws = Workspace.from_config(ROOT_DIR / "config.json")
    expt = Experiment(ws, 'tensorboard_test')
    run = next(expt.get_runs())

    log_dir = "outputs"

    local_root = tmp_path / log_dir
    local_root.mkdir(exist_ok=True)
    remote_root = str(local_root.relative_to(tmp_path)) + "/"

    ts = WrappedTensorboard(remote_root=remote_root, local_root=str(local_root), runs=[run], port=6006)
    url = ts.start()
    assert url == "http://localhost:6006/"
    assert ts.remote_root == str(remote_root)
    assert ts._local_root == str(local_root)
    ts.stop()
