#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
import subprocess

from pathlib import Path
from typing import List
from unittest.mock import MagicMock

from health_azure import himl_tensorboard, himl
from health_azure import utils as azure_util
from health_azure.himl_tensorboard import WrappedTensorboard
from testazure.test_himl import render_and_run_test_script, RunTarget
from testazure.utils_testazure import DEFAULT_WORKSPACE

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


def test_run_tensorboard_no_runs(tmp_path: Path) -> None:
    # if no such run exists, will fail
    with pytest.raises(Exception) as e:
        subprocess.Popen(["python", TENSORBOARD_SCRIPT_PATH, "--run_recovery_ids", "madeuprun",
                          "--log_dir", str(tmp_path)])
        assert "No runs were found" in str(e)


def test_wrapped_tensorboard_local_logs(tmp_path: Path) -> None:
    mock_run = MagicMock()
    mock_run.id = "id123"
    local_root = Path("test_data") / "dummy_summarywriter_logs"
    remote_root = tmp_path / "tensorboard_logs"
    ts = WrappedTensorboard(remote_root=str(remote_root), local_root=str(local_root), runs=[mock_run])
    url = ts.start()
    assert url is not None
    assert ts.remote_root == str(remote_root)
    assert ts._local_root == str(local_root)

    # If start is called again, should not return a new url
    new_url = ts.start()
    assert new_url is None
    ts.stop()


def test_wrapped_tensorboard_remote_logs(tmp_path: Path) -> None:
    """
    This test will create a new run under an experiment named "test_script" in your default Workspace.
    The run will create some dummy TensorBoard-compatible logs. The run is then passed to the
    WrappedTensorboard class to ensure that it works as expected.
    If running for the first time it may take a while longer since it will install PyTorch in the
    AML environment
    """
    ws = DEFAULT_WORKSPACE.workspace

    # call the script here
    extra_options = {
        "conda_channels": ["pytorch"],
        "conda_dependencies": ["pytorch=1.4.0"],
        "imports": """
import sys
""",

        "body": """
    import torch
    from torch.utils.tensorboard import SummaryWriter
    log_dir = Path("outputs")
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    x = torch.arange(-20, 20, 0.1).view(-1, 1)
    y = -2 * x + 0.1 * torch.randn(x.size())

    model = torch.nn.Linear(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(10):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.flush()
            """,
    }

    extra_args: List[str] = []
    # For this test, do not use the latest package for the AzureML run. The script only uses torch, the logic
    # that should be tested is not in AzureML. This saves one Docker image build.
    render_and_run_test_script(tmp_path, RunTarget.AZUREML, extra_options, extra_args,
                               expected_pass=True, upload_package=False)

    run = azure_util.get_most_recent_run(run_recovery_file=tmp_path / himl.RUN_RECOVERY_FILE,
                                         workspace=ws)
    run.wait_for_completion()

    log_dir = "outputs"
    local_root = tmp_path / log_dir
    local_root.mkdir(exist_ok=True)
    remote_root = str(local_root.relative_to(tmp_path)) + "/"

    ts = WrappedTensorboard(remote_root=remote_root, local_root=str(local_root), runs=[run], port=6006)
    _ = ts.start()
    assert ts.remote_root == str(remote_root)
    assert ts._local_root == str(local_root)
    ts.stop()
