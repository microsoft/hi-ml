import json
import os

from argparse import ArgumentParser
from pathlib import Path
from typing import List
from unittest import mock

from azureml.core import Workspace

from health.azure.run_tensorboard import get_azure_secrets, get_aml_runs


class MockArgsWithLatestRunPath:
    def __init__(self) -> None:
        self.latest_run_path = ""


class MockRun:
    def __init__(self) -> None:
        self.id = "run1234"


def test_get_azure_secrets(tmp_path: Path) -> None:
    tmp_config_path = Path(tmp_path) / "config.json"
    expected_secrets = {
        "workspace_name": "workspace123",
        "resource_group": "rg123",
        "subscription_id": "subscription123"
    }

    # Check that vars are correctly read from environment vars if config file NOT provided
    with mock.patch.dict(os.environ, expected_secrets):
        subscription_id, resource_group, workspace_name = get_azure_secrets()
    assert resource_group == expected_secrets["resource_group"]
    assert subscription_id == expected_secrets["subscription_id"]
    assert workspace_name == expected_secrets["workspace_name"]

    with open(tmp_config_path, "w+") as f_path:
        json.dump(expected_secrets, f_path)

    # Check that vars are correctly read from environment vars if config file IS provided
    assert tmp_config_path.is_file()
    subscription_id, resource_group, workspace_name = get_azure_secrets(tmp_config_path)
    assert resource_group == expected_secrets["resource_group"]
    assert subscription_id == expected_secrets["subscription_id"]
    assert workspace_name == expected_secrets["workspace_name"]


def test_get_aml_runs() -> None:
    def _mock_get_most_recent_run(path: Path, workspace: Workspace) -> MockRun:
        return MockRun()

    def _get_experiment_runs() -> List[MockRun]:
        return [MockRun(), MockRun(), MockRun(), MockRun()]

    parser = ArgumentParser()
    parser.add_argument("--latest_run_path", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--tags", action="append", default=[])
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--run_recovery_ids", type=str, action="append", default=None)

    # Test that expected runs are returned if latest_run_path is provided
    mock_args = parser.parse_args(["--latest_run_path", "latest_run.txt"])
    with mock.patch("health.azure.run_tensorboard.Workspace") as mock_workspace:
        with mock.patch("health.azure.run_tensorboard.get_most_recent_run", _mock_get_most_recent_run):
            runs = get_aml_runs(mock_args, mock_workspace)  # type: ignore
    assert len(runs) == 1
    assert runs[0].id == "run1234"

    # Test that expected runs are returned if experiment_name is provided
    mock_args = parser.parse_args(["--experiment_name", "fake_experiment"])
    with mock.patch("health.azure.run_tensorboard.Experiment") as mock_experiment:
        mock_experiment.get_runs.return_value = _get_experiment_runs()
        with mock.patch("health.azure.run_tensorboard.Workspace",
                        experiments={'fake_experiment': mock_experiment}
                        ) as mock_workspace:
            runs = get_aml_runs(mock_args, mock_workspace)  # type: ignore
    assert len(runs) == 1
    assert runs[0].id == "run1234"

    # Test that correct number of runs are returned if both experiment_name and num_runs are provided
    mock_args = parser.parse_args(["--experiment_name", "fake_experiment", "--num_runs", "3"])
    with mock.patch("health.azure.run_tensorboard.Experiment") as mock_experiment:
        mock_experiment.get_runs.return_value = _get_experiment_runs()
        with mock.patch("health.azure.run_tensorboard.Workspace",
                        experiments={'fake_experiment': mock_experiment}
                        ) as mock_workspace:
            runs = get_aml_runs(mock_args, mock_workspace)  # type: ignore
    assert len(runs) == 3
    assert runs[0].id == "run1234"

    # Test that correct number of returns if both experiment_name and tags are provided
    mock_args = parser.parse_args(["--experiment_name", "fake_experiment", "--tags", "3"])
    with mock.patch("health.azure.run_tensorboard.Experiment") as mock_experiment:
        mock_experiment.get_runs.return_value = _get_experiment_runs()
        with mock.patch("health.azure.run_tensorboard.Workspace",
                        experiments={'fake_experiment': mock_experiment}
                        ) as mock_workspace:
            runs = get_aml_runs(mock_args, mock_workspace)  # type: ignore
    assert len(runs) == 1
    assert runs[0].id == "run1234"

    # Test that the correct number of runs are returned if run_recovery_id(s) is(are) provided
    mock_args = parser.parse_args(["--run_recovery_id", "expt:run123"])
    with mock.patch("health.azure.run_tensorboard.Workspace") as mock_workspace:
        with mock.patch("health.azure.run_tensorboard.fetch_run", _mock_get_most_recent_run):
            runs = get_aml_runs(mock_args, mock_workspace)  # type: ignore
    assert len(runs) == 1
    assert runs[0].id == "run1234"
