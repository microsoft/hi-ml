from pathlib import Path
from typing import Dict, List, Mapping
from unittest.mock import MagicMock

import pytest
from ruamel.yaml import YAML
from torchmetrics.metric import Metric

from histopathology.utils.output_utils import DeepMILOutputsHandler, OutputsPolicy
from histopathology.utils.naming import MetricsKey

_PRIMARY_METRIC_KEY = MetricsKey.ACC


def _create_outputs_policy(outputs_root: Path) -> OutputsPolicy:
    return OutputsPolicy(outputs_root=outputs_root,
                         primary_val_metric=_PRIMARY_METRIC_KEY,
                         maximise=True)


def _create_outputs_handler(outputs_root: Path) -> DeepMILOutputsHandler:
    return DeepMILOutputsHandler(
        outputs_root=outputs_root,
        n_classes=1,
        tile_size=224,
        level=1,
        slides_dataset=None,
        class_names=None,
        primary_val_metric=_PRIMARY_METRIC_KEY,
        maximise=True,
    )


def _get_mock_metrics_dict(value: float) -> Dict[MetricsKey, Metric]:
    mock_metric = MagicMock()
    mock_metric.compute.return_value = value
    return {_PRIMARY_METRIC_KEY: mock_metric}


def test_outputs_policy_persistence(tmp_path: Path) -> None:
    initial_epoch = 0
    initial_value = float('-inf')

    # New policy should match initial settings
    policy = _create_outputs_policy(tmp_path)
    assert policy._best_metric_epoch == initial_epoch
    assert policy._best_metric_value == initial_value

    # Recreating a policy should recover the same (arbitrary) settings
    arbitrary_epoch = 42
    arbitrary_value = 0.123
    policy._best_metric_epoch = arbitrary_epoch
    policy._best_metric_value = arbitrary_value
    policy._save_best_metric()

    reloaded_policy = _create_outputs_policy(tmp_path)
    assert reloaded_policy._best_metric_epoch == arbitrary_epoch
    assert reloaded_policy._best_metric_value == arbitrary_value

    # Policy re-creation should fail if primary metric name differs from what is saved
    wrong_metric_name = 'wrong_metric_name'
    yaml = YAML()
    contents = yaml.load(policy.best_metric_file_path)
    contents[OutputsPolicy._PRIMARY_METRIC_KEY] = wrong_metric_name
    yaml.dump(contents, policy.best_metric_file_path)

    with pytest.raises(ValueError) as e:
        _create_outputs_policy(tmp_path)
    assert wrong_metric_name in str(e.value)

    # If the best-metric file is missing, a new policy should have a fresh initialisation
    policy.best_metric_file_path.unlink()  # delete best metric file

    fresh_policy = _create_outputs_policy(tmp_path)
    assert fresh_policy._best_metric_epoch == initial_epoch
    assert fresh_policy._best_metric_value == initial_value


def test_overwriting_val_outputs(tmp_path: Path) -> None:
    mock_output_filename = "mock_output.txt"

    def mock_save_outputs(epoch_results: List, metrics_dict: Mapping[MetricsKey, Metric], outputs_dir: Path) -> None:
        outputs_dir.mkdir(exist_ok=True, parents=True)
        metric_value = metrics_dict[_PRIMARY_METRIC_KEY].compute()
        mock_output_file = outputs_dir / mock_output_filename
        mock_output_file.write_text(str(metric_value))

    outputs_handler = _create_outputs_handler(tmp_path)
    outputs_handler._save_outputs = MagicMock(side_effect=mock_save_outputs)
    mock_output_file = outputs_handler.validation_outputs_dir / mock_output_filename
    previous_mock_output_file = outputs_handler.previous_validation_outputs_dir / mock_output_filename

    assert not outputs_handler.validation_outputs_dir.exists()
    assert not outputs_handler.previous_validation_outputs_dir.exists()

    # Call first time: expected to save
    initial_metric_value = 0.5
    outputs_handler.save_validation_outputs(epoch_results=[],
                                            metrics_dict=_get_mock_metrics_dict(initial_metric_value),
                                            epoch=0)
    outputs_handler._save_outputs.assert_called_once()
    assert mock_output_file.read_text() == str(initial_metric_value)
    assert not outputs_handler.previous_validation_outputs_dir.exists()
    outputs_handler._save_outputs.reset_mock()

    # Call second time with worse metric value: expected to skip
    worse_metric_value = 0.3
    outputs_handler.save_validation_outputs(epoch_results=[],
                                            metrics_dict=_get_mock_metrics_dict(worse_metric_value),
                                            epoch=1)
    outputs_handler._save_outputs.assert_not_called()
    assert mock_output_file.read_text() == str(initial_metric_value)
    assert not outputs_handler.previous_validation_outputs_dir.exists()
    outputs_handler._save_outputs.reset_mock()

    # Call third time with better metric value: expected to overwrite
    better_metric_value = 0.8
    outputs_handler.save_validation_outputs(epoch_results=[],
                                            metrics_dict=_get_mock_metrics_dict(better_metric_value),
                                            epoch=2)
    outputs_handler._save_outputs.assert_called_once()
    assert mock_output_file.read_text() == str(better_metric_value)
    assert not outputs_handler.previous_validation_outputs_dir.exists()
    outputs_handler._save_outputs.reset_mock()

    # Call fourth time with best metric value, but saving fails: expected to keep previous as back-up
    best_metric_value = 0.9
    outputs_handler._save_outputs.side_effect = RuntimeError()
    with pytest.raises(RuntimeError):
        outputs_handler.save_validation_outputs(epoch_results=[],
                                                metrics_dict=_get_mock_metrics_dict(best_metric_value),
                                                epoch=3)
    assert previous_mock_output_file.read_text() == str(better_metric_value)
