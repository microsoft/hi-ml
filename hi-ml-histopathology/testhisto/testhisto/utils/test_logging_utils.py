from pathlib import Path

from histopathology.utils.logging_utils import DeepMILOutputsHandler
from histopathology.utils.naming import MetricsKey


def _create_outputs_handler(outputs_root: Path) -> DeepMILOutputsHandler:
    return DeepMILOutputsHandler(
        outputs_root=outputs_root,
        n_classes=1,
        tile_size=224,
        level=1,
        slide_dataset=None,
        class_names=None,
        primary_val_metric=MetricsKey.ACC,
        maximise=True,
    )


def test_best_val_metric_persistence(tmp_path: Path) -> None:
    initial_epoch = 0
    initial_value = float('-inf')

    outputs_handler = _create_outputs_handler(tmp_path)
    assert outputs_handler._best_metric_epoch == initial_epoch
    assert outputs_handler._best_metric_value == initial_value

    arbitrary_epoch = 42
    arbitrary_value = 0.123
    outputs_handler._best_metric_epoch = arbitrary_epoch
    outputs_handler._best_metric_value = arbitrary_value
    outputs_handler._save_best_metric()

    reloaded_outputs_handler = _create_outputs_handler(tmp_path)
    assert reloaded_outputs_handler._best_metric_epoch == arbitrary_epoch
    assert reloaded_outputs_handler._best_metric_value == arbitrary_value

    outputs_handler.best_metric_file_path.unlink()  # delete best metric file

    fresh_outputs_handler = _create_outputs_handler(tmp_path)
    assert fresh_outputs_handler._best_metric_epoch == initial_epoch
    assert fresh_outputs_handler._best_metric_value == initial_value
