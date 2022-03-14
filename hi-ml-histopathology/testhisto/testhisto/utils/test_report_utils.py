import json
from pathlib import Path
from typing import Dict, List, Sequence, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from health_azure.utils import download_file_if_necessary
from histopathology.utils.report_utils import (collect_crossval_outputs, get_best_epoch_metrics, get_best_epochs,
                                               get_crossval_metrics_table)


@pytest.mark.parametrize('overwrite', [False, True])
def test_download_from_run_if_necessary(tmp_path: Path, overwrite: bool) -> None:
    filename = "test_output.csv"
    download_dir = tmp_path
    remote_filename = "outputs/" + filename
    expected_local_path = download_dir / filename

    def create_mock_file(name: str, output_file_path: str, _validate_checksum: bool) -> None:
        Path(output_file_path).write_text("mock content")

    run = MagicMock()
    run.download_file.side_effect = create_mock_file

    local_path = download_file_if_necessary(run, remote_filename, expected_local_path)
    assert local_path == expected_local_path
    assert local_path.exists()
    run.download_file.assert_called_once()

    run.reset_mock()
    new_local_path = download_file_if_necessary(run, remote_filename, expected_local_path, overwrite=overwrite)
    assert new_local_path == local_path
    assert new_local_path.exists()
    if overwrite:
        run.download_file.assert_called_once()
    else:
        run.download_file.assert_not_called()


class MockChildRun:
    def __init__(self, run_id: str, cross_val_index: int):
        self.run_id = run_id
        self.tags = {"hyperparameters": json.dumps({"child_run_index": cross_val_index})}

    def get_metrics(self) -> Dict[str, Union[float, List[Union[int, float]]]]:
        num_epochs = 5
        return {
            "epoch": list(range(num_epochs)),
            "train/loss": [np.random.rand() for _ in range(num_epochs)],
            "train/auroc": [np.random.rand() for _ in range(num_epochs)],
            "val/loss": [np.random.rand() for _ in range(num_epochs)],
            "val/recall": [np.random.rand() for _ in range(num_epochs)],
            "test/f1score": np.random.rand(),
            "test/accuracy": np.random.rand()
        }


class MockHyperDriveRun:
    def __init__(self, child_indices: Sequence[int]) -> None:
        self.child_indices = child_indices

    def get_children(self) -> List[MockChildRun]:
        return [MockChildRun(f"run_abc_{i}456", i) for i in self.child_indices]


def test_collect_crossval_outputs(tmp_path: Path) -> None:
    download_dir = tmp_path
    crossval_arg_name = "child_run_index"
    output_filename = "output.csv"
    child_indices = [0, 3, 1]  # Missing and unsorted children

    columns = ['id', 'value', 'split']
    for child_index in child_indices:
        csv_contents = ','.join(columns) + f"\n0,0.1,{child_index}\n1,0.2,{child_index}"
        csv_path = download_dir / str(child_index) / output_filename
        csv_path.parent.mkdir()
        csv_path.write_text(csv_contents)

    with patch('histopathology.utils.report_utils.get_aml_run_from_run_id',
               return_value=MockHyperDriveRun(child_indices)):
        crossval_dfs = collect_crossval_outputs(parent_run_id="",
                                                download_dir=download_dir,
                                                aml_workspace=None,
                                                crossval_arg_name=crossval_arg_name,
                                                output_filename=output_filename)

    assert set(crossval_dfs.keys()) == set(child_indices)
    assert list(crossval_dfs.keys()) == sorted(crossval_dfs.keys())

    for child_index, child_df in crossval_dfs.items():
        assert child_df.columns.tolist() == columns
        assert child_df.loc[0, 'split'] == child_index


@pytest.fixture
def metrics_df() -> pd.DataFrame:
    return pd.DataFrame({
        0: {
            'val/accuracy': [0.3, 0.1, 0.2],
            'val/auroc': [0.3, 0.1, 0.2],
            'test/accuracy': 0.3,
            'test/auroc': 0.3
        },
        3: {
            'val/accuracy': [0.4, 0.5, 0.6],
            'val/auroc': [0.4, 0.5, 0.6],
            'test/accuracy': 0.6,
            'test/auroc': 0.6
        },
        1: {
            'val/accuracy': [0.8, 0.9, 0.7],
            'val/auroc': [0.8, 0.9, 0.7],
            'test/accuracy': 0.9,
            'test/auroc': 0.9
        }
    })


@pytest.fixture
def best_epochs(metrics_df: pd.DataFrame) -> Dict[int, int]:
    return get_best_epochs(metrics_df, 'val/accuracy', maximise=True)


@pytest.fixture
def best_epoch_metrics(metrics_df: pd.DataFrame, best_epochs: Dict[int, int]) -> pd.Series:
    metrics_list = ['val/accuracy', 'val/auroc']
    return get_best_epoch_metrics(metrics_df, metrics_list, best_epochs)


@pytest.mark.parametrize('maximise', [True, False])
def test_get_best_epochs(metrics_df: pd.DataFrame, maximise: bool) -> None:
    best_epochs = get_best_epochs(metrics_df, 'val/accuracy', maximise=maximise)
    assert list(best_epochs.keys()) == list(metrics_df.columns)
    assert all(isinstance(epoch, int) for epoch in best_epochs.values())

    expected_best = {0: 0, 1: 1, 3: 2} if maximise else {0: 1, 1: 2, 3: 0}
    for split in metrics_df.columns:
        assert best_epochs[split] == expected_best[split]


def test_get_best_epoch_metrics(metrics_df: pd.DataFrame, best_epochs: Dict[int, int]) -> None:
    metrics_list = ['val/accuracy', 'val/auroc']
    best_metrics_df = get_best_epoch_metrics(metrics_df, metrics_list, best_epochs)
    assert list(best_metrics_df.index) == metrics_list
    assert list(best_metrics_df.columns) == list(metrics_df.columns)
    # Check that all values are now scalars instead of lists:
    for metric in metrics_list:
        assert best_metrics_df.loc[metric].map(pd.api.types.is_number).all()


def _test_get_crossval_metrics_table(df: pd.DataFrame, metrics_list: List[str]) -> None:
    metrics_table = get_crossval_metrics_table(df, metrics_list)
    assert list(metrics_table.index) == metrics_list
    assert len(metrics_table.columns) == len(df.columns) + 1

    original_values = df.loc[metrics_list].values
    table_values = metrics_table.iloc[:, :-1].applymap(float).values
    assert (table_values == original_values).all()


def test_get_crossval_metrics_table_val(best_epoch_metrics: pd.DataFrame) -> None:
    _test_get_crossval_metrics_table(best_epoch_metrics, ['val/accuracy', 'val/auroc'])


def test_get_crossval_metrics_table_test(metrics_df: pd.DataFrame) -> None:
    _test_get_crossval_metrics_table(metrics_df, ['test/accuracy', 'test/auroc'])
