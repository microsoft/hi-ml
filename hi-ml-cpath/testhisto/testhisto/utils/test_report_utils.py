import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pandas.testing
import pytest

from health_cpath.utils.output_utils import (AML_LEGACY_TEST_OUTPUTS_CSV, AML_OUTPUTS_DIR, AML_TEST_OUTPUTS_CSV,
                                             AML_VAL_OUTPUTS_CSV)
from health_cpath.utils.report_utils import (collect_hyperdrive_metrics, collect_hyperdrive_outputs,
                                             child_runs_have_val_and_test_outputs, get_best_epoch_metrics,
                                             get_best_epochs, get_hyperdrive_metrics_table,
                                             run_has_val_and_test_outputs, download_hyperdrive_metrics_if_required)


def test_run_has_val_and_test_outputs() -> None:
    arbitrary_filename = AML_OUTPUTS_DIR + "/some_other_file"

    run = MagicMock(display_name="mock run", id="abc123")

    with patch.object(run, 'get_file_names', return_value=[AML_LEGACY_TEST_OUTPUTS_CSV]):
        assert not run_has_val_and_test_outputs(run)

    with patch.object(run, 'get_file_names', return_value=[AML_VAL_OUTPUTS_CSV, AML_TEST_OUTPUTS_CSV]):
        assert run_has_val_and_test_outputs(run)

    with patch.object(run, 'get_file_names', return_value=[AML_VAL_OUTPUTS_CSV]):
        with pytest.raises(ValueError, match="does not have the expected files"):
            run_has_val_and_test_outputs(run)

    with patch.object(run, 'get_file_names', return_value=[AML_TEST_OUTPUTS_CSV]):
        with pytest.raises(ValueError, match="does not have the expected files"):
            run_has_val_and_test_outputs(run)

    with patch.object(run, 'get_file_names', return_value=[arbitrary_filename]):
        with pytest.raises(ValueError, match="does not have the expected files"):
            run_has_val_and_test_outputs(run)


def test_hyperdrive_runs_have_val_and_test_outputs() -> None:
    legacy_run = MagicMock(display_name="legacy run", id="child1")
    legacy_run.get_file_names.return_value = [AML_LEGACY_TEST_OUTPUTS_CSV]

    run_with_val_and_test = MagicMock(display_name="run with val and test", id="child2")
    run_with_val_and_test.get_file_names.return_value = [AML_VAL_OUTPUTS_CSV, AML_TEST_OUTPUTS_CSV]

    arbitrary_filename = AML_OUTPUTS_DIR + "/some_other_file"
    invalid_run = MagicMock(display_name="invalid run", id="child3")
    invalid_run.get_file_names.return_value = [arbitrary_filename]

    parent_run = MagicMock()

    with patch.object(parent_run, 'get_children', return_value=[legacy_run, legacy_run]):
        assert not child_runs_have_val_and_test_outputs(parent_run)

    with patch.object(parent_run, 'get_children', return_value=[run_with_val_and_test, run_with_val_and_test]):
        assert child_runs_have_val_and_test_outputs(parent_run)

    with patch.object(parent_run, 'get_children', return_value=[legacy_run, invalid_run]):
        with pytest.raises(ValueError, match="does not have the expected files"):
            child_runs_have_val_and_test_outputs(parent_run)

    with patch.object(parent_run, 'get_children', return_value=[run_with_val_and_test, invalid_run]):
        with pytest.raises(ValueError, match="does not have the expected files"):
            child_runs_have_val_and_test_outputs(parent_run)

    with patch.object(parent_run, 'get_children', return_value=[legacy_run, run_with_val_and_test]):
        with pytest.raises(ValueError, match="has mixed children"):
            child_runs_have_val_and_test_outputs(parent_run)


class MockChildRun:
    def __init__(self, run_id: str, child_run_index: int):
        self.run_id = run_id
        self.tags = {"hyperparameters": json.dumps({"child_run_index": child_run_index})}

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


def test_collect_hyperdrive_outputs(tmp_path: Path) -> None:
    download_dir = tmp_path
    hyperdrive_arg_name = "child_run_index"
    output_filename = "output.csv"
    child_indices = [0, 3, 1]  # Missing and unsorted children

    columns = ['id', 'value', 'split']
    for child_index in child_indices:
        csv_contents = ','.join(columns) + f"\n0,0.1,{child_index}\n1,0.2,{child_index}"
        csv_path = download_dir / str(child_index) / output_filename
        csv_path.parent.mkdir()
        csv_path.write_text(csv_contents)

    with patch('health_cpath.utils.report_utils.get_aml_run_from_run_id',
               return_value=MockHyperDriveRun(child_indices)):
        hyperdrive_dfs = collect_hyperdrive_outputs(parent_run_id="",
                                                    download_dir=download_dir,
                                                    aml_workspace=None,
                                                    hyperdrive_arg_name=hyperdrive_arg_name,
                                                    output_filename=output_filename)

    assert set(hyperdrive_dfs.keys()) == set(child_indices)
    assert list(hyperdrive_dfs.keys()) == sorted(hyperdrive_dfs.keys())

    for child_index, child_df in hyperdrive_dfs.items():
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
        },
        4: {'val/accuracy': None,
            'val/auroc': None,
            'test/accuracy': None,
            'test/auroc': None
            }
    })


@pytest.fixture
def max_epochs_dict() -> Dict[int, int]:
    return {0: 3, 1: 10, 3: 3, 4: 3}


@pytest.fixture
def best_epochs(metrics_df: pd.DataFrame, max_epochs_dict: Dict[int, int]) -> Dict[int, Any]:
    return get_best_epochs(metrics_df=metrics_df, primary_metric='val/accuracy',
                           max_epochs_dict=max_epochs_dict, maximise=True)


@pytest.fixture
def best_epoch_metrics(metrics_df: pd.DataFrame, best_epochs: Dict[int, int]) -> pd.Series:
    metrics_list = ['val/accuracy', 'val/auroc']
    return get_best_epoch_metrics(metrics_df, metrics_list, best_epochs)


@pytest.mark.parametrize('overwrite', [False, True])
def test_collect_hyperdrive_metrics(metrics_df: pd.DataFrame, tmp_path: Path, overwrite: bool) -> None:
    with patch('health_cpath.utils.report_utils.aggregate_hyperdrive_metrics',
               return_value=metrics_df) as mock_aggregate:
        returned_json = download_hyperdrive_metrics_if_required(parent_run_id="", download_dir=tmp_path,
                                                                aml_workspace=None, overwrite=overwrite)
        returned_df = collect_hyperdrive_metrics(metrics_json=returned_json)
        mock_aggregate.assert_called_once()
        mock_aggregate.reset_mock()

        new_returned_json = download_hyperdrive_metrics_if_required(parent_run_id="", download_dir=tmp_path,
                                                                    aml_workspace=None, overwrite=overwrite)
        new_returned_df = collect_hyperdrive_metrics(metrics_json=new_returned_json)
        if overwrite:
            mock_aggregate.assert_called_once()
        else:
            mock_aggregate.assert_not_called()

        pandas.testing.assert_frame_equal(returned_df, new_returned_df, check_exact=False)


@pytest.mark.parametrize('maximise', [True, False])
def test_get_best_epochs(metrics_df: pd.DataFrame, max_epochs_dict: Dict[int, int], maximise: bool) -> None:
    best_epochs = get_best_epochs(metrics_df=metrics_df, primary_metric='val/accuracy',
                                  max_epochs_dict=max_epochs_dict, maximise=maximise)
    assert list(best_epochs.keys()) == list(metrics_df.columns)
    assert all(isinstance(epoch, (int, type(None))) for epoch in best_epochs.values())

    expected_best = {0: 0, 1: 1, 3: 2, 4: None} if maximise else {0: 1, 1: 2, 3: 0, 4: None}
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


@pytest.mark.parametrize('fixture_name, metrics_list', [('metrics_df', ['test/accuracy', 'test/auroc']),
                                                        ('best_epoch_metrics', ['val/accuracy', 'val/auroc'])])
def test_get_hyperdrive_metrics_table(
    fixture_name: str, metrics_list: List[str], request: pytest.FixtureRequest
) -> None:
    df = request.getfixturevalue(fixture_name)

    metrics_table = get_hyperdrive_metrics_table(df, metrics_list)
    assert list(metrics_table.index) == metrics_list
    assert len(metrics_table.columns) == len(df.columns) + 1

    original_values = df.loc[metrics_list].values
    table_values = metrics_table.iloc[:, :-1].applymap(float).values
    assert (table_values[~pd.isnull(table_values)] == original_values[~pd.isnull(original_values)]).all()
