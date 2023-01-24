from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed
import torch.multiprocessing
from ruamel.yaml import YAML
from health_cpath.preprocessing.loading import LoadingParams
from health_cpath.utils.tiles_selection_utils import TilesSelector
from testhisto.utils.utils_testhisto import run_distributed
from torch.testing import assert_close
from torchmetrics.metric import Metric

from health_cpath.utils.naming import MetricsKey, ModelKey, ResultsKey
from health_cpath.utils.output_utils import (BatchResultsType, DeepMILOutputsHandler, EpochResultsType, OutputsPolicy,
                                             collate_results_on_cpu, gather_results)

_PRIMARY_METRIC_KEY = MetricsKey.ACC
_RANK_KEY = 'rank'


def _create_outputs_policy(outputs_root: Path) -> OutputsPolicy:
    return OutputsPolicy(outputs_root=outputs_root,
                         primary_val_metric=_PRIMARY_METRIC_KEY,
                         maximise=True)


def _create_outputs_handler(outputs_root: Path) -> DeepMILOutputsHandler:
    return DeepMILOutputsHandler(
        outputs_root=outputs_root,
        n_classes=1,
        tile_size=224,
        loading_params=LoadingParams(level=1),
        class_names=None,
        primary_val_metric=_PRIMARY_METRIC_KEY,
        maximise=True,
        val_plot_options=MagicMock(),
        test_plot_options=MagicMock(),
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


def test_overwriting_val_outputs(tmp_path: Path, rank: int = 0, world_size: int = 1, device: str = 'cpu') -> None:
    mock_output_filename = "mock_output.txt"
    is_rank_zero = (rank == 0)

    def mock_save_outputs(epoch_results: List, outputs_dir: Path, stage: ModelKey) -> None:
        assert rank == 0, f"Expected to save only on rank 0, got rank {rank}"
        assert stage == ModelKey.VAL, "Wrong model stage. Expected ModelKey.VAL"
        assert len(epoch_results) == world_size, f"Expected {world_size} results, got {len(epoch_results)}"
        assert [batch_results[_RANK_KEY] for batch_results in epoch_results] == list(range(world_size))

        outputs_dir.mkdir(exist_ok=True, parents=True)
        metric_value = epoch_results[0][_PRIMARY_METRIC_KEY]
        mock_output_file = outputs_dir / mock_output_filename
        mock_output_file.write_text(str(metric_value))

    outputs_handler = _create_outputs_handler(tmp_path)
    outputs_handler._save_outputs = MagicMock(side_effect=mock_save_outputs)  # type: ignore
    mock_output_file = outputs_handler.validation_outputs_dir / mock_output_filename
    previous_mock_output_file = outputs_handler.previous_validation_outputs_dir / mock_output_filename

    def save_validation_outputs(handler: DeepMILOutputsHandler, metric_value: float, epoch: int) -> None:
        handler.save_validation_outputs(epoch_results=[{_PRIMARY_METRIC_KEY: metric_value,  # type: ignore
                                                        _RANK_KEY: rank}],  # type: ignore
                                        metrics_dict=_get_mock_metrics_dict(metric_value),
                                        epoch=epoch,
                                        is_global_rank_zero=is_rank_zero)

    assert not outputs_handler.validation_outputs_dir.exists()
    assert not outputs_handler.previous_validation_outputs_dir.exists()

    # Call first time: expected to save
    initial_metric_value = 0.5
    save_validation_outputs(outputs_handler, initial_metric_value, epoch=0)
    if is_rank_zero:
        outputs_handler._save_outputs.assert_called_once()
        assert mock_output_file.read_text() == str(initial_metric_value)
        assert not outputs_handler.previous_validation_outputs_dir.exists()
    else:
        outputs_handler._save_outputs.assert_not_called()
    outputs_handler._save_outputs.reset_mock()

    # Call second time with worse metric value: expected to skip
    worse_metric_value = 0.3
    save_validation_outputs(outputs_handler, worse_metric_value, epoch=1)
    outputs_handler._save_outputs.assert_not_called()
    assert mock_output_file.read_text() == str(initial_metric_value)
    assert not outputs_handler.previous_validation_outputs_dir.exists()
    outputs_handler._save_outputs.reset_mock()

    # Call third time with better metric value: expected to overwrite
    better_metric_value = 0.8
    save_validation_outputs(outputs_handler, better_metric_value, epoch=2)
    if is_rank_zero:
        outputs_handler._save_outputs.assert_called_once()
        assert mock_output_file.read_text() == str(better_metric_value)
        assert not outputs_handler.previous_validation_outputs_dir.exists()
    else:
        outputs_handler._save_outputs.assert_not_called()
    outputs_handler._save_outputs.reset_mock()

    # Call fourth time with best metric value, but saving fails: expected to keep previous as back-up
    best_metric_value = 0.9
    outputs_handler._save_outputs.side_effect = RuntimeError()
    if is_rank_zero:
        with pytest.raises(RuntimeError):
            save_validation_outputs(outputs_handler, best_metric_value, epoch=3)
        assert previous_mock_output_file.read_text() == str(better_metric_value)
    else:  # Error is thrown only on rank 0
        save_validation_outputs(outputs_handler, best_metric_value, epoch=3)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_overwriting_val_outputs_distributed(tmp_path: Path) -> None:
    run_distributed(test_overwriting_val_outputs, args=(tmp_path,), world_size=2)


def _create_batch_results(batch_idx: int, batch_size: int, num_batches: int, rank: int,
                          device: str) -> BatchResultsType:
    bag_sizes = [(rank * num_batches + batch_idx) * batch_size + slide_idx + 1 for slide_idx in range(batch_size)]
    print(rank, bag_sizes)
    results: BatchResultsType = {
        ResultsKey.SLIDE_ID: [[bag_size] * bag_size for bag_size in bag_sizes],
        ResultsKey.TILE_ID: [[bag_size * bag_size + tile_idx for tile_idx in range(bag_size)]
                             for bag_size in bag_sizes],
        ResultsKey.BAG_ATTN: [torch.rand(1, bag_size, device=device) for bag_size in bag_sizes],
        ResultsKey.TRUE_LABEL: torch.randint(2, size=(batch_size,), device=device),
        ResultsKey.LOSS: torch.randn(1, device=device),
    }
    for key, values in results.items():
        if key is ResultsKey.LOSS:
            continue  # loss is a scalar
        assert len(values) == batch_size
        if key is ResultsKey.TRUE_LABEL:
            continue  # label is slide-level
        for value, ref_value in zip(values, results[ResultsKey.SLIDE_ID]):
            if isinstance(value, torch.Tensor):
                assert value.numel() == len(ref_value)
            else:
                assert len(value) == len(ref_value)
    return results


def _create_epoch_results(
    batch_size: int, num_batches: int, uneven_samples: bool, rank: int, device: str
) -> EpochResultsType:
    epoch_results: EpochResultsType = []
    for batch_idx in range(num_batches):
        if uneven_samples and rank != 0 and batch_idx == num_batches - 1:
            batch_size -= 1  # last batch has one less sample to simulate uneven samples
        batch_results = _create_batch_results(batch_idx, batch_size, num_batches, rank, device)
        epoch_results.append(batch_results)
    return epoch_results


def test_gather_results(uneven_samples: bool = False, rank: int = 0, world_size: int = 1, device: str = 'cpu') -> None:
    num_batches = 5
    batch_size = 3
    epoch_results = _create_epoch_results(batch_size, num_batches, uneven_samples, rank, device)
    assert len(epoch_results) == num_batches

    gathered_results = gather_results(epoch_results)
    assert len(gathered_results) == world_size * num_batches

    rank_offset = rank * num_batches
    for batch_idx in range(num_batches):
        assert_close(actual=gathered_results[batch_idx + rank_offset],
                     expected=epoch_results[batch_idx])


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_gather_results_distributed() -> None:
    # These tests need to be called sequentially to prevent them to be run in parallel
    run_distributed(test_gather_results, [False], world_size=1)
    run_distributed(test_gather_results, [False], world_size=2)
    run_distributed(test_gather_results, [True], world_size=2)  # uneven samples


def _test_collate_results(epoch_results: EpochResultsType, total_num_samples: int) -> None:
    collated_results = collate_results_on_cpu(epoch_results)

    for key, epoch_elements in collated_results.items():
        expected_elements = [batch_results[key] for batch_results in epoch_results]
        if key != ResultsKey.LOSS:  # loss is a single tensor per batch
            assert len(epoch_elements) == total_num_samples
            # Concatenated lists:
            expected_elements = [elem for batch_elements in expected_elements for elem in batch_elements]

        for elem in epoch_elements:
            if isinstance(elem, torch.Tensor):
                assert not elem.is_cuda, f"{key} tensor must be on CPU: {elem}"
        assert_close(epoch_elements, expected_elements, check_device=False)


def test_collate_results_cpu() -> None:
    num_batches = 5
    batch_size = 3
    epoch_results = _create_epoch_results(batch_size, num_batches, uneven_samples=False, rank=0, device='cpu')
    _test_collate_results(epoch_results, total_num_samples=num_batches * batch_size)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
@pytest.mark.parametrize('uneven_samples', [False, True])
def test_collate_results_multigpu(uneven_samples: bool) -> None:
    num_batches = 5
    batch_size = 3
    epoch_results = _create_epoch_results(batch_size, num_batches, uneven_samples, rank=0, device='cuda:0') \
        + _create_epoch_results(batch_size, num_batches, uneven_samples, rank=1, device='cuda:1')
    _test_collate_results(epoch_results, total_num_samples=2 * num_batches * batch_size - int(uneven_samples))


@pytest.mark.parametrize('save_intermediate_outputs', [True, False])
def test_results_gather_only_if_necessary(save_intermediate_outputs: bool, tmp_path: Path) -> None:
    outputs_handler = _create_outputs_handler(tmp_path)
    outputs_handler.tiles_selector = TilesSelector(2, num_slides=2, num_tiles=2)
    outputs_handler.save_intermediate_outputs = save_intermediate_outputs
    outputs_handler._save_outputs = MagicMock()  # type: ignore
    metric_value = 0.5
    with patch("health_cpath.utils.output_utils.gather_results") as mock_gather_results:
        with patch.object(outputs_handler.tiles_selector, "gather_selected_tiles_across_devices") as mock_gather_tiles:
            with patch.object(outputs_handler.tiles_selector, "_clear_cached_slides_heaps") as mock_clear_cache:
                with patch.object(outputs_handler, "should_gather_tiles") as mock_should_gather_tiles:
                    mock_should_gather_tiles.return_value = True
                    # Intermediate outputs are gathered only if save_intermediate_outputs is True
                    for rank in range(2):
                        epoch_results = [{_PRIMARY_METRIC_KEY: [metric_value] * 5, _RANK_KEY: rank}]
                        outputs_handler.save_validation_outputs(
                            epoch_results=epoch_results,  # type: ignore
                            metrics_dict=_get_mock_metrics_dict(metric_value),
                            epoch=0,
                            is_global_rank_zero=rank == 0,
                            on_extra_val=False)
                        assert mock_gather_results.called == save_intermediate_outputs
                        assert mock_gather_tiles.called == save_intermediate_outputs
                        mock_clear_cache.assert_called()
                    # Or if it's an extra validation epoch
                    outputs_handler.save_validation_outputs(
                        epoch_results=epoch_results,  # type: ignore
                        metrics_dict=_get_mock_metrics_dict(metric_value),
                        epoch=1,
                        is_global_rank_zero=True,
                        on_extra_val=True)
                    mock_gather_results.assert_called()
                    mock_gather_tiles.assert_called()
                    mock_clear_cache.assert_called()
