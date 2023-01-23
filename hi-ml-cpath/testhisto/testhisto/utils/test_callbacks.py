import time
import torch
import pytest
import numpy as np
import pandas as pd

from pathlib import Path

from typing import Callable, List
from unittest.mock import MagicMock, patch
from health_cpath.configs.classification.BaseMIL import BaseMIL

from health_cpath.utils.naming import ModelKey, ResultsKey
from health_cpath.utils.callbacks import LossAnalysisCallback, LossCacheDictType
from testhisto.mocks.container import MockDeepSMILETilesPanda
from testhisto.utils.utils_testhisto import run_distributed


def _assert_is_sorted(array: np.ndarray) -> None:
    assert np.all(np.diff(array) <= 0)


def _assert_loss_cache_contains_n_elements(loss_cache: LossCacheDictType, n: int) -> None:
    for key in loss_cache:
        assert len(loss_cache[key]) == n


def get_loss_cache(n_slides: int = 4, offset: int = 0, rank: int = 0) -> LossCacheDictType:
    """Get a loss cache with n_slides elements.

    :param n_slides: The number of slides, defaults to 4
    :param offset: An offset to simulate uneven samples scenario, defaults to 0
    :param rank: The rank of the process, defaults to 0
    :return: A loss cache dictionary with n_slides elements
    """
    return {
        ResultsKey.LOSS: list(range(1, n_slides + 1 - offset)),
        ResultsKey.ENTROPY: list(range(1, n_slides + 1 - offset)),
        ResultsKey.SLIDE_ID: [f"id_{i}" for i in range(rank * n_slides + offset, (rank + 1) * n_slides)],
        ResultsKey.TILE_ID: [f"a${i * (rank + 1)}$b" for i in range(rank * n_slides + offset, (rank + 1) * n_slides)],
    }


def dump_loss_cache_for_epochs(loss_callback: LossAnalysisCallback, epochs: int, stage: ModelKey) -> None:
    for epoch in range(epochs):
        loss_callback.loss_cache[stage] = get_loss_cache(n_slides=4, rank=0)
        loss_callback.save_loss_cache(epoch, stage)


@pytest.mark.parametrize("create_outputs_folders", [True, False])
def test_loss_callback_outputs_folder_exist(create_outputs_folders: bool, tmp_path: Path) -> None:
    outputs_folder = tmp_path / "outputs"
    callback = LossAnalysisCallback(outputs_folder=outputs_folder, create_outputs_folders=create_outputs_folders)
    for stage in [ModelKey.TRAIN, ModelKey.VAL]:
        for folder in [
            callback.outputs_folder,
            callback.get_cache_folder(stage),
            callback.get_scatter_folder(stage),
            callback.get_heatmap_folder(stage),
            callback.get_anomalies_folder(stage),
        ]:
            assert folder.exists() == create_outputs_folders


@pytest.mark.parametrize("analyse_loss", [True, False])
def test_analyse_loss_param(analyse_loss: bool) -> None:
    container = BaseMIL(analyse_loss=analyse_loss)
    container.data_module = MagicMock()
    callbacks = container.get_callbacks()
    assert isinstance(callbacks[-1], LossAnalysisCallback) == analyse_loss


@pytest.mark.parametrize("save_tile_ids", [True, False])
def test_save_tile_ids_param(save_tile_ids: bool) -> None:
    callback = LossAnalysisCallback(outputs_folder=Path("foo"), save_tile_ids=save_tile_ids)
    assert callback.save_tile_ids == save_tile_ids
    assert (ResultsKey.TILE_ID in callback.loss_cache[ModelKey.TRAIN]) == save_tile_ids
    assert (ResultsKey.TILE_ID in callback.loss_cache[ModelKey.VAL]) == save_tile_ids


@pytest.mark.parametrize("patience", [0, 1, 2])
def test_loss_analysis_patience(patience: int) -> None:
    callback = LossAnalysisCallback(outputs_folder=Path("foo"), patience=patience, max_epochs=10)
    assert callback.patience == patience
    assert callback.epochs_range[0] == patience

    current_epoch = 0
    if patience > 0:
        assert callback.should_cache_loss_values(current_epoch) is False
    else:
        assert callback.should_cache_loss_values(current_epoch)
    current_epoch = 5
    assert callback.should_cache_loss_values(current_epoch)


@pytest.mark.parametrize("epochs_interval", [1, 2])
def test_loss_analysis_epochs_interval(epochs_interval: int) -> None:
    max_epochs = 10
    callback = LossAnalysisCallback(
        outputs_folder=Path("foo"), patience=0, max_epochs=max_epochs, epochs_interval=epochs_interval
    )
    assert callback.epochs_interval == epochs_interval
    assert len(callback.epochs_range) == max_epochs // epochs_interval

    # First time to cache loss values
    current_epoch = 0
    assert callback.should_cache_loss_values(current_epoch)

    current_epoch = 4  # Note that PL starts counting epochs from 0, 4th epoch is actually the 5th
    if epochs_interval == 2:
        assert callback.should_cache_loss_values(current_epoch) is False
    else:
        assert callback.should_cache_loss_values(current_epoch)

    current_epoch = 5
    assert callback.should_cache_loss_values(current_epoch)

    current_epoch = max_epochs  # no loss caching for extra validation epoch
    assert callback.should_cache_loss_values(current_epoch) is False


def test_on_train_and_val_batch_end(tmp_path: Path, mock_panda_tiles_root_dir: Path) -> None:
    batch_size = 2
    container = MockDeepSMILETilesPanda(tmp_path=mock_panda_tiles_root_dir, analyse_loss=True, batch_size=batch_size)
    container.setup()
    container.create_lightning_module_and_store()

    current_epoch = 5
    trainer = MagicMock(current_epoch=current_epoch)

    callback = LossAnalysisCallback(outputs_folder=tmp_path)
    _assert_loss_cache_contains_n_elements(callback.loss_cache[ModelKey.TRAIN], 0)
    _assert_loss_cache_contains_n_elements(callback.loss_cache[ModelKey.VAL], 0)
    dataloader = iter(container.data_module.train_dataloader())

    def _call_on_batch_end_hook(on_batch_end_hook: Callable, batch_idx: int) -> None:
        batch = next(dataloader)
        outputs = container.model.training_step(batch, batch_idx)
        on_batch_end_hook(trainer, container.model, outputs, batch, batch_idx, 0)  # type: ignore

    stages = [ModelKey.TRAIN, ModelKey.VAL]
    hooks: List[Callable] = [callback.on_train_batch_end, callback.on_validation_batch_end]
    for stage, on_batch_end_hook in zip(stages, hooks):
        _call_on_batch_end_hook(on_batch_end_hook, batch_idx=0)
        _assert_loss_cache_contains_n_elements(callback.loss_cache[stage], batch_size)

        _call_on_batch_end_hook(on_batch_end_hook, batch_idx=1)
        _assert_loss_cache_contains_n_elements(callback.loss_cache[stage], 2 * batch_size)


def test_on_train_and_val_epoch_end(
    tmp_path: Path, duplicate: bool = False, uneven_samples: bool = False, rank: int = 0, world_size: int = 1,
    device: str = "cpu"
) -> None:
    current_epoch = 2
    n_slides_per_process = 4
    trainer = MagicMock(current_epoch=current_epoch)
    pl_module = MagicMock(global_rank=rank)

    loss_callback = LossAnalysisCallback(
        outputs_folder=tmp_path, num_slides_heatmap=2, num_slides_scatter=2, max_epochs=10
    )
    stages = [ModelKey.TRAIN, ModelKey.VAL]
    hooks = [loss_callback.on_train_epoch_end, loss_callback.on_validation_epoch_end]
    for stage, on_epoch_hook in zip(stages, hooks):
        offset = rank * int(uneven_samples) if stage == ModelKey.VAL else 0  # simulate uneven samples for validation
        loss_callback.loss_cache[stage] = get_loss_cache(n_slides=n_slides_per_process, offset=offset, rank=rank)

        if duplicate:
            # Duplicate slide "id_0" to test that the duplicates are removed
            loss_callback.loss_cache[stage][ResultsKey.SLIDE_ID][0] = "id_0"
        _assert_loss_cache_contains_n_elements(loss_callback.loss_cache[stage], n_slides_per_process - offset)
        on_epoch_hook(trainer, pl_module)
        # Loss cache is flushed after each epoch
        _assert_loss_cache_contains_n_elements(loss_callback.loss_cache[stage], 0)

        if rank > 0:
            time.sleep(10)  # Wait for rank 0 to save the loss cache in a csv file

        loss_cache_path = loss_callback.get_loss_cache_file(current_epoch, stage)
        assert loss_callback.get_cache_folder(stage).exists()
        assert loss_cache_path.exists()
        assert loss_cache_path.parent == loss_callback.get_cache_folder(stage)

        loss_cache = pd.read_csv(loss_cache_path)
        total_slides = n_slides_per_process * world_size if not duplicate else n_slides_per_process * world_size - 1
        total_slides = total_slides - int(uneven_samples) if stage == ModelKey.VAL else total_slides
        _assert_loss_cache_contains_n_elements(loss_cache, total_slides)
        _assert_is_sorted(loss_cache[ResultsKey.LOSS].values)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_on_train_epoch_end_distributed(tmp_path: Path) -> None:
    # Test that the loss cache is saved correctly when using multiple GPUs
    # First scenario: no duplicates
    run_distributed(test_on_train_and_val_epoch_end, [tmp_path, False, False], world_size=2)
    # Second scenario: introduce duplicates
    run_distributed(test_on_train_and_val_epoch_end, [tmp_path, True, False], world_size=2)
    # Third scenario: uneven number of samples per process
    run_distributed(test_on_train_and_val_epoch_end, [tmp_path, False, True], world_size=2)


def test_on_train_and_val_end(tmp_path: Path) -> None:
    pl_module = MagicMock(global_rank=0, _on_extra_val_epoch=False)
    max_epochs = 4
    trainer = MagicMock(current_epoch=max_epochs - 1)

    loss_callback = LossAnalysisCallback(
        outputs_folder=tmp_path, max_epochs=max_epochs, num_slides_heatmap=2, num_slides_scatter=2
    )
    stages = [ModelKey.TRAIN, ModelKey.VAL]
    hooks = [loss_callback.on_train_end, loss_callback.on_validation_end]
    for stage, on_end_hook in zip(stages, hooks):
        dump_loss_cache_for_epochs(loss_callback, max_epochs, stage)
        on_end_hook(trainer, pl_module)

        for epoch in range(max_epochs):
            assert loss_callback.get_loss_cache_file(epoch, stage).exists()

        # check save_loss_ranks outputs
        assert loss_callback.get_all_epochs_loss_cache_file(stage).exists()
        assert loss_callback.get_loss_stats_file(stage).exists()
        assert loss_callback.get_loss_ranks_file(stage).exists()
        assert loss_callback.get_loss_ranks_stats_file(stage).exists()

        # check plot_slides_loss_scatter outputs
        assert loss_callback.get_scatter_plot_file(loss_callback.HIGHEST, stage).exists()
        assert loss_callback.get_scatter_plot_file(loss_callback.LOWEST, stage).exists()

        # check plot_loss_heatmap_for_slides_of_epoch outputs
        for epoch in range(max_epochs):
            assert loss_callback.get_heatmap_plot_file(epoch, loss_callback.HIGHEST, stage).exists()
            assert loss_callback.get_heatmap_plot_file(epoch, loss_callback.LOWEST, stage).exists()


def test_on_validation_end_not_called_if_extra_val_epoch(tmp_path: Path) -> None:
    pl_module = MagicMock(global_rank=0, _on_extra_val_epoch=True)
    max_epochs = 4
    trainer = MagicMock(current_epoch=0)
    loss_callback = LossAnalysisCallback(
        outputs_folder=tmp_path, max_epochs=max_epochs, num_slides_heatmap=2, num_slides_scatter=2
    )
    with patch.object(loss_callback, "save_loss_outliers_analaysis_results") as mock_func:
        loss_callback.on_validation_end(trainer, pl_module)
        mock_func.assert_not_called()


def test_nans_detection(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    max_epochs = 2
    n_slides = 4
    loss_callback = LossAnalysisCallback(
        outputs_folder=tmp_path, max_epochs=max_epochs, num_slides_heatmap=2, num_slides_scatter=2
    )
    stages = [ModelKey.TRAIN, ModelKey.VAL]
    for stage in stages:
        for epoch in range(max_epochs):
            loss_callback.loss_cache[stage] = get_loss_cache(n_slides)
            loss_callback.loss_cache[stage][ResultsKey.LOSS][epoch] = np.nan  # Introduce NaNs
            loss_callback.save_loss_cache(epoch, stage)

        all_slides = loss_callback.select_slides_for_epoch(epoch=0, stage=stage)
        all_loss_values_per_slides = loss_callback.select_all_losses_for_selected_slides(all_slides, stage)
        loss_callback.sanity_check_loss_values(all_loss_values_per_slides, stage)

        assert "NaNs found in loss values for slide id_0" in caplog.records[-1].getMessage()
        assert "NaNs found in loss values for slide id_1" in caplog.records[0].getMessage()

        assert loss_callback.nan_slides[stage] == ["id_1", "id_0"]
        assert loss_callback.get_nan_slides_file(stage).exists()
        assert loss_callback.get_nan_slides_file(stage).parent == loss_callback.get_anomalies_folder(stage)


@pytest.mark.parametrize("log_exceptions", [True, False])
def test_log_exceptions_flag(log_exceptions: bool, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    max_epochs = 3
    trainer = MagicMock(current_epoch=max_epochs - 1)
    pl_module = MagicMock(global_rank=0, _on_extra_val_epoch=False)
    loss_callback = LossAnalysisCallback(
        outputs_folder=tmp_path, max_epochs=max_epochs,
        num_slides_heatmap=2, num_slides_scatter=2, log_exceptions=log_exceptions
    )
    stages = [ModelKey.TRAIN, ModelKey.VAL]
    hooks = [loss_callback.on_train_end, loss_callback.on_validation_end]
    for stage, on_end_hook in zip(stages, hooks):
        message = "Error while detecting " + stage.value + " loss values outliers"
        if log_exceptions:
            on_end_hook(trainer, pl_module)
            assert message in caplog.records[-1].getMessage()
        else:
            with pytest.raises(Exception, match=fr"{message}"):
                on_end_hook(trainer, pl_module)
