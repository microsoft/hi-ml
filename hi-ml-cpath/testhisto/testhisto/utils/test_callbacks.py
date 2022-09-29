import time
import torch
import pytest
import numpy as np
import pandas as pd

from pathlib import Path
from unittest.mock import MagicMock

from health_cpath.utils.callbacks import (
    LOWEST,
    HIGHEST,
    ALL_EPOCHS_FILENAME,
    LOSS_VALUES_FILENAME,
    LOSS_RANKS_FILENAME,
    LOSS_RANKS_STATS_FILENAME,
    HEATMAP_PLOT_FILENAME,
    NAN_SLIDES_FILENAME,
    SCATTER_PLOT_FILENAME,
    LossAnalysisCallback,
    LossDictType,
)
from health_cpath.utils.naming import ResultsKey
from testhisto.mocks.container import MockDeepSMILETilesPanda
from testhisto.utils.utils_testhisto import run_distributed


def _assert_list_is_sorted(list_: np.ndarray) -> None:
    assert all(list_ == sorted(list_, reverse=True))


def _assert_loss_cache_contains_n_elements(loss_cache: LossDictType, n: int) -> None:
    for key in loss_cache:
        assert len(loss_cache[key]) == n


def dump_loss_cache_for_epochs(loss_callback: LossAnalysisCallback, epochs: int) -> None:
    for epoch in range(epochs):
        loss_callback.loss_cache = get_loss_cache()
        loss_callback.save_loss_cache(epoch)


def test_loss_callback_outputs_folder_exist(tmp_path: Path) -> None:
    outputs_folder = tmp_path / "outputs"
    outputs_folder.mkdir()
    callback = LossAnalysisCallback(outputs_folder=outputs_folder)
    for folder in [
        callback.outputs_folder,
        callback.cache_folder,
        callback.scatter_folder,
        callback.heatmap_folder,
        callback.exception_folder,
    ]:
        assert folder.exists()


@pytest.mark.parametrize("analyse_loss", [True, False])
def test_analyse_loss_param(analyse_loss: bool) -> None:
    container = MockDeepSMILETilesPanda(tmp_path=Path("foo"), analyse_loss=analyse_loss)
    callbacks = container.get_callbacks()
    assert isinstance(callbacks[-1], LossAnalysisCallback) == analyse_loss


@pytest.mark.parametrize("save_tile_ids", [True, False])
def test_save_tile_ids_param(save_tile_ids: bool) -> None:
    callback = LossAnalysisCallback(outputs_folder=Path("foo"), save_tile_ids=save_tile_ids)
    assert callback.save_tile_ids == save_tile_ids
    assert (ResultsKey.TILE_ID in callback.loss_cache) == save_tile_ids


@pytest.mark.parametrize("patience", [0, 1, 2])
def test_loss_analysis_patience(patience: int) -> None:
    callback = LossAnalysisCallback(outputs_folder=Path("foo"), patience=patience, max_epochs=10)
    assert callback.patience == patience
    assert callback.epochs_range[0] == patience

    current_epoch = 0
    if patience > 0:
        assert callback.is_time_to_cache_loss_values(current_epoch) is False
    else:
        assert callback.is_time_to_cache_loss_values(current_epoch)
    current_epoch = 5
    assert callback.is_time_to_cache_loss_values(current_epoch)


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
    assert callback.is_time_to_cache_loss_values(current_epoch)

    current_epoch = 4  # Note that PL starts counting epochs from 0, 4th epoch is actually the 5th
    if epochs_interval == 2:
        assert callback.is_time_to_cache_loss_values(current_epoch) is False
    else:
        assert callback.is_time_to_cache_loss_values(current_epoch)

    current_epoch = 5
    assert callback.is_time_to_cache_loss_values(current_epoch)


def test_on_train_batch_start(tmp_path: Path, mock_panda_tiles_root_dir: Path) -> None:
    batch_size = 2
    container = MockDeepSMILETilesPanda(tmp_path=mock_panda_tiles_root_dir)
    container.batch_size = batch_size
    container.setup()
    container.create_lightning_module_and_store()

    current_epoch = 5
    trainer = MagicMock(current_epoch=current_epoch)
    batch = next(iter(container.data_module.train_dataloader()))

    callback = LossAnalysisCallback(outputs_folder=tmp_path)
    _assert_loss_cache_contains_n_elements(callback.loss_cache, 0)

    callback.on_train_batch_start(trainer, container.model, batch, 0, None)  # type: ignore
    _assert_loss_cache_contains_n_elements(callback.loss_cache, batch_size)

    batch = next(iter(container.data_module.train_dataloader()))
    callback.on_train_batch_start(trainer, container.model, batch, 1, None)  # type: ignore
    _assert_loss_cache_contains_n_elements(callback.loss_cache, 2 * batch_size)


def get_loss_cache(n_slides: int = 4, rank: int = 0) -> LossDictType:
    return {
        ResultsKey.LOSS: list(range(1, n_slides + 1)),
        ResultsKey.SLIDE_ID: [f"id_{i * (rank + 1)}" for i in range(n_slides)],
        ResultsKey.TILE_ID: [f"a${i * (rank + 1)}$b" for i in range(n_slides)],
    }


def test_on_train_epoch_end(tmp_path: Path, rank: int = 0, world_size: int = 1, device: str = "cpu") -> None:
    current_epoch = 5
    n_slides_per_process = 4
    trainer = MagicMock(current_epoch=current_epoch)
    pl_module = MagicMock(global_rank=rank)

    loss_callback = LossAnalysisCallback(outputs_folder=tmp_path, num_slides_heatmap=2, num_slides_scatter=2)
    loss_callback.loss_cache = get_loss_cache(rank=rank, n_slides=n_slides_per_process)

    _assert_loss_cache_contains_n_elements(loss_callback.loss_cache, n_slides_per_process)
    loss_callback.on_train_epoch_end(trainer, pl_module)
    # Loss cache is flushed after each epoch
    _assert_loss_cache_contains_n_elements(loss_callback.loss_cache, 0)

    if rank > 0:
        time.sleep(10)  # Wait for the rank 0 to save the loss cache in a csv file

    loss_cache_path = loss_callback.cache_folder / LOSS_VALUES_FILENAME.format(current_epoch)
    assert loss_callback.cache_folder.exists()
    assert loss_cache_path.exists()
    assert loss_cache_path in loss_callback.cache_folder.iterdir()

    loss_cache = pd.read_csv(loss_cache_path)
    _assert_loss_cache_contains_n_elements(loss_cache, n_slides_per_process * world_size)
    _assert_list_is_sorted(loss_cache[ResultsKey.LOSS].values)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_on_train_epoch_end_distributed(tmp_path: Path) -> None:
    run_distributed(test_on_train_epoch_end, [tmp_path], world_size=2)


def test_on_train_end(tmp_path: Path) -> None:
    trainer = MagicMock()
    pl_module = MagicMock(global_rank=0)
    max_epochs = 4

    loss_callback = LossAnalysisCallback(
        outputs_folder=tmp_path, max_epochs=max_epochs, num_slides_heatmap=2, num_slides_scatter=2
    )
    dump_loss_cache_for_epochs(loss_callback, max_epochs)
    loss_callback.on_train_end(trainer, pl_module)

    for epoch in range(max_epochs):
        assert (loss_callback.cache_folder / LOSS_VALUES_FILENAME.format(epoch)).exists()

    # check save_loss_ranks outputs
    assert (loss_callback.cache_folder / ALL_EPOCHS_FILENAME).exists()
    assert (loss_callback.rank_folder / LOSS_RANKS_FILENAME).exists()
    assert (loss_callback.rank_folder / LOSS_RANKS_STATS_FILENAME).exists()

    # check plot_slides_loss_scatter outputs
    assert (loss_callback.scatter_folder / SCATTER_PLOT_FILENAME.format(HIGHEST)).exists()
    assert (loss_callback.scatter_folder / SCATTER_PLOT_FILENAME.format(LOWEST)).exists()

    # check plot_loss_heatmap_for_slides_of_epoch outputs
    for epoch in range(max_epochs):
        assert (loss_callback.heatmap_folder / HEATMAP_PLOT_FILENAME.format(epoch, HIGHEST)).exists()
        assert (loss_callback.heatmap_folder / HEATMAP_PLOT_FILENAME.format(epoch, LOWEST)).exists()


def test_nans_detection(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    max_epochs = 2
    n_slides = 4
    loss_callback = LossAnalysisCallback(
        outputs_folder=tmp_path, max_epochs=2, num_slides_heatmap=2, num_slides_scatter=2
    )
    for epoch in range(max_epochs):
        loss_callback.loss_cache = get_loss_cache(n_slides)
        loss_callback.loss_cache[ResultsKey.LOSS][epoch] = np.nan
        loss_callback.save_loss_cache(epoch)

    slides_loss_values = loss_callback.select_loss_for_slides_of_epoch(epoch=0, high=None)
    loss_callback.sanity_check_loss_values(slides_loss_values)

    assert "NaNs found in loss values for slide id_0" in caplog.records[-1].getMessage()
    assert "NaNs found in loss values for slide id_1" in caplog.records[0].getMessage()

    assert loss_callback.nan_slides == ["id_1", "id_0"]
    assert loss_callback.exception_folder / NAN_SLIDES_FILENAME in loss_callback.exception_folder.iterdir()
