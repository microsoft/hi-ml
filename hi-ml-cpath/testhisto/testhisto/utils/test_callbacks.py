import time
import torch
import pytest
import numpy as np
import pandas as pd

from pathlib import Path
from unittest.mock import MagicMock

from health_cpath.utils.naming import ResultsKey
from health_cpath.utils.callbacks import LossAnalysisCallback, LossCacheDictType
from testhisto.mocks.container import MockDeepSMILETilesPanda
from testhisto.utils.utils_testhisto import run_distributed


def _assert_is_sorted(array: np.ndarray) -> None:
    assert np.all(np.diff(array) <= 0)


def _assert_loss_cache_contains_n_elements(loss_cache: LossCacheDictType, n: int) -> None:
    for key in loss_cache:
        assert len(loss_cache[key]) == n


def get_loss_cache(n_slides: int = 4, rank: int = 0) -> LossCacheDictType:
    return {
        ResultsKey.LOSS: list(range(1, n_slides + 1)),
        ResultsKey.SLIDE_ID: [f"id_{i}" for i in range(rank * n_slides, (rank + 1) * n_slides)],
        ResultsKey.TILE_ID: [f"a${i * (rank + 1)}$b" for i in range(rank * n_slides, (rank + 1) * n_slides)],
    }


def dump_loss_cache_for_epochs(loss_callback: LossAnalysisCallback, epochs: int) -> None:
    for epoch in range(epochs):
        loss_callback.train_loss_cache = get_loss_cache()
        loss_callback.save_loss_cache(epoch)


def test_loss_callback_outputs_folder_exist(tmp_path: Path) -> None:
    outputs_folder = tmp_path / "outputs"
    outputs_folder.mkdir()
    callback = LossAnalysisCallback(outputs_folder=outputs_folder)
    for folder in [
        callback.outputs_folder,
        callback.get_cache_folder,
        callback.get_scatter_folder,
        callback.get_heatmap_folder,
        callback.get_anomalies_folder,
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
    assert (ResultsKey.TILE_ID in callback.train_loss_cache) == save_tile_ids


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
    _assert_loss_cache_contains_n_elements(callback.train_loss_cache, 0)

    callback.on_train_batch_start(trainer, container.model, batch, 0, None)  # type: ignore
    _assert_loss_cache_contains_n_elements(callback.train_loss_cache, batch_size)

    batch = next(iter(container.data_module.train_dataloader()))
    callback.on_train_batch_start(trainer, container.model, batch, 1, None)  # type: ignore
    _assert_loss_cache_contains_n_elements(callback.train_loss_cache, 2 * batch_size)



def test_on_train_epoch_end(
    tmp_path: Path, duplicate: bool = False, rank: int = 0, world_size: int = 1, device: str = "cpu"
) -> None:
    current_epoch = 5
    n_slides_per_process = 4
    trainer = MagicMock(current_epoch=current_epoch)
    pl_module = MagicMock(global_rank=rank)

    loss_callback = LossAnalysisCallback(outputs_folder=tmp_path, num_slides_heatmap=2, num_slides_scatter=2)
    loss_callback.train_loss_cache = get_loss_cache(rank=rank, n_slides=n_slides_per_process)
    if duplicate:
        # Duplicate slide "id_0" to test that the duplicates are removed
        loss_callback.train_loss_cache[ResultsKey.SLIDE_ID][0] = "id_0"

    _assert_loss_cache_contains_n_elements(loss_callback.train_loss_cache, n_slides_per_process)
    loss_callback.on_train_epoch_end(trainer, pl_module)
    # Loss cache is flushed after each epoch
    _assert_loss_cache_contains_n_elements(loss_callback.train_loss_cache, 0)

    if rank > 0:
        time.sleep(10)  # Wait for rank 0 to save the loss cache in a csv file

    loss_cache_path = loss_callback.get_loss_cache_file(current_epoch)
    assert loss_callback.get_cache_folder.exists()
    assert loss_cache_path.exists()
    assert loss_cache_path.parent == loss_callback.get_cache_folder

    loss_cache = pd.read_csv(loss_cache_path)
    total_slides = n_slides_per_process * world_size if not duplicate else n_slides_per_process * world_size - 1
    _assert_loss_cache_contains_n_elements(loss_cache, total_slides)
    _assert_is_sorted(loss_cache[ResultsKey.LOSS].values)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_on_train_epoch_end_distributed(tmp_path: Path) -> None:
    # Test that the loss cache is saved correctly when using multiple GPUs
    # First scenario: no duplicates
    run_distributed(test_on_train_epoch_end, [tmp_path, False], world_size=2)
    # Second scenario: introduce duplicates
    run_distributed(test_on_train_epoch_end, [tmp_path, True], world_size=2)


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
        assert loss_callback.get_loss_cache_file(epoch).exists()

    # check save_loss_ranks outputs
    assert loss_callback.get_all_epochs_loss_cache_file().exists()
    assert loss_callback.get_loss_stats_file().exists()
    assert loss_callback.get_loss_ranks_file().exists()
    assert loss_callback.get_loss_ranks_stats_file().exists()

    # check plot_slides_loss_scatter outputs
    assert loss_callback.get_scatter_plot_file(loss_callback.HIGHEST).exists()
    assert loss_callback.get_scatter_plot_file(loss_callback.LOWEST).exists()

    # check plot_loss_heatmap_for_slides_of_epoch outputs
    for epoch in range(max_epochs):
        assert loss_callback.get_heatmap_plot_file(epoch, loss_callback.HIGHEST).exists()
        assert loss_callback.get_heatmap_plot_file(epoch, loss_callback.LOWEST).exists()


def test_nans_detection(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    max_epochs = 2
    n_slides = 4
    loss_callback = LossAnalysisCallback(
        outputs_folder=tmp_path, max_epochs=max_epochs, num_slides_heatmap=2, num_slides_scatter=2
    )
    for epoch in range(max_epochs):
        loss_callback.train_loss_cache = get_loss_cache(n_slides)
        loss_callback.train_loss_cache[ResultsKey.LOSS][epoch] = np.nan
        loss_callback.save_loss_cache(epoch)

    all_slides = loss_callback.select_slides_for_epoch(epoch=0)
    all_loss_values_per_slides = loss_callback.select_all_losses_for_selected_slides(all_slides)
    loss_callback.sanity_check_loss_values(all_loss_values_per_slides)

    assert "NaNs found in loss values for slide id_0" in caplog.records[-1].getMessage()
    assert "NaNs found in loss values for slide id_1" in caplog.records[0].getMessage()

    assert loss_callback.nan_slides == ["id_1", "id_0"]
    assert loss_callback.get_nan_slides_file().exists()
    assert loss_callback.get_nan_slides_file().parent == loss_callback.get_anomalies_folder


@pytest.mark.parametrize("log_exceptions", [True, False])
def test_log_exceptions_flag(log_exceptions: bool, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    max_epochs = 3
    trainer = MagicMock()
    pl_module = MagicMock(global_rank=0)
    loss_callback = LossAnalysisCallback(
        outputs_folder=tmp_path, max_epochs=max_epochs,
        num_slides_heatmap=2, num_slides_scatter=2, log_exceptions=log_exceptions
    )
    message = "Error while detecting loss values outliers:"
    if log_exceptions:
        loss_callback.on_train_end(trainer, pl_module)
        assert message in caplog.records[-1].getMessage()
    else:
        with pytest.raises(Exception, match=fr"{message}"):
            loss_callback.on_train_end(trainer, pl_module)
