import pytest
import torch
import pandas as pd

from pathlib import Path
from unittest.mock import MagicMock

from health_cpath.utils.callbacks import LOSS_VALUES_FILENAME, LossAnalysisCallback, LossDictType
from health_cpath.utils.naming import ResultsKey
from testhisto.mocks.container import MockDeepSMILETilesPanda
from testhisto.utils.utils_testhisto import run_distributed


def _assert_list_is_sorted(list_: list) -> None:
    assert list_ == sorted(list_)


def _assert_loss_cache_contains_n_elements(loss_cache: LossDictType, n: int) -> None:
    for key in loss_cache:
        assert len(loss_cache[key]) == n


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


def get_loss_cache() -> LossDictType:
    return {
        ResultsKey.LOSS: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ResultsKey.SLIDE_ID: ["_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10"],
        ResultsKey.TILE_ID: ["a$b", "c$d", "e$f", "g$h", "i$j", "k$l", "m$n", "o$p", "q$r", "s$t"],
    }


def test_on_train_epoch_end(tmp_path: Path, rank: int = 0, world_size: int = 1, device: str = "cpu") -> None:
    current_epoch = 5
    n_slides_per_process = 10
    trainer = MagicMock(current_epoch=current_epoch)
    pl_module = MagicMock(global_rank=rank)

    loss_callback = LossAnalysisCallback(outputs_folder=tmp_path)
    loss_callback.loss_cache = get_loss_cache()

    print(f"Rank {rank} is running", loss_callback.cache_folder)

    _assert_loss_cache_contains_n_elements(loss_callback.loss_cache, n_slides_per_process)
    loss_callback.on_train_epoch_end(trainer, pl_module)
    # Loss cache is flushed after each epoch
    _assert_loss_cache_contains_n_elements(loss_callback.loss_cache, 0)

    loss_cache_path = loss_callback.cache_folder / LOSS_VALUES_FILENAME.format(current_epoch)
    assert loss_callback.cache_folder.exists()
    assert loss_cache_path.exists()
    assert loss_cache_path in loss_callback.cache_folder.iterdir()

    loss_cache = pd.read_csv(loss_cache_path)
    _assert_loss_cache_contains_n_elements(loss_cache, n_slides_per_process * world_size)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_on_train_epoch_end_distributed(tmp_path: Path) -> None:
    run_distributed(test_on_train_epoch_end, [tmp_path], world_size=2)
