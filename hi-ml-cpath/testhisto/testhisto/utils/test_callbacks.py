# from typing import Dict
from lib2to3 import pytree
import pytest

from pathlib import Path
from unittest.mock import MagicMock

from health_cpath.utils.callbacks import LossAnalysisCallback
from health_cpath.utils.naming import ResultsKey
from testhisto.mocks.container import MockDeepSMILETilesPanda


def _assert_list_is_sorted(list_: list) -> None:
    assert list_ == sorted(list_)


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
    for key in callback.loss_cache:
        assert len(callback.loss_cache[key]) == 0

    callback.on_train_batch_start(trainer, container.model, batch, 0, None)  # type: ignore
    for key in callback.loss_cache:
        assert len(callback.loss_cache[key]) == batch_size

    batch = next(iter(container.data_module.train_dataloader()))
    callback.on_train_batch_start(trainer, container.model, batch, 1, None)  # type: ignore
    for key in callback.loss_cache:
        assert len(callback.loss_cache[key]) == 2 * batch_size


@pytest.fixture
def loss_callback(tmp_path: Path) -> LossAnalysisCallback:
    callback = LossAnalysisCallback(outputs_folder=tmp_path, max_epochs=10)
    callback.loss_cache = {
        ResultsKey.LOSS: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ResultsKey.SLIDE_ID: ["_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10"],
        ResultsKey.TILE_ID: ["a$b", "c$d", "e$f", "g$h", "i$j", "k$l", "m$n", "o$p", "q$r", "s$t"],
    }
    return callback
