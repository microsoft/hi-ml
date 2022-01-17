import pytest
from unittest.mock import patch

from pytorch_lightning.core.datamodule import LightningDataModule

from health_ml.experiment_config import ExperimentConfig
from health_ml.lightning_container import LightningContainer
from health_ml.run_ml import MLRunner
from health_ml.utils.common_utils import ModelExecutionMode


@pytest.fixture
def ml_runner() -> MLRunner:
    experiment_config = ExperimentConfig()
    container = LightningContainer(num_epochs=1)
    return MLRunner(experiment_config=experiment_config, container=container)


def test_ml_runner_setup(ml_runner: MLRunner) -> None:
    """
    Check that all the necessary methods get called during setup
    """
    assert not ml_runner._has_setup_run
    with patch.object(ml_runner, "container", spec=LightningContainer) as mock_container:
        with patch("health_ml.run_ml.seed_everything") as mock_seed:
            # mock_container.get_effectie_random_seed = Mock()
            ml_runner.setup()
            mock_container.get_effective_random_seed.assert_called_once()
            mock_container.setup.assert_called_once()
            mock_container.create_lightning_module_and_store.assert_called_once()
            assert ml_runner._has_setup_run
            mock_seed.assert_called_once()


def test_is_offline_run(ml_runner: MLRunner) -> None:
    experiment_config = ExperimentConfig()
    container = LightningContainer()
    # when is_running_in_azure_ml returns True, is_offline_result should return False
    with patch("health_ml.run_ml.is_running_in_azure_ml") as mock_is_running_in_aml:
        mock_is_running_in_aml.return_value = True
        new_ml_runner = MLRunner(experiment_config=experiment_config, container=container)
        assert not new_ml_runner.is_offline_run

        # when is_running_in_azure_ml returns False, is_offline_result should return True
        mock_is_running_in_aml.return_value = False
        new_ml_runner = MLRunner(experiment_config=experiment_config, container=container)
        assert new_ml_runner.is_offline_run


def test_config_namespace(ml_runner: MLRunner) -> None:
    assert ml_runner.config_namespace == ml_runner.container.__class__.__module__ == "health_ml.lightning_container"


def test_set_run_tags_from_parent(ml_runner: MLRunner) -> None:
    with pytest.raises(AssertionError) as ae:
        ml_runner.set_run_tags_from_parent()
        assert "should only be called in a Hyperdrive run" in str(ae)

    with patch("health_ml.run_ml.PARENT_RUN_CONTEXT") as mock_parent_run_context:
        with patch("health_ml.run_ml.RUN_CONTEXT") as mock_run_context:
            mock_parent_run_context.get_tags.return_value = {"tag": "dummy_tag"}
            ml_runner.set_run_tags_from_parent()
            mock_run_context.set_tags.assert_called()


def test_run(ml_runner: MLRunner) -> None:

    def _mock_model_train(container, num_nodes):
        return "trainer", dummy_storing_logger

    dummy_storing_logger = "storing_logger"

    with patch.object(ml_runner, "setup") as mock_setup:
        with patch("health_ml.run_ml.model_train", new=_mock_model_train):
            ml_runner.run()
            mock_setup.assert_called_once()
            # expect _mock_model_train to be called and the result of ml_runner.storing_logger
            # updated accordingly
            assert ml_runner.storing_logger == dummy_storing_logger


@pytest.mark.parametrize("perform_cross_val, cross_val_split_index, should_return_true", [
    (True, 0, True),
    (True, 1, False),
    (True, 2, False),
    (False, 0, True),
    (False, 1, True),
    (False, 2, True)
])
def test_is_normal_run_or_crossval_child_0(ml_runner: MLRunner, perform_cross_val: bool, cross_val_split_index: int,
                                           should_return_true: bool) -> None:
    with patch.object(ml_runner, "container", spec=LightningContainer) as mock_container:
        mock_container.perform_cross_validation = perform_cross_val
        mock_container.crossval_split_index = cross_val_split_index
        is_run_0 = ml_runner.is_normal_run_or_crossval_child_0()

        if should_return_true:
            assert is_run_0
        else:
            assert not is_run_0


def test_lightning_data_module_dataloaders(ml_runner: MLRunner):
    data = LightningDataModule()

    dataloaders = ml_runner.lightning_data_module_dataloaders(data)
    assert len(dataloaders) == 3  # one for each of train, val, test
    assert isinstance(dataloaders, dict)
    assert dataloaders[ModelExecutionMode.TRAIN] == data.train_dataloader
    assert dataloaders[ModelExecutionMode.VAL] == data.val_dataloader
    assert dataloaders[ModelExecutionMode.TEST] == data.test_dataloader
