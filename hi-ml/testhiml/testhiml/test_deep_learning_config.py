#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from random import randint
from unittest.mock import patch, MagicMock

import pytest
from _pytest.logging import LogCaptureFixture

from param import Number
from pathlib import Path

from health_ml.deep_learning_config import DatasetParams, WorkflowParams, OutputParams, OptimizerParams, \
    ExperimentFolderHandler, TrainerParams
from health_ml.utils.checkpoint_utils import CheckpointParser
from testhiml.utils.fixed_paths_for_tests import full_test_data_path, mock_run_id


def test_validate_workflow_params_for_inference_only() -> None:
    with pytest.raises(ValueError, match=r"Cannot run inference without a src_checkpoint."):
        WorkflowParams(local_datasets=Path("foo"), run_inference_only=True).validate()

    full_file_path = full_test_data_path(suffix="hello_world_checkpoint.ckpt")
    run_id = mock_run_id(id=0)
    WorkflowParams(local_dataset=Path("foo"), run_inference_only=True,
                   src_checkpoint=CheckpointParser(run_id)).validate()
    WorkflowParams(local_dataset=Path("foo"), run_inference_only=True,
                   src_checkpoint=CheckpointParser(f"{run_id}:best_val_loss.ckpt")).validate()
    WorkflowParams(local_dataset=Path("foo"), run_inference_only=True,
                   src_checkpoint=CheckpointParser(f"{run_id}:custom/path/model.ckpt")).validate()
    WorkflowParams(local_dataset=Path("foo"), run_inference_only=True,
                   src_checkpoint=CheckpointParser(str(full_file_path))).validate()


def test_validate_workflow_params_for_resume_training() -> None:
    with pytest.raises(ValueError, match=r"Cannot resume training without a src_checkpoint."):
        WorkflowParams(local_datasets=Path("foo"), resume_training=True).validate()

    full_file_path = full_test_data_path(suffix="hello_world_checkpoint.ckpt")
    run_id = mock_run_id(id=0)
    WorkflowParams(local_dataset=Path("foo"), resume_training=True,
                   src_checkpoint=CheckpointParser(run_id)).validate()
    WorkflowParams(local_dataset=Path("foo"), resume_training=True,
                   src_checkpoint=CheckpointParser(f"{run_id}:best_val_loss.ckpt")).validate()
    WorkflowParams(local_dataset=Path("foo"), resume_training=True,
                   src_checkpoint=CheckpointParser(f"{run_id}:custom/path/model.ckpt")).validate()
    WorkflowParams(local_dataset=Path("foo"), resume_training=True,
                   src_checkpoint=CheckpointParser(str(full_file_path))).validate()


@pytest.mark.fast
def test_workflow_params_get_effective_random_seed() -> None:
    seed0 = 123
    params = WorkflowParams(random_seed=seed0)
    assert params.get_effective_random_seed() == seed0
    params.crossval_count = 5
    params.crossval_index = None
    assert params.is_crossvalidation_parent_run
    assert not params.is_crossvalidation_child_run
    assert params.get_effective_random_seed() == seed0
    params.crossval_count = 5
    params.crossval_index = 0
    assert not params.is_crossvalidation_parent_run
    assert params.is_crossvalidation_child_run
    assert params.get_effective_random_seed() != seed0


@pytest.mark.fast
def test_validate_dataset_params() -> None:
    # DatasetParams cannot be initialized with neither of azure_datasets or local_datasets set
    with pytest.raises(ValueError) as ex:
        DatasetParams(local_datasets=[], azure_datasets=[]).validate()
    assert ex.value.args[0] == "Either local_datasets or azure_datasets must be set."

    # If azure_datasets or local_datasets is not a list an exception should be raised
    with pytest.raises(Exception) as e:
        DatasetParams(local_datasets="", azure_datasets=[]).validate()
    assert "must be a list" in str(e)

    with pytest.raises(Exception) as e:
        DatasetParams(local_datasets=[], azure_datasets=None).validate()
    assert "must be a list" in str(e)

    # local datasets and dataset_mountpoints must be Paths
    with pytest.raises(Exception) as e:
        DatasetParams(local_datasets=["foo"])
    assert "items must be instances of type <class 'pathlib.Path'>" in str(e)

    with pytest.raises(Exception) as e:
        DatasetParams(dataset_mountpoints=["foo"])
    assert "items must be instances of type <class 'pathlib.Path'>" in str(e)

    # The following should be okay
    DatasetParams(local_datasets=[Path("foo")]).validate()
    DatasetParams(azure_datasets=["bar"]).validate()

    config = DatasetParams(local_datasets=[Path("foo")],
                           azure_datasets=[""])
    config.validate()
    assert config.azure_datasets == [""]

    config = DatasetParams(azure_datasets=["foo"])
    config.validate()
    assert len(config.azure_datasets) == 1

    config = DatasetParams(local_datasets=[Path("foo")],
                           azure_datasets=[""])
    config.validate()
    assert len(config.azure_datasets) == 1

    config = DatasetParams(azure_datasets=["foo", "bar"])
    config.validate()
    assert len(config.azure_datasets) == 2

    config = DatasetParams(azure_datasets=["foo"],
                           dataset_mountpoints=[Path()])
    config.validate()
    assert config.dataset_mountpoints == [Path()]

    config = DatasetParams(azure_datasets=["foo"],
                           dataset_mountpoints=[Path("foo")])
    config.validate()
    assert len(config.dataset_mountpoints) == 1

    # the number of mountpoints must not be larger than the number of datasets
    with pytest.raises(ValueError) as e:
        DatasetParams(azure_datasets=["foo"],
                      dataset_mountpoints=[Path("foo"), Path("bar")]).validate()
    assert "Expected the number of azure datasets to equal the number of mountpoints" in str(e)


def test_output_params_set_output_to() -> None:
    # output_to must be Path type
    with pytest.raises(Exception) as e:
        OutputParams(output_to="foo")
    assert "must be an instance of Path" in str(e)

    old_path = Path()
    config = OutputParams(output_to=old_path)
    assert config.outputs_folder == old_path
    new_path = Path("dummy")
    config.set_output_to(new_path)
    # create_filesystem gets called inside
    assert config.output_to == new_path


def test_output_params_create_filesystem(tmp_path: Path) -> None:
    # file_system_config must be of type ExperimentFolderHandler
    with pytest.raises(Exception) as e:
        OutputParams(file_system_config="foo")
    assert "value must be an instance of ExperimentFolderHandler" in str(e)

    config = OutputParams()
    default_file_system_config = config.file_system_config
    assert isinstance(default_file_system_config, ExperimentFolderHandler)
    assert default_file_system_config.project_root == Path(".")
    # Now call create_filesystem with a different path project_root
    config.create_filesystem(tmp_path)
    new_file_system_config = config.file_system_config
    assert new_file_system_config.project_root == tmp_path


def test_validate_optimizer_params() -> None:
    # Instantiating OptimizerParams with no non-default values should be ok
    config = OptimizerParams()
    config.validate()

    # assert that passing a string to a param expecting a numeric value causes an Exception to be raised
    numeric_params = [k for k, v in config.params().items() if isinstance(v, Number)]
    for numeric_param_name in numeric_params:
        with pytest.raises(Exception) as e:
            config = OptimizerParams()
            setattr(config, numeric_param_name, "foo")
            config.validate()

    # For non-numeric parametes, check that Exceptions are raised when params with invalid types are provided
    with pytest.raises(Exception) as e:
        OptimizerParams(l_rate_scheduler="foo").validate()
    assert "must be an instance of LRSchedulerType" in str(e)

    with pytest.raises(Exception) as e:
        OptimizerParams(l_rate_multi_step_milestones="foo")
    assert "must be a list" in str(e)

    with pytest.raises(Exception) as e:
        OptimizerParams(l_rate_warmup="foo").validate()
    assert "must be an instance of LRWarmUpType" in str(e)

    with pytest.raises(Exception) as e:
        OptimizerParams(optimizer_type="foo").validate()
    assert "must be an instance of OptimizerType" in str(e)

    with pytest.raises(Exception) as e:
        OptimizerParams(adam_betas="foo").validate()
    assert "only takes a tuple value" in str(e)


def test_optimizer_params_min_l_rate() -> None:
    config = OptimizerParams()
    min_l_rate = config.min_l_rate
    assert min_l_rate == config._min_l_rate


def test_trainer_params_use_gpu() -> None:
    config = TrainerParams()
    for patch_gpu in [False, True]:
        with patch("health_ml.utils.common_utils.is_gpu_available") as mock_gpu_available:
            mock_gpu_available.return_value = patch_gpu
            assert config.use_gpu is patch_gpu


def test_trainer_params_max_num_gpus_inference(caplog: LogCaptureFixture) -> None:
    config = TrainerParams(max_num_gpus_inference=2, pl_replce_sampler_ddp=True)
    config.validate()
    assert config.max_num_gpus_inference == 2
    assert config.pl_replace_sampler_ddp is True
    assert "The 'pl_replace_sampler_ddp' flag is set to True, but the 'max_num_gpus_inference'" in caplog.messages[-1]


@patch("health_ml.utils.common_utils.is_gpu_available")
def test_trainer_params_num_gpus_per_node(mock_gpu_available: MagicMock, caplog: LogCaptureFixture) -> None:
    mock_gpu_available.return_value = True
    # if the requested number of gpus is available and less than the total available number of gpus, a warning
    # should be logged to let the user know that they aren't using the full capacity
    requested_gpus = 3
    config = TrainerParams(max_num_gpus=requested_gpus)
    random_num_available_gpus = randint(requested_gpus, requested_gpus + 5)
    with patch("torch.cuda.device_count") as mock_gpu_count:
        mock_gpu_count.return_value = random_num_available_gpus
        assert config.num_gpus_per_node() == requested_gpus
        message = caplog.messages[-1]
        assert f"Restricting the number of GPUs to {requested_gpus}" in message

    # if the max number of gpus is set as less than the number available, expect a warning
    requested_gpus = 3
    random_num_available_gpus = randint(1, requested_gpus - 1)
    with patch("torch.cuda.device_count") as mock_gpu_count:
        mock_gpu_count.return_value = random_num_available_gpus
        config = TrainerParams(max_num_gpus=requested_gpus)
        assert config.num_gpus_per_node() == random_num_available_gpus
        message = caplog.messages[-1]
        assert f"You requested max_num_gpus {requested_gpus} but there are only {random_num_available_gpus}" \
               f" available." in message
