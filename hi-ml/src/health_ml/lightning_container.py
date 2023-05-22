#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import param
import logging
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive import HyperDriveConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from health_azure.utils import create_from_matching_params
from health_ml.deep_learning_config import DatasetParams, OptimizerParams, OutputParams, TrainerParams, WorkflowParams
from health_ml.experiment_config import ExperimentConfig, RunnerMode
from health_ml.utils.checkpoint_utils import get_best_checkpoint_path
from health_ml.utils.lr_scheduler import SchedulerWithWarmUp
from health_ml.utils.model_util import create_optimizer


class LightningContainer(WorkflowParams, DatasetParams, OutputParams, TrainerParams, OptimizerParams):
    """
    A LightningContainer contains all information to train a user-specified PyTorch Lightning model. The model that
    should be trained is returned by the `get_model` method. The training data must be returned in the form of
    a LightningDataModule, by the `get_data_module` method.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model: Optional[LightningModule] = None
        self._model_name = type(self).__name__
        self.trained_weights_path: Optional[Path] = None
        # Number of nodes and the runner mode are read from the ExperimentConfig, and will be copied here
        self.num_nodes = 1
        self.runner_mode = RunnerMode.TRAIN

    def validate(self) -> None:
        WorkflowParams.validate(self)
        OptimizerParams.validate(self)
        TrainerParams.validate(self)

    def setup(self) -> None:
        """
        This method is called as one of the first operations of the training/testing workflow, before any other
        operations on the present object. At the point when called, the datasets are already available in
        the locations given by self.local_datasets. Use this method to prepare datasets or data loaders, for example.
        """
        pass

    def create_model(self) -> LightningModule:  # type: ignore
        """
        This method must create the actual Lightning model that will be trained. It can read out parameters from the
        container and pass them into the model, for example.
        """
        pass

    def get_data_module(self) -> LightningDataModule:
        """
        Gets the data that is used for the training, validation, and test steps.
        This should read datasets from the self.local_datasets folder or download from a web location.
        The format of the data is not specified any further.

        :return: A LightningDataModule
        """
        return None  # type: ignore

    def get_eval_data_module(self) -> LightningDataModule:
        """
        Gets the data that is used when evaluating the model on a new dataset.
        This data module should read datasets from the self.local_datasets folder or download from a web location.
        Only the test dataloader is used, hence the method needs to put all data into the test dataloader, rather
        than splitting into train/val/test.

        :return: A LightningDataModule
        """
        return None  # type: ignore

    def get_trainer_arguments(self) -> Dict[str, Any]:
        """
        Gets additional parameters that will be passed on to the PyTorch Lightning trainer.
        """
        return dict()

    def get_callbacks(self) -> List[Callback]:
        """
        Gets additional callbacks that the trainer should use when training this model.
        """
        return []

    def get_parameter_tuning_config(self, run_config: ScriptRunConfig) -> HyperDriveConfig:  # type: ignore
        """
        Returns a configuration for hyperparameter tuning via AzureML's Hyperdrive capability.
        Hyperparameter tuning can be triggered on the commandline via the "--hyperdrive" flag.
        Override this method in your LightningContainer to use hyperparameter tuning.

        The HyperDriveConfig config object needs to specify which parameters should be searched over, and which
        metric should be monitored.

        :param run_config: The ScriptRunConfig object that needs to be passed into the constructor of
            HyperDriveConfig.
        """
        raise NotImplementedError(
            "Parameter search is not implemented. Please override 'get_parameter_tuning_config' "
            "in your model container."
        )

    def get_parameter_tuning_args(self) -> Dict[str, Any]:
        """
        Returns a dictionary of hyperperameter argument names and values as expected by a AML SDK v2 job
        to perform hyperparameter search
        """
        raise NotImplementedError(
            "Parameter search is not implemented. Please override 'get_parameter_tuning_args' "
            "in your model container."
        )

    def update_experiment_config(self, experiment_config: ExperimentConfig) -> None:
        """
        This method allows overriding ExperimentConfig parameters from within a LightningContainer.
        It is called right after the ExperimentConfig and container are initialised.
        Be careful when using class parameters to override these values. If the parameter names clash,
        CLI values will be consumed by the ExperimentConfig, but container parameters will keep their defaults.
        This can be avoided by always using unique parameter names.
        Also note that saving a reference to `experiment_config` and updating its attributes at any other
        point may lead to unexpected behaviour.

        :param experiment_config: The initialised ExperimentConfig whose parameters to override in-place.
        """
        pass

    def before_training_on_global_rank_zero(self) -> None:
        """
        A hook that will be called before starting model training, before creating the Lightning Trainer object.
        In distributed training, this is only run on global rank zero (i.e, on the process that runs on node 0, GPU 0).
        The order in which hooks are called is: before_training_on_global_rank_zero, before_training_on_local_rank_zero,
        before_training_on_all_ranks.
        """
        pass

    def before_training_on_local_rank_zero(self) -> None:
        """
        A hook that will be called before starting model training.
        In distributed training, this hook will be called once per node (i.e., whenever the LOCAL_RANK environment
        variable is zero).
        The order in which hooks are called is: before_training_on_global_rank_zero, before_training_on_local_rank_zero,
        before_training_on_all_ranks.
        """
        pass

    def before_training_on_all_ranks(self) -> None:
        """
        A hook that will be called before starting model training.
        In distributed training, this hook will be called on all ranks (i.e., once per GPU).
        The order in which hooks are called is: before_training_on_global_rank_zero, before_training_on_local_rank_zero,
        before_training_on_all_ranks.
        """
        pass

    def get_checkpoint_to_test(self) -> Path:
        """Returns the path of the model checkpoint that should be used for testing/inference. By default, this will
        return the checkpoint that is written in the last training epoch. Override this method if you implement a
        custom checkpointing logic, for example if you added a model checkpoint callback that looks at validation
        accuracy.

        :return: The path of the checkpoint file that should be used for inference.
        """
        return get_best_checkpoint_path(self.checkpoint_folder)

    # The code from here on does not need to be modified.

    @property
    def model(self) -> LightningModule:
        """
        Returns the PyTorch Lightning module that the present container object manages.

        :return: A PyTorch Lightning module
        """
        if self._model is None:
            raise ValueError("No Lightning module has been set yet.")
        return self._model

    def create_lightning_module_and_store(self) -> None:
        """
        Creates the Lightning model
        """
        self._model = self.create_model()
        if isinstance(self._model, LightningModuleWithOptimizer):
            self._model._optimizer_params = create_from_matching_params(self, OptimizerParams)
            self._model._trainer_params = create_from_matching_params(self, TrainerParams)

    def get_hyperdrive_config(self) -> Optional[HyperDriveConfig]:
        """
        Returns the HyperDrive config for either hyperparameter tuning, cross validation, or running with
        different seeds.

        :return: A configuration object for HyperDrive
        """
        if self.hyperdrive:
            return self.get_parameter_tuning_config(ScriptRunConfig(source_directory=""))
        if self.is_crossvalidation_parent_run:
            return self.get_crossval_hyperdrive_config()
        if self.different_seeds > 0:
            return self.get_different_seeds_hyperdrive_config()
        return None

    def get_hyperparam_args(self) -> Optional[Dict[str, Any]]:
        """
        Returns a dictionary of hyperparameter search arguments that will be passed to an AML v2 command to
        enable either hyperparameter tuning,  cross validation, or running with different seeds.

        :return: A dictionary of hyperparameter search arguments and values.
        """
        if self.hyperdrive:
            return self.get_parameter_tuning_args()
        if self.is_crossvalidation_parent_run:
            return self.get_crossval_hyperparam_args_v2()
        if self.different_seeds > 0:
            return self.get_grid_hyperparam_args_v2()
        return None

    def load_model_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load a checkpoint from the given path. We need to define a separate method since pytorch lightning cannot
        access the _model attribute to modify it.
        """
        if self._model is None:
            raise ValueError("No Lightning module has been set yet.")
        self._model = type(self._model).load_from_checkpoint(checkpoint_path=str(checkpoint_path))

    def __str__(self) -> str:
        """Returns a string describing the present object, as a list of key: value strings."""
        arguments_str = "\nContainer:\n"
        # Avoid callable params, the bindings that are printed out can be humongous.
        # Avoid dataframes
        skip_params = {
            name for name, value in self.param.params().items() if isinstance(value, (param.Callable, param.DataFrame))
        }
        for key, value in self.param.get_param_values():
            if key not in skip_params:
                arguments_str += f"\t{key:40}: {value}\n"
        # Print out all other separate vars that are not under the guidance of the params library,
        # skipping the two that are introduced by params
        skip_vars = {"param", "initialized"}
        for key, value in vars(self).items():
            if key not in skip_vars and key[0] != "_":
                arguments_str += f"\t{key:40}: {value}\n"
        return arguments_str

    def has_custom_test_step(self) -> bool:
        """
        Determines if the lightning module has a custom test step so that the runner can determine whether to
        run inference or skip it.
        """
        return type(self.model).test_step != LightningModule.test_step

    @property
    def effective_experiment_name(self) -> str:
        """Returns the name of the AzureML experiment that should be used. This is taken from the commandline
        argument `experiment`, falling back to the model class name if not set."""
        return self.experiment or self.model_name

    def get_additional_aml_run_tags(self) -> Dict[str, str]:
        """Returns a dictionary of tags that should be added to the AzureML run."""
        return {}

    def get_additional_environment_variables(self) -> Dict[str, str]:
        """Returns a dictionary of environment variables that should be added to the AzureML run."""
        return {}

    def on_run_extra_validation_epoch(self) -> None:
        if hasattr(self.model, "on_run_extra_validation_epoch"):
            assert self._model, "Model is not initialized."
            self.model.on_run_extra_validation_epoch()  # type: ignore
        else:
            logging.warning(
                "Hook `on_run_extra_validation_epoch` is not implemented by lightning module."
                "The extra validation epoch won't produce any extra outputs."
            )

    def set_model_variant(self, variant_name: str) -> None:
        """Choose which variant of the model to use. A variant can for example have a different number of layers
        compared to the base model. This method is called by the runner to set the variant that should be used, passing
        in the variant name. A typical implement would set parameters of the object, based on the value of the
        `variant_name` argument.

        :param variant_name: The name of the model variant that should be set.
        """
        pass


class LightningModuleWithOptimizer(LightningModule):
    """
    A base class that supplies a method to configure optimizers and LR schedulers. To use this in your model,
    inherit from this class instead of from LightningModule.
    If this class is used, all configuration options for the optimizers and LR schedulers will be also available as
    commandline arguments (for example, you can supply the hi-ml runner with "--l_rate=1e-2" to change the learning
    rate.
    """

    # These fields will be set by the LightningContainer when the model is created.
    _optimizer_params = OptimizerParams()
    _trainer_params = TrainerParams()

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        """
        This is the default implementation of the method that provides the optimizer and LR scheduler for
        PyTorch Lightning. It reads out the optimizer and scheduler settings from the model fields,
        and creates the two objects.
        Override this method for full flexibility to define any optimizer and scheduler.
        :return: A tuple of (optimizer, LR scheduler)
        """
        optimizer = create_optimizer(self._optimizer_params, self.parameters())
        l_rate_scheduler = SchedulerWithWarmUp(
            self._optimizer_params, optimizer, num_epochs=self._trainer_params.max_epochs
        )
        return [optimizer], [l_rate_scheduler]
