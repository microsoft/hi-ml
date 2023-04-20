#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
import logging
import os
import sys
from typing import Dict

import torch
from pytorch_lightning import LightningDataModule, seed_everything

from health_azure.logging import logging_section, print_message_with_rank_pid
from health_azure.utils import (
    ENV_GLOBAL_RANK,
    ENV_LOCAL_RANK,
    ENV_NODE_RANK,
    ENV_OMPI_COMM_WORLD_RANK,
    PARENT_RUN_CONTEXT,
    RUN_CONTEXT,
    get_metrics_for_hyperdrive_run,
    get_metrics_for_run,
    is_global_rank_zero,
    is_local_rank_zero,
    is_running_in_azure_ml,
)
from health_ml.model_trainer import create_lightning_trainer, write_experiment_summary_file
from health_ml.runner_base import RunnerBase
from health_ml.utils.checkpoint_utils import cleanup_checkpoints
from health_ml.utils.common_utils import (
    change_working_directory,
)
from health_ml.utils.regression_test_utils import REGRESSION_TEST_METRICS_FILENAME, compare_folders_and_run_outputs


class TrainingRunner(RunnerBase):
    def get_data_module(self) -> LightningDataModule:
        return self.container.get_data_module()

    def get_multiple_trainloader_mode(self) -> str:
        # Workaround for a bug in PL 1.5.5: We need to pass the cycle mode for the training data as a trainer argument
        # because training data that uses a CombinedLoader is not split correctly in DDP. This flag cannot be passed
        # through the get_trainer_arguments method of the container because cycle mode is not yet available.
        multiple_trainloader_mode = "max_size_cycle"
        try:
            from SSL.data.datamodules import CombinedDataModule  # type: ignore

            if isinstance(self.data_module, CombinedDataModule):
                self.data_module.prepare_data()
                multiple_trainloader_mode = self.data_module.train_loader_cycle_mode  # type: ignore
                assert multiple_trainloader_mode, "train_loader_cycle_mode should be available now"
        except ModuleNotFoundError:
            pass
        return multiple_trainloader_mode

    def init_training(self) -> None:
        """
        Execute some bookkeeping tasks only once if running distributed and initialize the runner's trainer object.
        """
        if is_global_rank_zero():
            logging.info(f"Model checkpoints are saved at {self.container.checkpoint_folder}")
            write_experiment_summary_file(self.container, outputs_folder=self.container.outputs_folder)
            self.container.before_training_on_global_rank_zero()

        if is_local_rank_zero():
            self.container.before_training_on_local_rank_zero()
        self.container.before_training_on_all_ranks()

        # Set random seeds just before training. Ensure that dataloader workers are also seeded correctly.
        seed_everything(self.container.get_effective_random_seed(), workers=True)

        # Get the container's datamodule
        self.data_module = self.get_data_module()

        if not self.container.run_inference_only:
            checkpoint_path_for_recovery = self.checkpoint_handler.get_recovery_or_checkpoint_path_train()
            if not checkpoint_path_for_recovery and self.container.resume_training:
                # If there is no recovery checkpoint (e.g job hasn't been resubmitted) and a source checkpoint is given,
                # use it to resume training.
                checkpoint_path_for_recovery = self.checkpoint_handler.trained_weights_path

            self.trainer, self.storing_logger = create_lightning_trainer(
                container=self.container,
                resume_from_checkpoint=checkpoint_path_for_recovery,
                num_nodes=self.container.num_nodes,
                multiple_trainloader_mode=self.get_multiple_trainloader_mode(),
                azureml_run_for_logging=self.azureml_run_for_logging,
            )

            rank_info = ", ".join(
                f"{env}: {os.getenv(env)}" for env in [ENV_GLOBAL_RANK, ENV_LOCAL_RANK, ENV_NODE_RANK]
            )
            logging.info(f"Environment variables: {rank_info}. trainer.global_rank: {self.trainer.global_rank}")

    def after_ddp_cleanup(self, environ_before_training: Dict) -> None:
        """
        Run processes cleanup after ddp context to prepare for single device inference.
        Kill all processes in DDP besides rank 0.
        """

        # DDP will start multiple instances of the runner, one for each GPU. Those should terminate here after training.
        # We can now use the global_rank of the Lightning model, rather than environment variables, because DDP has set
        # all necessary properties.
        if self.container.model.global_rank != 0:
            print_message_with_rank_pid(
                f"Terminating training thread with rank {self.container.model.global_rank}.", level='INFO'
            )
            sys.exit()

        # Lightning modifies a ton of environment variables. If we first run training and then the test suite,
        # those environment variables will mislead the training runs in the test suite, and make them crash.
        # Hence, restore the original environment after training.
        os.environ.clear()
        os.environ.update(environ_before_training)

        if ENV_OMPI_COMM_WORLD_RANK in os.environ:
            del os.environ[ENV_OMPI_COMM_WORLD_RANK]

        # From the training setup, torch still thinks that it should run in a distributed manner,
        # and would block on some GPU operations. Hence, clean up distributed training.
        if torch.distributed.is_initialized():  # type: ignore
            torch.distributed.destroy_process_group()  # type: ignore

    def end_training(self, environ_before_training: Dict) -> None:
        """Cleanup after training is done. This is called after the trainer has finished fitting the data.
        This is called to update the checkpoint handler state and remove redundant checkpoint files. If running
        inference on a single device, this is also called to kill all processes besides rank 0.
        """
        # Update the checkpoint handler state
        self.checkpoint_handler.additional_training_done()
        if self.container.model.global_rank == 0:
            logging.info("Removing redundant checkpoint files.")
            cleanup_checkpoints(self.container.checkpoint_folder)

        if self.container.max_num_gpus_inference == 1:
            # Kill all processes besides rank 0 after training is done to start inference on a single device
            self.after_ddp_cleanup(environ_before_training)

    def is_crossval_disabled_or_child_0(self) -> bool:
        """
        Returns True if the present run is a non-cross-validation run, or child run 0 of a cross-validation run.
        """
        if self.container.is_crossvalidation_child_run:
            return self.container.crossval_index == 0
        return True

    def run_training(self) -> None:
        """
        The main training loop. It creates the Pytorch model based on the configuration options passed in,
        creates a Pytorch Lightning trainer, and trains the model.
        If a checkpoint was specified, then it loads the checkpoint before resuming training.
        The cwd is changed to the outputs folder so that the model can write to current working directory, and still
        everything is put into the right place in AzureML (only the contents of the "outputs" folder is treated as a
        result file).
        """
        with change_working_directory(self.container.outputs_folder):
            assert self.trainer, "Trainer should be initialized before training. Call self.init_training() first."
            self.trainer.fit(self.container.model, datamodule=self.data_module)
        for logger in self.trainer.loggers:
            assert logger is not None
            logger.finalize('success')

    def init_inference(self) -> None:
        """
        Prepare the trainer for running inference on the validation and test set. This chooses a checkpoint,
        initializes the PL Trainer object, and chooses the right data module. The hook for running
        inference on the validation set is run (`LightningContainer.on_run_extra_validation_epoch`) is first called to
        reflect any changes to the model or datamodule states before running inference.
        """
        if self.container.run_extra_val_epoch:
            logging.info("Preparing to run an extra validation epoch to evaluate the model on the validation set.")
            self.container.on_run_extra_validation_epoch()
        super().init_inference()

    def run_validation(self) -> None:
        """Run validation on the validation set for all models to save time/memory consuming outputs. This is done in
        inference only mode or when the user has requested an extra validation epoch. The cwd is changed to the outputs
        folder"""
        if self.container.run_extra_val_epoch or self.container.run_inference_only:
            with change_working_directory(self.container.outputs_folder):
                assert self.trainer, "Trainer should be initialized before validation. Call self.init_inference()."
                self.trainer.validate(
                    self.container.model, datamodule=self.data_module, ckpt_path=self.inference_checkpoint
                )
        else:
            logging.info("Skipping extra validation because the user has not requested it.")

    def run_regression_test(self) -> None:
        if self.container.regression_test_folder:
            with logging_section("Regression Test"):
                # Comparison with stored results for cross-validation runs only operates on child run 0. This run
                # has usually already downloaded the results for the other runs, and uploaded files to the parent
                # run context.
                regression_metrics_str = self.container.regression_metrics
                regression_metrics = regression_metrics_str.split(',') if regression_metrics_str else None
                # TODO: user should be able to override this value
                crossval_arg_name = self.container.CROSSVAL_INDEX_ARG_NAME

                logging.info("Comparing the current results against stored results.")
                if self.is_crossval_disabled_or_child_0():
                    if is_running_in_azure_ml():
                        if PARENT_RUN_CONTEXT is not None:
                            metrics = get_metrics_for_hyperdrive_run(
                                child_run_arg_name=crossval_arg_name,
                                run=PARENT_RUN_CONTEXT,
                                keep_metrics=regression_metrics,
                            )
                        else:
                            metrics = get_metrics_for_run(run=RUN_CONTEXT, keep_metrics=regression_metrics)

                        if metrics:
                            metrics_filename = self.container.outputs_folder / REGRESSION_TEST_METRICS_FILENAME
                            logging.info(f"Saving metrics to {metrics_filename}")
                            metrics_filename.write_text(json.dumps(metrics))

                    compare_folders_and_run_outputs(
                        expected=self.container.regression_test_folder,
                        actual=self.container.outputs_folder,
                        csv_relative_tolerance=self.container.regression_test_csv_tolerance,
                    )
                else:
                    logging.info("Skipping as this is not cross-validation child run 0.")
        else:
            logging.info("Skipping regression test, no available results to compare with.")

    def run(self) -> None:
        """
        Driver function to run a ML experiment
        """
        self.init_training()

        if not self.container.run_inference_only:
            # Backup the environment variables in case we need to run a second training in the unit tests.
            environ_before_training = dict(os.environ)

            with logging_section("Model training"):
                self.run_training()

            self.end_training(environ_before_training)

        self.init_inference()

        with logging_section("Model validation"):
            self.run_validation()

        with logging_section("Model inference"):
            self.run_inference()

        self.run_regression_test()
