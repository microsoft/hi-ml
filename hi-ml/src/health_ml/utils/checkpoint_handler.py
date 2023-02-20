#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import logging
import tempfile
from azureml.core import Run
from pathlib import Path
from typing import Optional

from health_azure import RUN_CONTEXT
from health_azure.utils import is_global_rank_zero, is_running_in_azure_ml
from health_ml.utils.common_utils import DEFAULT_AML_UPLOAD_DIR
from health_ml.lightning_container import LightningContainer
from health_ml.utils.checkpoint_utils import (
    download_highest_epoch_checkpoint,
    find_recovery_checkpoint_on_disk_or_cloud
)


class CheckpointHandler:
    """
    This class handles which checkpoints are used to initialize the model during train or test time
    """

    def __init__(self, container: LightningContainer, project_root: Path, run_context: Optional[Run] = None):
        self.container = container
        self.project_root = project_root
        self.run_context = run_context
        self.trained_weights_path: Optional[Path] = None
        self.has_continued_training = False

    def download_recovery_checkpoints_or_weights(self) -> None:
        """
        Download checkpoints from a run recovery object or from a given checkpoint. Set the checkpoints path based on
        the checkpoint_url, local_checkpoint or checkpoint from an azureml run id.
        This is called at the start of training.
        """
        if self.container.src_checkpoint:
            self.trained_weights_path = self.container.src_checkpoint.get_or_download_checkpoint(
                download_dir=self.container.checkpoint_folder)
            self.container.trained_weights_path = self.trained_weights_path

    def additional_training_done(self) -> None:
        """
        Lets the handler know that training was done in this run.
        """
        self.has_continued_training = True

    def get_recovery_or_checkpoint_path_train(self) -> Optional[Path]:
        """
        Decides the checkpoint path to use for the current training run. Looks for the latest checkpoint in the
        checkpoint folder. If run_recovery is provided, the checkpoints will have been downloaded to this folder
        prior to calling this function. Else, if the run gets pre-empted and automatically restarted in AML,
        the latest checkpoint will be present in this folder too.

        :return: Constructed checkpoint path to recover from.
        """
        if is_global_rank_zero():
            checkpoints = list(self.container.checkpoint_folder.rglob("*"))
            logging.info(f"There are {len(checkpoints)} checkpoints in the checkpoint folder:")
            for f in checkpoints:
                logging.info(f.relative_to(self.container.checkpoint_folder))
        return find_recovery_checkpoint_on_disk_or_cloud(self.container.checkpoint_folder)

    def get_checkpoint_to_test(self) -> Path:
        """
        Find the model checkpoint that should be used for inference.

        If the model has been doing training epochs, get the best checkpoint as defined by the container.
        It is possible that the best checkpoint is not available on disk because the job got preempted. In those
        cases, try to find the inference checkpoint by going through all inference checkpoints stored in the AzureML
        run, downloading them and finding the one that has the highest epoch number (that must be the most recent
        among the possibly multiple retry results).

        If the model has not been doing training, but is set up to use a pre-trained
        set of weights in `trained_weights_path`, return that.
        """
        if self.has_continued_training:
            # If model was trained, look for the best checkpoint
            checkpoint_from_current_run = self.container.get_checkpoint_to_test()
            if checkpoint_from_current_run.is_file():
                logging.info(f"Using checkpoint from current run: {checkpoint_from_current_run}")
                return checkpoint_from_current_run
            logging.warning(f"No inference checkpoint found from the current run: {checkpoint_from_current_run}")
            logging.info("Trying to find an inference checkpoint in AzureML.")
            downloaded_checkpoint = self.download_inference_checkpoint()
            if downloaded_checkpoint is not None:
                logging.info(f"Using a checkpoint found in the AzureML run: {downloaded_checkpoint}")
                return downloaded_checkpoint
            raise FileNotFoundError("No inference checkpoint file found locally nor in AzureML.")
        elif self.trained_weights_path:
            # Model was not trained, check if there is a local weight path.
            logging.info(f"Using pre-trained weights from {self.trained_weights_path}")
            return self.trained_weights_path
        raise ValueError("Unable to determine which checkpoint should be used for testing.")

    def get_relative_inference_checkpoint_path(self) -> Path:
        """Returns the path of the model's inference checkpoint, relative to the container's output folder.

        This will be the path where the inference checkpoint can be found in the AzureML run output (except
        the `outputs` prefix that has to be added at the start)."""
        expected_checkpoint_path = self.container.get_checkpoint_to_test()
        try:
            return expected_checkpoint_path.relative_to(self.container.outputs_folder)
        except ValueError:
            raise ValueError("Inference checkpoint path should be relative to the container's output folder. "
                             f"Checkpoint path: {expected_checkpoint_path}, "
                             f"output folder: {self.container.outputs_folder}")

    def download_inference_checkpoint(self, download_folder: Optional[Path] = None) -> Optional[Path]:
        """
        For low-priority preemption that occured after training, try to download the inference checkpoint if that
        is available in the AzureML run from a previous incarnation of the job. Downloads go into the `download_folder`.

        The method returns None if no checkpoint was found, or if the
        current code is executing outside of AzureML and hence can't access previous inference checkpoints in AzureML.

        :param download_folder: The folder where the checkpoints should be downloaded to. If not provided, use a
            temp folder.
        :return: The path to a downloaded inference checkpoint, or None if no checkpoint was available.
        """
        # This logic will only trigger in AzureML. Download should only happen once per node.
        if is_running_in_azure_ml() and is_global_rank_zero():
            download_folder = download_folder or Path(tempfile.mkdtemp())
            inference_checkpoint_azureml_path = (
                Path(DEFAULT_AML_UPLOAD_DIR) / self.get_relative_inference_checkpoint_path()
            ).as_posix()
            highest_epoch_checkpoint = download_highest_epoch_checkpoint(
                run=RUN_CONTEXT,
                checkpoint_suffix=inference_checkpoint_azureml_path,
                output_folder=download_folder)
            if highest_epoch_checkpoint is None:
                logging.info("No inference checkpoint was found in the AzureML run.")
                return None
            logging.info("An inference checkpoint was found in the AzureML run.")
            return highest_epoch_checkpoint
        return None
