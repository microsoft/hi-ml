#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import logging
import shutil
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
            logging.info(f"Number of checkpoints in the checkpoint folder: {len(checkpoints)}")
            for f in checkpoints:
                logging.info(f.relative_to(self.container.checkpoint_folder))
        return find_recovery_checkpoint_on_disk_or_cloud(self.container.checkpoint_folder)

    def get_checkpoint_to_test(self) -> Path:
        """
        Find the model checkpoint that should be used for inference. If the model
        has been training, get the best checkpoint as defined by the container.
        If the model was not trained in this run, then return the checkpoint from the trained_weights_path.
        """
        if self.has_continued_training:
            # If model was trained, look for the best checkpoint
            checkpoint_from_current_run = self.container.get_checkpoint_to_test()
            if checkpoint_from_current_run.is_file():
                logging.info(f"Using checkpoint from current run: {checkpoint_from_current_run}")
                return checkpoint_from_current_run
            else:
                raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_from_current_run}")
        elif self.trained_weights_path:
            # Model was not trained, check if there is a local weight path.
            logging.info(f"Using pre-trained weights from {self.trained_weights_path}")
            return self.trained_weights_path
        raise ValueError("Unable to determine which checkpoint should be used for testing.")

    def get_relative_inference_checkpoint_path(self) -> Path:
        """Returns the path of the model's inference checkpoint, relative to the container's output folder."""
        expected_checkpoint_path = self.container.get_checkpoint_to_test()
        try:
            # Find the Unix-style path of the checkpoint relative to the outputs folder, this will be the path
            # where we find the file in AzureML.
            return expected_checkpoint_path.relative_to(self.container.outputs_folder)
        except ValueError:
            raise ValueError("Inference checkpoint path should be relative to the container's output folder. "
                             f"Checkpoint path: {expected_checkpoint_path}, "
                             f"output folder: {self.container.outputs_folder}")

    def download_inference_checkpoint(self, temp_folder: Optional[Path] = None) -> None:
        """
        For low-priority preemption that occured after training, try to download the inference checkpoint if that
        is available in the AzureML run from a previous incarnation of the job.
        """
        # This logic will only trigger in AzureML. Download should only happen once per node.
        if is_running_in_azure_ml() and is_global_rank_zero():
            temp_folder = temp_folder or Path(tempfile.mkdtemp())
            inference_checkpoint_azureml_path = (
                Path(DEFAULT_AML_UPLOAD_DIR) / self.get_relative_inference_checkpoint_path()
            ).as_posix()
            highest_epoch_checkpoint = download_highest_epoch_checkpoint(
                run=RUN_CONTEXT,
                checkpoint_suffix=inference_checkpoint_azureml_path,
                output_folder=temp_folder)
            if highest_epoch_checkpoint is None:
                logging.info("No inference checkpoint was found in the AzureML run.")
            else:
                destination = self.container.get_checkpoint_to_test()
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(highest_epoch_checkpoint), str(destination))
