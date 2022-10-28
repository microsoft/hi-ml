#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import logging
from azureml.core import Run
from pathlib import Path
from typing import Optional

from health_azure.utils import is_global_rank_zero
from health_ml.lightning_container import LightningContainer
from health_ml.utils.checkpoint_utils import find_recovery_checkpoint_on_disk_or_cloud


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
            self.trained_weights_path = self.get_local_checkpoints_path_or_download()
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
            logging.info(f"Available checkpoints: {len(checkpoints)}")
            for f in checkpoints:
                logging.info(f)
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

    def get_local_checkpoints_path_or_download(self) -> Path:
        """
        Get the path to the local weights to use or download them.
        """
        return self.container.src_checkpoint.get_path(self.container.checkpoint_folder)
