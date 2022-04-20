#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
from azureml.core import Run

from health_azure.utils import is_global_rank_zero
from health_ml.lightning_container import LightningContainer
from health_ml.utils.checkpoint_utils import (MODEL_WEIGHTS_DIR_NAME, find_recovery_checkpoint_on_disk_or_cloud)


class CheckpointHandler:
    """
    This class handles which checkpoints are used to initialize the model during train or test time
    """

    def __init__(self,
                 container: LightningContainer,
                 project_root: Path,
                 run_context: Optional[Run] = None):
        self.container = container
        self.project_root = project_root
        self.run_context = run_context
        self.trained_weights_paths: List[Path] = []
        self.has_continued_training = False

    def download_recovery_checkpoints_or_weights(self) -> None:
        """
        Download checkpoints from a run recovery object or from a weights url. Set the checkpoints path based on the
        run_recovery_object, weights_url or local_weights_path.
        This is called at the start of training.

        :param: only_return_path: if True, return a RunRecovery object with the path to the checkpoint without actually
        downloading the checkpoints. This is useful to avoid duplicating checkpoint download when running on multiple
        nodes. If False, return the RunRecovery object and download the checkpoint to disk.
        """
        if self.container.weights_url or self.container.local_weights_path:
            self.trained_weights_paths = self.get_local_checkpoints_path_or_download()

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
        Find the model checkpoint that should be used for inference. If a run recovery is provided, or if the model
        has been training, get the best checkpoint as defined by the container.
        If there is no run recovery and the model was
        not trained in this run, then return the checkpoint from the trained_weights_paths.
        """
        if self.has_continued_training:
            # If model was trained, look for the best checkpoint
            checkpoint_from_current_run = self.container.get_checkpoint_to_test()
            if checkpoint_from_current_run.is_file():
                logging.info(f"Using checkpoint from current run: {checkpoint_from_current_run}")
                return checkpoint_from_current_run
            else:
                raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_from_current_run}")
        elif self.trained_weights_paths:
            # Model was not trained, check if there is a local weight path.
            logging.info(f"Using pre-trained weights from {self.trained_weights_paths}")
            return self.trained_weights_paths
        raise ValueError("Unable to determine which checkpoint should be used for testing.")

    @staticmethod
    def download_weights(urls: List[str], download_folder: Path) -> List[Path]:
        """
        Download a checkpoint from weights_url to the modelweights directory.
        """
        checkpoint_paths = []
        for url in urls:
            # assign the same filename as in the download url if possible, so that we can check for duplicates
            # If that fails, map to a random uuid
            file_name = os.path.basename(urlparse(url).path) or str(uuid.uuid4().hex)
            result_file = download_folder / file_name
            checkpoint_paths.append(result_file)
            # only download if hasn't already been downloaded
            if result_file.exists():
                logging.info(f"File already exists, skipping download: {result_file}")
            else:
                logging.info(f"Downloading weights from URL {url}")

                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(result_file, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)

        return checkpoint_paths

    def get_local_checkpoints_path_or_download(self) -> List[Path]:
        """
        Get the path to the local weights to use or download them and set local_weights_path
        """
        if not self.container.local_weights_path and not self.container.weights_url:
            raise ValueError("Cannot download weights - none of local_weights_path or weights_url is set in "
                             "the model config.")

        checkpoint_paths: List[Path] = []
        if self.container.local_weights_path:
            checkpoint_paths = self.container.local_weights_path
        elif self.container.weights_url:
            download_folder = self.container.checkpoint_folder / MODEL_WEIGHTS_DIR_NAME
            download_folder.mkdir(exist_ok=True, parents=True)
            urls = self.container.weights_url
            checkpoint_paths = CheckpointHandler.download_weights(urls=urls,
                                                                  download_folder=download_folder)

        for checkpoint_path in checkpoint_paths:
            if not checkpoint_path or not checkpoint_path.is_file():
                raise FileNotFoundError(f"Could not find the weights file at {checkpoint_path}")
        return checkpoint_paths
