#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
from azureml.core import Run

from health_azure import get_workspace
from health_azure.utils import is_global_rank_zero
from health_ml.lightning_container import LightningContainer
from health_ml.utils.checkpoint_utils import (MODEL_WEIGHTS_DIR_NAME, find_recovery_checkpoint_on_disk_or_cloud,
                                              get_best_checkpoint_path, get_recovery_checkpoint_path)
from health_ml.utils.common_utils import check_properties_are_not_none


@dataclass(frozen=True)
class RunRecovery:
    """
    Class to encapsulate information relating to run recovery (eg: check point paths for parent and child runs)
    """
    checkpoints_roots: List[Path]

    def get_recovery_checkpoint_paths(self) -> List[Path]:
        return [get_recovery_checkpoint_path(x) for x in self.checkpoints_roots]

    def get_best_checkpoint_paths(self) -> List[Path]:
        return [get_best_checkpoint_path(x) for x in self.checkpoints_roots]

    def _validate(self) -> None:
        check_properties_are_not_none(self)
        if len(self.checkpoints_roots) == 0:
            raise ValueError("checkpoints_roots must not be empty")

    def __post_init__(self) -> None:
        self._validate()
        logging.info(f"Storing {len(self.checkpoints_roots)}checkpoints roots:")
        for p in self.checkpoints_roots:
            logging.info(str(p))


class CheckpointHandler:
    """
    This class handles which checkpoints are used to initialize the model during train or test time
    """

    def __init__(self,
                 container: LightningContainer,
                 project_root: Path,
                 run_context: Optional[Run] = None):
        self.container = container
        self.run_recovery: Optional[RunRecovery] = None
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
        if self.container.weights_url or self.container.local_weights_path or self.container.model_id:
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

    def get_best_checkpoint(self) -> List[Path]:
        """
        Get a list of checkpoints per epoch for testing/registration from the current training run.
        This function also checks that the checkpoint at the returned checkpoint path exists.
        """
        if not self.run_recovery and not self.has_continued_training:
            raise ValueError("Cannot recover checkpoint, no run recovery object provided and "
                             "no training has been done in this run.")

        checkpoint_paths = []
        if self.run_recovery:
            checkpoint_paths = self.run_recovery.get_best_checkpoint_paths()

            checkpoint_exists = []
            # Discard any checkpoint paths that do not exist - they will make inference/registration fail.
            # This can happen when some child runs in a hyperdrive run fail; it may still be worth running inference
            # or registering the model.
            for path in checkpoint_paths:
                if path.is_file():
                    checkpoint_exists.append(path)
                else:
                    logging.warning(f"Could not recover checkpoint path {path}")
            checkpoint_paths = checkpoint_exists

        if self.has_continued_training:
            # Checkpoint is from the current run, whether a new run or a run recovery which has been doing more
            # training, so we look for it there.
            checkpoint_from_current_run = self.container.get_checkpoint_to_test()
            if checkpoint_from_current_run.is_file():
                logging.info(f"Using checkpoint from current run: {checkpoint_from_current_run}")
                checkpoint_paths = [checkpoint_from_current_run]
            else:
                logging.info("Training has continued, but not yet written a checkpoint. Using recovery checkpoints.")
        else:
            logging.info("Using checkpoints from run recovery")

        return checkpoint_paths

    def get_checkpoint_to_test(self) -> Path:
        """
        Find the model checkpoint that should be used for inference. If a run recovery is provided, or if the model
        has been training, get the best checkpoint as defined by the container.
        If there is no run recovery and the model was
        not trained in this run, then return the checkpoint from the trained_weights_paths.
        """
        # If model was trained, look for the best checkpoint
        if self.run_recovery or self.has_continued_training:
            return self.get_best_checkpoint()
        elif self.trained_weights_paths:
            # Model was not trained, check if there is a local weight path.
            logging.info(f"Using model weights from {self.trained_weights_paths} to initialize model")
            return self.trained_weights_paths
        raise ValueError("Could not find any local_weights_path, model_weights or model_id to get checkpoints from")

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
        if not self.container.model_id and not self.container.local_weights_path and not self.container.weights_url:
            raise ValueError("Cannot download weights - none of model_id, local_weights_path or weights_url is set in "
                             "the model config.")

        checkpoint_paths: List[Path] = []
        if self.container.local_weights_path:
            checkpoint_paths = self.container.local_weights_path
        else:
            download_folder = self.container.checkpoint_folder / MODEL_WEIGHTS_DIR_NAME
            download_folder.mkdir(exist_ok=True, parents=True)

            if self.container.model_id:
                checkpoint_paths = CheckpointHandler.get_checkpoints_from_model(  # type: ignore
                    model_id=self.container.model_id,
                    workspace=get_workspace(),
                    download_path=download_folder)
            elif self.container.weights_url:
                urls = self.container.weights_url
                checkpoint_paths = CheckpointHandler.download_weights(urls=urls,
                                                                      download_folder=download_folder)

        for checkpoint_path in checkpoint_paths:
            if not checkpoint_path or not checkpoint_path.is_file():
                raise FileNotFoundError(f"Could not find the weights file at {checkpoint_path}")
        return checkpoint_paths
