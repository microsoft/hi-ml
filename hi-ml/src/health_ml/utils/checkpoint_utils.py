#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
import torch
from azureml.core import Run, Workspace

from health_azure import get_workspace
from health_azure.utils import RUN_CONTEXT, download_files_from_run_id, get_run_file_names, is_global_rank_zero, \
    is_running_in_azure_ml
from health_ml.deep_learning_config import OutputParams
from health_ml.lightning_container import LightningContainer
from health_ml.utils.common_utils import AUTOSAVE_CHECKPOINT_CANDIDATES, CHECKPOINT_FOLDER, DEFAULT_AML_UPLOAD_DIR

CHECKPOINT_SUFFIX = ".ckpt"
# This is a constant that must match a filename defined in pytorch_lightning.ModelCheckpoint, but we don't want
# to import that here.
LAST_CHECKPOINT_FILE_NAME = "last"
LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX = LAST_CHECKPOINT_FILE_NAME + CHECKPOINT_SUFFIX
LEGACY_RECOVERY_CHECKPOINT_FILE_NAME = "recovery"
MODEL_INFERENCE_JSON_FILE_NAME = "model_inference_config.json"
MODEL_WEIGHTS_DIR_NAME = "trained_models"


class CheckpointHandler:
    """
    This class handles which checkpoints are used to initialize the model during train or test time based on the
    azure config and model config.
    """

    def __init__(self, container: LightningContainer,
                 project_root: Path, run_context: Optional[Run] = None):
        self.container = container
        # self.run_recovery: Optional[RunRecovery] = None
        self.project_root = project_root
        self.run_context = run_context
        self.trained_weights_paths: List[Path] = []
        self.has_continued_training = False

    @property
    def output_params(self) -> OutputParams:
        """
        Gets the part of the configuration that is responsible for output paths.
        """
        return self.container

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
            download_folder = self.output_params.checkpoint_folder / MODEL_WEIGHTS_DIR_NAME
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


def get_best_checkpoint_path(path: Path) -> Path:
    """
    Given a path and checkpoint, formats a path based on the checkpoint file name format.

    :param path to checkpoint folder
    """
    return path / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX


def download_folder_from_run_to_temp_folder(folder: str,
                                            run: Optional[Run] = None,  # type: ignore
                                            workspace: Optional[Workspace] = None) -> Path:
    """
    Downloads all files from a run that have the given prefix to a temporary folder.
    For example, if the run contains files "foo/bar.txt" and "nothing.txt", and this function is called with
    argument folder = "foo", it will return a path in a temp file system pointing to "bar.txt".

    In distributed training, the download only happens once per node.

    :param run: If provided, download the files from that run. If omitted, download the files from the current run
        (taken from RUN_CONTEXT)
    :param workspace: The AML workspace where the run is located. If omitted, the hi-ml defaults of finding a workspace
        are used (current workspace when running in AzureML, otherwise expecting a config.json file)
    :return: The path to which the files were downloaded. The files are located in that folder, without any further
        subfolders.
    """
    if not is_running_in_azure_ml() and run is None:
        raise ValueError("When running outside AzureML, the run to download from must be set.")
    run: Run = run or RUN_CONTEXT  # type: ignore
    temp_folder = Path(tempfile.mkdtemp())
    cleaned_prefix = folder.strip("/") + "/"
    existing_checkpoints = get_run_file_names(run, prefix=cleaned_prefix)
    logging.info(f"Number of checkpoints available in AzureML: {len(existing_checkpoints)}")
    if len(existing_checkpoints) > 0:
        try:
            logging.info(f"Downloading checkpoints to {temp_folder}")
            download_files_from_run_id(run_id=run.id,  # type: ignore
                                       output_folder=temp_folder,
                                       prefix=cleaned_prefix,
                                       workspace=workspace)
        except Exception as ex:
            logging.warning(f"Unable to download checkpoints from AzureML. Error: {str(ex)}")
    # Checkpoint downloads preserve the full folder structure, point the caller directly to the folder where the
    # checkpoints really are.
    return temp_folder / cleaned_prefix


def find_recovery_checkpoint_on_disk_or_cloud(path: Path) -> Optional[Path]:
    """
    Looks at all the checkpoint files and returns the path to the one that should be used for recovery.
    If no checkpoint files are found on disk, the function attempts to download from the current AzureML
    run.
    :param path: The folder to start searching in.
    :return: None if there is no suitable recovery checkpoints, or else a full path to the checkpoint file.
    """
    recovery_checkpoint = find_recovery_checkpoint(path)
    if recovery_checkpoint is None and is_running_in_azure_ml():
        logging.info(
            "No checkpoints available in the checkpoint folder. Trying to find checkpoints in AzureML.")
        # Download checkpoints from AzureML, then try to find recovery checkpoints among those.
        # Downloads should go to a temporary folder because downloading the files to the checkpoint
        # folder might
        # cause artifact conflicts later.
        temp_folder = download_folder_from_run_to_temp_folder(
            folder=f"{DEFAULT_AML_UPLOAD_DIR}/{CHECKPOINT_FOLDER}/")
        recovery_checkpoint = find_recovery_checkpoint(temp_folder)
    return recovery_checkpoint


def find_recovery_checkpoint(path: Path) -> Optional[Path]:
    """
    Finds the checkpoint file in the given path that can be used for re-starting the present job.
    This can be an autosave checkpoint, or the last checkpoint. All existing checkpoints are loaded, and the one
    for the highest epoch is used for recovery.
    :param path: The folder to search in.
    :return: Returns the checkpoint file to use for re-starting, or None if no such file was found.
    """
    legacy_recovery_checkpoints = list(path.glob(LEGACY_RECOVERY_CHECKPOINT_FILE_NAME + "*"))
    if len(legacy_recovery_checkpoints) > 0:
        logging.warning(f"Found these legacy checkpoint files: {legacy_recovery_checkpoints}")
        raise ValueError("The legacy recovery checkpoint setup is no longer supported. As a workaround, you can take "
                         f"one of the legacy checkpoints and upload as '{AUTOSAVE_CHECKPOINT_CANDIDATES[0]}'")
    candidates = [*AUTOSAVE_CHECKPOINT_CANDIDATES, LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX]
    highest_epoch: Optional[int] = None
    file_with_highest_epoch: Optional[Path] = None
    for f in candidates:
        full_path = path / f
        if full_path.is_file():
            try:
                checkpoint = torch.load(str(full_path), map_location=torch.device("cpu"))
                epoch = checkpoint["epoch"]
                logging.info(f"Checkpoint for epoch {epoch} in {full_path}")
                if (highest_epoch is None) or (epoch > highest_epoch):
                    highest_epoch = epoch
                    file_with_highest_epoch = full_path
            except Exception as ex:
                logging.warning(f"Unable to load checkpoint file {full_path}: {ex}")
    return file_with_highest_epoch


def cleanup_checkpoints(path: Path) -> None:
    """
    Remove autosave checkpoints from the given checkpoint folder, and check if a "last.ckpt" checkpoint is present.
    :param path: The folder that contains all checkpoint files.
    """
    logging.info(f"Files in checkpoint folder: {' '.join(p.name for p in path.glob('*'))}")
    last_ckpt = path / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    all_files = f"Existing files: {' '.join(p.name for p in path.glob('*'))}"
    if not last_ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint file {LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX} not found. {all_files}")
    # Training is finished now. To save storage, remove the autosave checkpoint which is now obsolete.
    # Lightning does not overwrite checkpoints in-place. Rather, it writes "autosave.ckpt",
    # then "autosave-1.ckpt" and deletes "autosave.ckpt", then "autosave.ckpt" and deletes "autosave-v1.ckpt"
    for candidate in AUTOSAVE_CHECKPOINT_CANDIDATES:
        autosave = path / candidate
        if autosave.is_file():
            autosave.unlink()
