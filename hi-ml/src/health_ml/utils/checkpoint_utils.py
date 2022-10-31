#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import re
import os
import uuid
import torch
import logging
import tempfile
import requests

from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from azureml.core import Run, Workspace
from health_azure import download_checkpoints_from_run_id, get_workspace
from health_azure.utils import (RUN_CONTEXT, download_files_from_run_id, get_run_file_names, is_running_in_azure_ml)
from health_ml.utils.common_utils import (AUTOSAVE_CHECKPOINT_CANDIDATES, DEFAULT_AML_CHECKPOINT_DIR, CHECKPOINT_SUFFIX)
from health_ml.utils.type_annotations import PathOrString

# This is a constant that must match a filename defined in pytorch_lightning.ModelCheckpoint, but we don't want
# to import that here.
LAST_CHECKPOINT_FILE_NAME = f"last{CHECKPOINT_SUFFIX}"
LEGACY_RECOVERY_CHECKPOINT_FILE_NAME = "recovery"
MODEL_INFERENCE_JSON_FILE_NAME = "model_inference_config.json"
MODEL_WEIGHTS_DIR_NAME = "pretrained_models"


def get_best_checkpoint_path(path: Path) -> Path:
    """
    Given a path and checkpoint, formats a path based on the checkpoint file name format.

    :param path to checkpoint folder
    """
    return path / LAST_CHECKPOINT_FILE_NAME


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
        temp_folder = download_folder_from_run_to_temp_folder(folder=DEFAULT_AML_CHECKPOINT_DIR)
        recovery_checkpoint = find_recovery_checkpoint(temp_folder)
    return recovery_checkpoint


def get_recovery_checkpoint_path(path: Path) -> Path:
    """
    Returns the path to the last recovery checkpoint in the given folder or the provided filename. Raises a
    FileNotFoundError if no recovery checkpoint file is present.
    :param path: Path to checkpoint folder
    """
    recovery_checkpoint = find_recovery_checkpoint(path)
    if recovery_checkpoint is None:
        files = [f.name for f in path.glob("*")]
        raise FileNotFoundError(f"No checkpoint files found in {path}. Existing files: {' '.join(files)}")
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
    candidates = [*AUTOSAVE_CHECKPOINT_CANDIDATES, LAST_CHECKPOINT_FILE_NAME]
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


def cleanup_checkpoints(ckpt_folder: Path) -> None:
    """
    Remove autosave checkpoints from the given checkpoint folder, and check if a "last.ckpt" checkpoint is present.
    :param ckpt_folder: The folder that contains all checkpoint files.
    """
    files_in_checkpoint_folder = [p.name for p in ckpt_folder.glob('*')]
    if len(files_in_checkpoint_folder) == 0:
        return
    logging.info(f"Files in checkpoint folder: {' '.join(files_in_checkpoint_folder)}")
    last_ckpt = ckpt_folder / LAST_CHECKPOINT_FILE_NAME
    all_files = f"Existing files: {' '.join(p.name for p in ckpt_folder.glob('*'))}"
    if not last_ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint file {LAST_CHECKPOINT_FILE_NAME} not found. {all_files}")
    # Training is finished now. To save storage, remove the autosave checkpoint which is now obsolete.
    # Lightning does not overwrite checkpoints in-place. Rather, it writes "autosave.ckpt",
    # then "autosave-1.ckpt" and deletes "autosave.ckpt", then "autosave.ckpt" and deletes "autosave-v1.ckpt"
    for candidate in AUTOSAVE_CHECKPOINT_CANDIDATES:
        autosave = ckpt_folder / candidate
        if autosave.is_file():
            autosave.unlink()


class AMLCheckpointDownloader:
    def __init__(
        self,
        run_id: str,
        checkpoint_filename: Optional[str] = None,
        download_dir: PathOrString = "checkpoints",
        remote_checkpoint_dir: Optional[Path] = None,
    ) -> None:
        """
        Utility class for downloading checkpoint files from an Azure ML run

        :param run_id: Recovery ID of the run from which to load the checkpoint.
        :param checkpoint_filename: Name of the checkpoint file, expected to be inside the
        `outputs/checkpoints/` directory (e.g. `"best_checkpoint.ckpt"`).
        :param download_dir: The local directory in which to save the downloaded checkpoint files.
        :param remote_checkpoint_dir: The remote folder from which to download the checkpoint file
        """
        self.run_id = run_id
        self.checkpoint_filename = checkpoint_filename or self.extract_checkpoint_filename_from_run_id()
        self.download_dir = Path(download_dir)
        self.remote_checkpoint_dir = (
            remote_checkpoint_dir or self.extract_remote_checkpoint_dir_from_checkpoint_filename()
        )

    def extract_checkpoint_filename_from_run_id(self) -> str:
        """
        Extracts the checkpoint filename from the run_id if run_id is in the format
        <MyContainer_xx>:<checkpoint_filename.ckpt>. Otherwise, uses the last epoch checkpoint filename as default.
        """
        run_id_split = self.run_id.split(":")
        self.run_id = run_id_split[0]
        return run_id_split[-1] if len(run_id_split) > 1 else LAST_CHECKPOINT_FILE_NAME

    def extract_remote_checkpoint_dir_from_checkpoint_filename(self) -> Path:
        """
        Extracts the remote checkpoint directory from the checkpoint filename if checkpoint_filename is in the format
        <custom/patch/checkpoint_filename.ckpt>. Otherwise, uses the default remote checkpoint directory
        'outputs/checkpoints/'.
        """
        tmp_checkpoint_filename = self.checkpoint_filename
        checkpoint_filename_split = self.checkpoint_filename.split("/")
        self.checkpoint_filename = checkpoint_filename_split[-1]
        return (
            Path(tmp_checkpoint_filename).parent
            if len(checkpoint_filename_split) > 1
            else Path(DEFAULT_AML_CHECKPOINT_DIR)
        )

    @property
    def local_checkpoint_dir(self) -> Path:
        return self.download_dir / self.run_id

    @property
    def remote_checkpoint_path(self) -> Path:
        assert self.checkpoint_filename is not None
        return self.remote_checkpoint_dir / self.checkpoint_filename

    @property
    def local_checkpoint_path(self) -> Path:
        return self.local_checkpoint_dir / self.remote_checkpoint_path

    def download_checkpoint_if_necessary(self) -> None:
        """Downloads the specified checkpoint if it does not already exist. """

        if not self.local_checkpoint_path.exists():
            workspace = get_workspace()
            self.local_checkpoint_dir.mkdir(exist_ok=True, parents=True)
            download_checkpoints_from_run_id(
                self.run_id, str(self.remote_checkpoint_path), self.local_checkpoint_dir, aml_workspace=workspace
            )
            assert self.local_checkpoint_path.exists(), f"Couln't download checkpoint from run {self.run_id}."


class CheckpointParser:
    AML_RUN_ID_FORMAT = (f"<AzureML_run_id>:<optional/custom/path/to/checkpoints/><filename{CHECKPOINT_SUFFIX}>"
                         f"If no custom path is provided (e.g., <AzureML_run_id>:<filename{CHECKPOINT_SUFFIX}>)"
                         "the checkpoint will be downloaded from the default checkpoint folder "
                         f"(e.g., '{DEFAULT_AML_CHECKPOINT_DIR}') If no filename is provided, "
                         "(e.g., `src_checkpoint=<AzureML_run_id>`) the latest checkpoint "
                         f"({LAST_CHECKPOINT_FILE_NAME}) will be used to initialize the model.")
    INFO_MESSAGE = ("Please provide a valid checkpoint path, URL or AzureML run ID. For custom checkpoint paths "
                    f"within an azureml run, provide a checkpoint in the format {AML_RUN_ID_FORMAT}.")
    DOC = ("We currently support three types of checkpoints: "
           "    a. A local checkpoint folder that contains a checkpoint file."
           "    b. A URL to a remote checkpoint to be downloaded."
           "    c. A previous azureml run id where the checkpoint is supposed to be "
           "       saved ('outputs/checkpoints/' folder by default.)"
           f"For the latter case 'c' : src_checkpoint should be in the format of {AML_RUN_ID_FORMAT}")

    def __init__(self, checkpoint: str = "") -> None:
        self.checkpoint = checkpoint
        self.validate()

    @property
    def is_url(self) -> bool:
        try:
            result = urlparse(self.checkpoint)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @property
    def is_local_file(self) -> bool:
        return Path(self.checkpoint).is_file()

    @property
    def is_aml_run_id(self) -> bool:
        match = re.match(r"[_\w-]*$", self.checkpoint.split(":")[0])
        return match is not None and not self.is_url and not self.is_local_file

    @property
    def is_valid(self) -> bool:
        if self.checkpoint:
            return self.is_local_file or self.is_url or self.is_aml_run_id
        return True

    def validate(self) -> None:
        if not self.is_valid:
            raise ValueError(f"Invalid checkpoint '{self.checkpoint}'. {self.INFO_MESSAGE}")

    @staticmethod
    def download_weights(url: str, download_folder: Path) -> Path:
        """
        Download a checkpoint from checkpoint_url to the modelweights directory. The file name is determined from
        from the file name in the URL. If that can't be determined, use a random file name.

        :param url: The URL from which the weights should be downloaded.
        :param download_folder: The target folder for the download.
        :return: A path to the downloaded file.
        """
        # assign the same filename as in the download url if possible, so that we can check for duplicates
        # If that fails, map to a random uuid
        file_name = os.path.basename(urlparse(url).path) or str(uuid.uuid4().hex)
        checkpoint_path = download_folder / file_name
        # only download if hasn't already been downloaded
        if checkpoint_path.is_file():
            logging.info(f"File already exists, skipping download: {checkpoint_path}")
        else:
            logging.info(f"Downloading weights from URL {url}")

            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(checkpoint_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
        return checkpoint_path

    def get_path(self, checkpoints_folder: Path) -> Path:
        if self.is_local_file:
            checkpoint_path = Path(self.checkpoint)
        elif self.is_url:
            download_folder = checkpoints_folder / MODEL_WEIGHTS_DIR_NAME
            download_folder.mkdir(exist_ok=True, parents=True)
            checkpoint_path = self.download_weights(url=self.checkpoint, download_folder=download_folder)
        elif self.is_aml_run_id:
            downloader = AMLCheckpointDownloader(run_id=self.checkpoint, download_dir=checkpoints_folder)
            downloader.download_checkpoint_if_necessary()
            checkpoint_path = downloader.local_checkpoint_path
        else:
            raise ValueError("Unable to determine how to get the checkpoint path.")

        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Could not find the weights file at {checkpoint_path}")
        return checkpoint_path
