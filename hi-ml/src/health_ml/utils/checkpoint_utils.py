#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import re
import os
import uuid
import logging
import tempfile
import requests

import torch

from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse
from azureml.core import Run

from health_azure import download_checkpoints_from_run_id, get_workspace, torch_barrier
from health_azure.utils import (
    RUN_CONTEXT,
    _download_files_from_run,
    get_run_file_names,
    is_running_in_azure_ml,
    is_local_rank_zero,
    download_files_by_suffix)
from health_ml.utils.common_utils import (
    AUTOSAVE_CHECKPOINT_CANDIDATES,
    CHECKPOINT_FOLDER,
    DEFAULT_AML_CHECKPOINT_DIR,
    CHECKPOINT_SUFFIX,
    DEFAULT_AML_UPLOAD_DIR)
from health_ml.utils.type_annotations import PathOrString

# This is a constant that must match a filename defined in pytorch_lightning.ModelCheckpoint, but we don't want
# to import that here.
LAST_CHECKPOINT_FILE_NAME = f"last{CHECKPOINT_SUFFIX}"
LEGACY_RECOVERY_CHECKPOINT_FILE_NAME = "recovery"
MODEL_INFERENCE_JSON_FILE_NAME = "model_inference_config.json"
MODEL_WEIGHTS_DIR_NAME = "pretrained_models"

# The dictionary field where PyTorch Lightning stores the epoch number in the checkpoint file.
CHECKPOINT_EPOCH_KEY = "epoch"

# The string that is used to prefix the folders with results from retries after low priority preemption in AzureML.
AZUREML_RETRY_PREFIX = "retry_"


def get_best_checkpoint_path(path: Path) -> Path:
    """
    Given a path and checkpoint, formats a path based on the checkpoint file name format.

    :param path to checkpoint folder
    """
    return path / LAST_CHECKPOINT_FILE_NAME


def download_folder_from_run_to_temp_folder(folder: str,
                                            run: Optional[Run] = None,  # type: ignore
                                            run_barrier: bool = True) -> Path:
    """
    Downloads all files from a run that have the given prefix to a temporary folder.
    For example, if the run contains files "foo/bar.txt" and "nothing.txt", and this function is called with
    argument folder = "foo", it will return a path in a temp file system pointing to "bar.txt".

    In distributed training, the download only happens once per node, on local rank zero. All ranks wait for the
    completion of the downloads, by calling `torch.barrier()`. This is to avoid multiple ranks trying to download
    the same files at the same time. Calling the barrier can be skipped by setting `run_barrier` to False.

    :param folder: The folder to download. This is the prefix of the files to download.
    :param run: If provided, download the files from that run. If omitted, download the files from the current run
        (taken from RUN_CONTEXT)
    :param run_barrier: If True, call torch.barrier() after the download.
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
            _download_files_from_run(
                run=run,  # type: ignore
                output_dir=temp_folder,
                prefix=cleaned_prefix)
        except Exception as ex:
            logging.warning(f"Unable to download checkpoints from AzureML. Error: {str(ex)}")
    # Checkpoint downloads preserve the full folder structure, point the caller directly to the folder where the
    # checkpoints really are.
    if run_barrier:
        torch_barrier()
    return temp_folder / cleaned_prefix


def _get_checkpoint_files(files: List[str]) -> List[str]:
    """Filters a list of files in an AzureML run to only retain those that could be recovery or last checkpoints.
    This takes the folder structures for retries into account, where files are written into subfolders like
    `retry_001`.

    :param files: The list of file names to check.
    :return: A list of file names that could be recovery checkpoints."""
    folder_pattern = f"{DEFAULT_AML_UPLOAD_DIR}(/{AZUREML_RETRY_PREFIX}[0-9]" + "{3})?" + f"/{CHECKPOINT_FOLDER}/"
    checkpoint_filenames = (*AUTOSAVE_CHECKPOINT_CANDIDATES, LAST_CHECKPOINT_FILE_NAME)
    file_pattern = "|".join(f"({filename})" for filename in checkpoint_filenames)
    pattern = f"{folder_pattern}({file_pattern})"
    regex = re.compile(pattern)
    return [f for f in files if regex.match(f)]


def download_checkpoints_from_run(run: Run, tmp_folder: Optional[Path] = None) -> Path:
    """This function will download all checkpoints from the current AzureML run to a temporary folder. It will take
    into account cases where a job got pre-empted, and wrote checkpoints to subfolders like `retry_001`.

    In distributed training, the download only happens once per node, on local rank zero. All ranks wait for the
    completion of the downloads, by calling `torch.barrier()`. This is to avoid multiple ranks trying to download
    the same files at the same time.

    :param run: The AzureML run to download the checkpoints from.
    :param tmp_folder: The folder to download the checkpoints to. If None, a temporary folder will be created.
    :return: The folder to which the checkpoints were downloaded."""
    tmp_folder = tmp_folder or Path(tempfile.mkdtemp())
    if is_local_rank_zero():
        for file in _get_checkpoint_files(run.get_file_names()):
            logging.info(f"Downloading checkpoint file {file} to temp folder")
            run.download_file(file, output_file_path=str(tmp_folder / file))
    torch_barrier()
    return tmp_folder


def download_highest_epoch_checkpoint(run: Run, checkpoint_suffix: str, output_folder: Path) -> Optional[Path]:
    """Downloads all checkpoint files from the run that have a given suffix, and returns the local path to the
    downloaded checkpoint with the highest epoch. Checkpoints are downloaded one-by-one and deleted right away
    if they are not the highest epoch.

    :param run: The AzureML run from where the files should be downloaded.
    :param checkpoint_suffix: The suffix for all files that should be returned.
    :param output_folder: The folder to download the checkpoints to.
    :return: The path to the downloaded file that has the highest epoch. Returns None if no suitable checkpoint
        was found, or if the epoch information could not be extracted.
    """
    files_from_run = download_files_by_suffix(
        run=run,
        suffix=str(checkpoint_suffix),
        output_folder=output_folder,
        validate_checksum=True)
    return find_checkpoint_with_highest_epoch(files_from_run, delete_files=True)


def find_recovery_checkpoint_on_disk_or_cloud(path: Path) -> Optional[Path]:
    """
    Looks at all the checkpoint files and returns the path to the one that should be used for recovery.
    If no checkpoint files are found on disk, the function attempts to download from the current AzureML
    run.

    :param path: The folder to start searching in.
    :return: None if there is no suitable recovery checkpoints, or else a full path to the checkpoint file.
    """
    recovery_checkpoint = find_local_recovery_checkpoint(path)
    if recovery_checkpoint is None and is_running_in_azure_ml():
        logging.info(
            "No recovery checkpoints available in the checkpoint folder. Trying to find checkpoints in AzureML.")
        # Download checkpoints from AzureML, then try to find recovery checkpoints among those.
        # Downloads should go to a temporary folder because downloading the files to the checkpoint
        # folder might cause artifact conflicts later.
        temp_folder = download_checkpoints_from_run(run=RUN_CONTEXT)
        recovery_checkpoint = find_recovery_checkpoint_in_downloaded_files(temp_folder)
    return recovery_checkpoint


def _load_epoch_from_checkpoint(path: Optional[Path]) -> int:
    """
    Loads the epoch number from the given checkpoint file.

    :param path: Path to checkpoint file
    """
    if path is None:
        raise ValueError("Checkpoint path is None")
    checkpoint = torch.load(str(path), map_location=torch.device("cpu"))
    return checkpoint[CHECKPOINT_EPOCH_KEY]


def find_checkpoint_with_highest_epoch(files: Iterable[Path], delete_files: bool = False) -> Optional[Path]:
    """Loads the epoch numbers from the given checkpoint files, and returns the one with the
    highest epoch number. If no files can be loaded, or the list is empty, None is returned.
    If the `delete_files` flag is set to True, all files apart from the one with the highest epoch are deleted.

    :param files: A list of checkpoint files.
    :param delete_files: If True, all files apart from the one with the highest epoch are deleted. If False,
        no files are deleted.
    :return: The checkpoint file with the highest epoch number, or None if no such file was found."""

    def update_file_with_highest_epoch(
        file: Path,
        highest_epoch: Optional[int],
        file_with_highest_epoch: Optional[Path]
    ) -> Tuple[int, Path]:
        """Reads the epoch number from the given `file`, and returns updated information about which file
        has been found to have the highest epoch number.

        :param file: The name of the file to process.
        :param highest_epoch: The highest epoch number encountered so far, or None if no files processed yet.
        :param file_with_highest_epoch: The file that contained the highest epoch number so far, or None if no
            files processed yet.
        :return: A tuple of updated (`highest_epoch`, `file_with_highest_epoch`) values.
        """
        epoch = _load_epoch_from_checkpoint(file)
        logging.info(f"Checkpoint for epoch {epoch} in {file}")
        if (highest_epoch is None) or (epoch > highest_epoch):
            if file_with_highest_epoch is not None and delete_files:
                logging.debug(f"Deleting checkpoint file {file_with_highest_epoch}")
                file_with_highest_epoch.unlink()
            return epoch, file
        assert file_with_highest_epoch is not None
        return highest_epoch, file_with_highest_epoch

    highest_epoch: Optional[int] = None
    file_with_highest_epoch: Optional[Path] = None
    for file in files:
        if file.is_file():
            try:
                highest_epoch, file_with_highest_epoch = \
                    update_file_with_highest_epoch(file, highest_epoch, file_with_highest_epoch)
            except Exception as ex:
                logging.warning(f"Unable to handle checkpoint file {file}: {ex}")
    return file_with_highest_epoch


def find_local_recovery_checkpoint(path: Path) -> Optional[Path]:
    """Checks for a recovery checkpoint in the local checkpoint folder. This can be either
    an autosave checkpoint or the last checkpoint.

    :param path: The folder to search in.
    :return: The checkpoint file with the highest epoch number, or None if no such file was found.
    """
    candidates = [path / f for f in (*AUTOSAVE_CHECKPOINT_CANDIDATES, LAST_CHECKPOINT_FILE_NAME)]
    return find_checkpoint_with_highest_epoch(candidates)


def find_recovery_checkpoint_in_downloaded_files(path: Path) -> Optional[Path]:
    """
    Finds the checkpoint file in the given path that can be used for re-starting the present job.
    All existing checkpoints are loaded, and the one for the highest epoch is used for recovery.

    :param path: The folder to search in.
    :return: The checkpoint file with the highest epoch number, or None if no such file was found.
    """
    candidates = path.glob(f"**/*{CHECKPOINT_SUFFIX}")
    return find_checkpoint_with_highest_epoch(candidates, delete_files=False)


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


class CheckpointDownloader:
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
    """Wrapper class for parsing checkpoint arguments. A checkpoint can be specified in one of the following ways:
        1. A local checkpoint file path
        2. A remote checkpoint file path
        3. A run ID from which to download the checkpoint file
    """
    AML_RUN_ID_FORMAT = (f"<AzureML_run_id>:<optional/custom/path/to/checkpoints/><filename{CHECKPOINT_SUFFIX}>"
                         f"If no custom path is provided (e.g., <AzureML_run_id>:<filename{CHECKPOINT_SUFFIX}>)"
                         "the checkpoint will be downloaded from the default checkpoint folder "
                         f"(e.g., '{DEFAULT_AML_CHECKPOINT_DIR}') If no filename is provided, "
                         "(e.g., `src_checkpoint=<AzureML_run_id>`) the latest checkpoint "
                         f"({LAST_CHECKPOINT_FILE_NAME}) will be downloaded.")
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
    def download_from_url(url: str, download_folder: Path) -> Path:
        """
        Download a checkpoint from checkpoint_url to the download folder. The file name is determined from
        from the file name in the URL. If that can't be determined, use a random file name.

        :param url: The URL from which to download.
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
            logging.info(f"Downloading from URL {url}")

            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(checkpoint_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
        return checkpoint_path

    def get_or_download_checkpoint(self, download_dir: Path) -> Path:
        """Returns the path to the checkpoint file. If the checkpoint is a URL, it will be downloaded to the checkpoints
        folder. If the checkpoint is an AzureML run ID, it will be downloaded from the run to the checkpoints folder.
        If the checkpoint is a local file, it will be returned as is.

        :param download_dir: The checkpoints folder to which the checkpoint should be downloaded if it is a URL or
            AzureML run ID.
        :raises ValueError: If the checkpoint is not a local file, URL or AzureML run ID.
        :raises FileNotFoundError: If the checkpoint is a URL or AzureML run ID and the download fails.
        :return: The path to the checkpoint file.
        """
        if self.is_local_file:
            checkpoint_path = Path(self.checkpoint)
        elif self.is_url:
            download_folder = download_dir / MODEL_WEIGHTS_DIR_NAME
            download_folder.mkdir(exist_ok=True, parents=True)
            checkpoint_path = self.download_from_url(url=self.checkpoint, download_folder=download_folder)
        elif self.is_aml_run_id:
            downloader = CheckpointDownloader(run_id=self.checkpoint, download_dir=download_dir)
            downloader.download_checkpoint_if_necessary()
            checkpoint_path = downloader.local_checkpoint_path
        else:
            raise ValueError("Unable to determine how to get the checkpoint path.")

        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Could not find the file at {checkpoint_path}")
        return checkpoint_path
