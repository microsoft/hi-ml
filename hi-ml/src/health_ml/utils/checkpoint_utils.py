#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import tempfile
from pathlib import Path
from typing import Optional

import torch
from azureml.core import Run, Workspace

from health_azure.utils import (RUN_CONTEXT, download_files_from_run_id, get_run_file_names, is_running_in_azure_ml)
from health_ml.utils.common_utils import (AUTOSAVE_CHECKPOINT_CANDIDATES, CHECKPOINT_FOLDER, DEFAULT_AML_UPLOAD_DIR)

CHECKPOINT_SUFFIX = ".ckpt"
# This is a constant that must match a filename defined in pytorch_lightning.ModelCheckpoint, but we don't want
# to import that here.
LAST_CHECKPOINT_FILE_NAME = "last"
LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX = LAST_CHECKPOINT_FILE_NAME + CHECKPOINT_SUFFIX
LEGACY_RECOVERY_CHECKPOINT_FILE_NAME = "recovery"
MODEL_INFERENCE_JSON_FILE_NAME = "model_inference_config.json"
MODEL_WEIGHTS_DIR_NAME = "trained_models"


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


def cleanup_checkpoints(ckpt_folder: Path) -> None:
    """
    Remove autosave checkpoints from the given checkpoint folder, and check if a "last.ckpt" checkpoint is present.
    :param ckpt_folder: The folder that contains all checkpoint files.
    """
    files_in_checkpoint_folder = [p.name for p in ckpt_folder.glob('*')]
    if len(files_in_checkpoint_folder) == 0:
        return
    logging.info(f"Files in checkpoint folder: {' '.join(files_in_checkpoint_folder)}")
    last_ckpt = ckpt_folder / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    all_files = f"Existing files: {' '.join(p.name for p in ckpt_folder.glob('*'))}"
    if not last_ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint file {LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX} not found. {all_files}")
    # Training is finished now. To save storage, remove the autosave checkpoint which is now obsolete.
    # Lightning does not overwrite checkpoints in-place. Rather, it writes "autosave.ckpt",
    # then "autosave-1.ckpt" and deletes "autosave.ckpt", then "autosave.ckpt" and deletes "autosave-v1.ckpt"
    for candidate in AUTOSAVE_CHECKPOINT_CANDIDATES:
        autosave = ckpt_folder / candidate
        if autosave.is_file():
            autosave.unlink()
