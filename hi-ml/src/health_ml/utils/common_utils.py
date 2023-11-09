#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from contextlib import contextmanager
from datetime import datetime
from enum import Enum, unique
from pathlib import Path
from typing import Any, Generator, List, Optional

import torch
from torch.nn import Module
import pandas as pd
from health_azure import paths

from health_azure.utils import PathOrString, is_conda_file_with_pip_include, find_file_in_parent_folders

MAX_PATH_LENGTH = 260

# convert string to None if an empty string or whitespace is provided


def empty_string_to_none(x: Optional[str]) -> Optional[str]:
    return None if (x is None or len(x.strip()) == 0) else x


def string_to_path(x: Optional[str]) -> Optional[Path]:
    return None if (x is None or len(x.strip()) == 0) else Path(x)


# file and directory names
CHECKPOINT_SUFFIX = ".ckpt"
AUTOSAVE_CHECKPOINT_FILE_NAME = "autosave"
AUTOSAVE_CHECKPOINT_CANDIDATES = [
    AUTOSAVE_CHECKPOINT_FILE_NAME + CHECKPOINT_SUFFIX,
    AUTOSAVE_CHECKPOINT_FILE_NAME + "-v1" + CHECKPOINT_SUFFIX,
]
CHECKPOINT_FOLDER = "checkpoints"
DEFAULT_AML_UPLOAD_DIR = "outputs"
DEFAULT_LOGS_DIR_NAME = "logs"
EXPERIMENT_SUMMARY_FILE = "experiment_summary.txt"
DEFAULT_AML_CHECKPOINT_DIR = f"{DEFAULT_AML_UPLOAD_DIR}/{CHECKPOINT_FOLDER}"

# run recovery
RUN_RECOVERY_ID_KEY = "run_recovery_id"
RUN_RECOVERY_FROM_ID_KEY_NAME = "recovered_from"

# other
EFFECTIVE_RANDOM_SEED_KEY_NAME = "effective_random_seed"


@unique
class ModelExecutionMode(Enum):
    """
    Model execution mode
    """

    TRAIN = "Train"
    TEST = "Test"
    VAL = "Val"


def is_windows() -> bool:
    """
    Returns True if the host operating system is Windows.
    """
    return os.name == "nt"


def is_linux() -> bool:
    """
    Returns True if the host operating system is a flavour of Linux.
    """
    return os.name == "posix"


def check_properties_are_not_none(obj: Any, ignore: Optional[List[str]] = None) -> None:
    """
    Checks to make sure the provided object has no properties that have a None value assigned.
    """
    if ignore is not None:
        none_props = [k for k, v in vars(obj).items() if v is None and k not in ignore]
        if len(none_props) > 0:
            raise ValueError("Properties had None value: {}".format(none_props))


@contextmanager
def change_working_directory(path_or_str: PathOrString) -> Generator:
    """
    Context manager for changing the current working directory to the value provided. Outside the context
    manager, the original working directory will be restored.

    :param path_or_str: The new directory to change to
    :yield: a _GeneratorContextManager object (this object itself is of no use, rather we are interested in
        the side effect of the working directory temporarily changing
    """
    new_path = Path(path_or_str).expanduser()
    if not new_path.is_dir():
        raise FileNotFoundError(f"Path {new_path} is not a directory")
    old_path = Path.cwd()
    os.chdir(new_path)
    yield
    os.chdir(str(old_path))


def _create_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Create Torch generator and sets seed with value if provided, or else with a random seed.

    :param seed: Optional seed to set the Generator object with
    :return: Torch Generator object
    """
    generator = torch.Generator()
    if seed is None:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator.manual_seed(seed)
    return generator


def choose_conda_env_file(env_file: Optional[Path] = None) -> Path:
    """
    Chooses the Conda environment file that should be used when submitting the present run to AzureML. If a Conda
    file is given explicitly on the commandline, return that. Otherwise, look in the current folder and its parents for
    a file called `environment.yml`.

    :param env_file: The Conda environment file that was specified on the commandline when starting the run.
    :return: The Conda environment files to use.
    :raises FileNotFoundError: If the specified Conda file does not exist, or none could be found at all.
    """
    if env_file is not None:
        if env_file.is_file():
            return env_file
        raise FileNotFoundError(f"The Conda file specified on the commandline could not be found: {env_file}")
    # When running from the Git repo, then stop search for environment file at repository root. Otherwise,
    # search from current folder all the way up
    stop_at = [paths.git_repo_root_folder()] if paths.is_himl_used_from_git_repo() else []
    current_folder = Path.cwd()
    env_file = find_file_in_parent_folders(
        paths.ENVIRONMENT_YAML_FILE_NAME, start_at_path=current_folder, stop_at_path=stop_at
    )
    if env_file is None:
        raise FileNotFoundError(
            f"No Conda environment file '{paths.ENVIRONMENT_YAML_FILE_NAME}' was found in the "
            f"current folder {current_folder} or its parent folders"
        )
    return env_file


def check_conda_environment(env_file: Path) -> None:
    """Tests if the given conda environment files is valid. In particular, it must not contain "include" statements
    in the pip section.

    :param env_file: The Conda environment YAML file to check.
    """
    has_pip_include, _ = is_conda_file_with_pip_include(env_file)
    if has_pip_include:
        raise ValueError(
            f"The Conda environment definition in {env_file} uses '-r' to reference pip requirements "
            "files. This does not work in AzureML. Please add the pip dependencies directly."
        )


def create_unique_timestamp_id() -> str:
    """
    Creates a unique string using the current time in UTC, up to seconds precision, with characters that
    are suitable for use in filenames. For example, on 31 Dec 2019 at 11:59:59pm UTC, the result would be
    2019-12-31T235959Z.
    """
    unique_id = datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    return unique_id


def is_gpu_available() -> bool:
    """

    :return: True if a GPU with at least 1 device is available.
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 0  # type: ignore


@contextmanager
def set_model_to_eval_mode(model: Module) -> Generator:
    """
    Puts the given torch model into eval mode. At the end of the context, resets the state of the training flag to
    what is was before the call.

    :param model: The model to modify.
    """
    old_mode = model.training
    model.eval()
    yield
    model.train(old_mode)


def is_long_path(path: PathOrString) -> bool:
    """
    A long path is a path that has more than MAX_PATH_LENGTH characters

    :param path: The path to check the length of
    :return: True if the length of the path is greater than MAX_PATH_LENGTH, else False
    """
    return len(str(path)) > MAX_PATH_LENGTH


def df_to_json(df: pd.DataFrame, json_path: Path, add_newline: bool = True) -> None:
    """Save a data frame to a JSON file.

    :param df: Input data frame.
    :param json_path: Path to output JSON file.
    :param add_newline: If ``True``, add newline at the end of the JSON file for POSIX compliance.
    """
    text = df.to_json()
    if add_newline:
        text += '\n'
    json_path.write_text(text)


def seed_monai_if_available(seed: int) -> None:
    """If the MONAI package is available, set its shared seed to make all MONAI operations deterministic.
    If MONAI is not available, nothing will happen.

    :param seed: The random seed to use for MONAI."""
    try:
        # MONAI is not part of the core hi-ml requirements, this import can fail.
        from monai.utils import set_determinism  # type: ignore

        set_determinism(seed=seed)
    except ImportError:
        pass
