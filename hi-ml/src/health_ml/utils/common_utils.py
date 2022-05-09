#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from enum import Enum, unique
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional, Union

import torch
from torch.nn import Module
from health_azure import paths

from health_azure.utils import PathOrString, is_conda_file_with_pip_include

MAX_PATH_LENGTH = 260

# convert string to None if an empty string or whitespace is provided
empty_string_to_none = lambda x: None if (x is None or len(x.strip()) == 0) else x
string_to_path = lambda x: None if (x is None or len(x.strip()) == 0) else Path(x)

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


def check_is_any_of(message: str, actual: Optional[str], valid: Iterable[Optional[str]]) -> None:
    """
    Raises an exception if 'actual' is not any of the given valid values.
    :param message: The prefix for the error message.
    :param actual: The actual value.
    :param valid: The set of valid strings that 'actual' is allowed to take on.
    :return:
    """
    if actual not in valid:
        all_valid = ", ".join(["<None>" if v is None else v for v in valid])
        raise ValueError("{} must be one of [{}], but got: {}".format(message, all_valid, actual))


logging_stdout_handler: Optional[logging.StreamHandler] = None
logging_to_file_handler: Optional[logging.StreamHandler] = None


def logging_to_stdout(log_level: Union[int, str] = logging.INFO) -> None:
    """
    Instructs the Python logging libraries to start writing logs to stdout up to the given logging level.
    Logging will use a timestamp as the prefix, using UTC.

    :param log_level: The logging level. All logging message with a level at or above this level will be written to
    stdout. log_level can be numeric, or one of the pre-defined logging strings (INFO, DEBUG, ...).
    """
    log_level = standardize_log_level(log_level)
    logger = logging.getLogger()
    # This function can be called multiple times, in particular in AzureML when we first run a training job and
    # then a couple of tests, which also often enable logging. This would then add multiple handlers, and repeated
    # logging lines.
    global logging_stdout_handler
    if not logging_stdout_handler:
        print("Setting up logging to stdout.")
        # At startup, logging has one handler set, that writes to stderr, with a log level of 0 (logging.NOTSET)
        if len(logger.handlers) == 1:
            logger.removeHandler(logger.handlers[0])
        logging_stdout_handler = logging.StreamHandler(stream=sys.stdout)
        _add_formatter(logging_stdout_handler)
        logger.addHandler(logging_stdout_handler)
    print(f"Setting logging level to {log_level}")
    logging_stdout_handler.setLevel(log_level)
    logger.setLevel(log_level)


def standardize_log_level(log_level: Union[int, str]) -> int:
    """

    :param log_level: integer or string (any casing) version of a log level, e.g. 20 or "INFO".
    :return: integer version of the level; throws if the string does not name a level.
    """
    if isinstance(log_level, str):
        log_level = log_level.upper()
        check_is_any_of("log_level", log_level, logging._nameToLevel.keys())
        return logging._nameToLevel[log_level]
    return log_level


def _add_formatter(handler: logging.StreamHandler) -> None:
    """
    Adds a logging formatter that includes the timestamp and the logging level.
    """
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    # noinspection PyTypeHints
    formatter.converter = time.gmtime  # type: ignore
    handler.setFormatter(formatter)


@contextmanager
def logging_section(gerund: str) -> Generator:
    """
    Context manager to print "**** STARTING: ..." and "**** FINISHED: ..." lines around sections of the log,
    to help people locate particular sections. Usage:
    with logging_section("doing this and that"):
       do_this_and_that()

    :param gerund: string expressing what happens in this section of the log.
    """
    from time import time

    logging.info("")
    msg = f"**** STARTING: {gerund} "
    logging.info(msg + (100 - len(msg)) * "*")
    logging.info("")
    start_time = time()
    yield
    elapsed = time() - start_time
    logging.info("")
    if elapsed >= 3600:
        time_expr = f"{elapsed / 3600:0.2f} hours"
    elif elapsed >= 60:
        time_expr = f"{elapsed / 60:0.2f} minutes"
    else:
        time_expr = f"{elapsed:0.2f} seconds"
    msg = f"**** FINISHED: {gerund} after {time_expr} "
    logging.info(msg + (100 - len(msg)) * "*")
    logging.info("")


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


def get_all_environment_files(project_root: Path, additional_files: Optional[List[Path]] = None) -> List[Path]:
    """
    Returns a list of all Conda environment files that should be used, comprised of the default conda environment
    definition file, plus any additional files specified in the input. If hi-ml has been downloaded from the
    git repo or installed as a git submodule, the default environment definition file exists in hi-ml/hi-ml.
    Otherwise, looks for a default environment file in project root folder.

    :param project_root: The root folder of the code that starts the present training run.
    :param additional_files: Optional list of additional environment files to merge
    :return: A list of Conda environment files to use.
    """
    env_files = []
    if paths.is_himl_used_from_git_repo():
        env_file = paths.shared_himl_conda_env_file()
        if env_file is None or not env_file.exists():
            # Searching for Conda file starts at current working directory, meaning it might not find
            # the file if cwd is outside the git repo
            env_file = project_root / paths.ENVIRONMENT_YAML_FILE_NAME
            assert env_file.is_file(), f"Didn't find an environment file at {env_file}"

        logging.info(f"Using Conda environment in {env_file}")
        env_files.append(env_file)

    else:
        project_yaml = project_root / paths.ENVIRONMENT_YAML_FILE_NAME
        print(f"Looking for project yaml: {project_yaml}")
        if project_yaml.exists():
            logging.info(f"Using Conda environment in current folder: {project_yaml}")
            env_files.append(project_yaml)

    if not env_files and not additional_files:
        raise ValueError(
            "No Conda environment files were found in the repository, and none were specified in the " "model itself."
        )
    if additional_files:
        for additional_file in additional_files:
            if additional_file.exists():
                env_files.append(additional_file)
    return env_files


def check_conda_environments(env_files: List[Path]) -> None:
    """Tests if all conda environment files are valid. In particular, they must not contain "include" statements
    in the pip section.

    :param env_files: The list of Conda environment YAML files to check.
    """
    if paths.is_himl_used_from_git_repo():
        repo_root_yaml: Path = paths.shared_himl_conda_env_file()
    else:
        repo_root_yaml = None
    for file in env_files:
        has_pip_include, _ = is_conda_file_with_pip_include(file)
        # PIP include statements are only valid when reading from the repository root YAML file, because we
        # are manually adding the included files in get_all_pip_requirements_files
        if has_pip_include and file != repo_root_yaml:
            raise ValueError(
                f"The Conda environment definition in {file} uses '-r' to reference pip requirements "
                "files. This does not work in AzureML. Please add the pip dependencies directly."
            )


def get_all_pip_requirements_files() -> List[Path]:
    """
    If the root level hi-ml directory is available (e.g. it has been installed as a submodule or
    downloaded directly into a parent repo) then we must add it's pip requirements to any environment
    definition. This function returns a list of the necessary pip requirements files. If the hi-ml
    root directory does not exist (e.g. hi-ml has been installed as a pip package, this is not necessary
    and so this function returns an empty list.)

    :return: An list list of pip requirements files in the hi-ml and hi-ml-azure packages if relevant,
        or else an empty list
    """
    files = []
    if paths.is_himl_used_from_git_repo():
        git_root = paths.git_repo_root_folder()
        for folder in [Path("hi-ml") / "run_requirements.txt", Path("hi-ml-azure") / "run_requirements.txt"]:
            files.append(git_root / folder)
    return files


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
