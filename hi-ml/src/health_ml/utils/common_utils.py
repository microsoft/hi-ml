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

from health_azure.utils import PathOrString

from health_ml.utils import fixed_paths


MAX_PATH_LENGTH = 260

# convert string to None if an empty string or whitespace is provided
empty_string_to_none = lambda x: None if (x is None or len(x.strip()) == 0) else x
string_to_path = lambda x: None if (x is None or len(x.strip()) == 0) else Path(x)

ARGS_TXT = "args.txt"
BEST_EPOCH_FOLDER_NAME = "best_validation_epoch"
CROSSVAL_SPLIT_KEY = "cross_validation_split_index"
DEFAULT_CROSSVAL_SPLIT_INDEX = -1
RUN_RECOVERY_ID_KEY = 'run_recovery_id'
OTHER_RUNS_SUBDIR_NAME = "OTHER_RUNS"
ENSEMBLE_SPLIT_NAME = "ENSEMBLE"
EFFECTIVE_RANDOM_SEED_KEY_NAME = "effective_random_seed"
RUN_RECOVERY_FROM_ID_KEY_NAME = "recovered_from"
DEFAULT_AML_UPLOAD_DIR = "outputs"
DEFAULT_LOGS_DIR_NAME = "logs"


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


def logging_to_file(file_path: Path) -> None:
    """
    Instructs the Python logging libraries to start writing logs to the given file.
    Logging will use a timestamp as the prefix, using UTC. The logging level will be the same as defined for
    logging to stdout.

    :param file_path: The path and name of the file to write to.
    """
    # This function can be called multiple times, and should only add a handler during the first call.
    global logging_to_file_handler
    if not logging_to_file_handler:
        global logging_stdout_handler
        log_level = logging_stdout_handler.level if logging_stdout_handler else logging.INFO
        print(f"Setting up logging with level {log_level} to file {file_path}")
        file_path.parent.mkdir(exist_ok=True, parents=True)
        handler = logging.FileHandler(filename=str(file_path))
        _add_formatter(handler)
        handler.setLevel(log_level)
        logging.getLogger().addHandler(handler)
        logging_to_file_handler = handler


def disable_logging_to_file() -> None:
    """
    If logging to a file has been enabled previously via logging_to_file, this call will remove that logging handler.
    """
    global logging_to_file_handler
    if logging_to_file_handler:
        logging_to_file_handler.close()
        logging.getLogger().removeHandler(logging_to_file_handler)
        logging_to_file_handler = None


def _add_formatter(handler: logging.StreamHandler) -> None:
    """
    Adds a logging formatter that includes the timestamp and the logging level.
    """
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s",
                                  datefmt="%Y-%m-%dT%H:%M:%SZ")
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
    return os.name == 'nt'


def is_linux() -> bool:
    """
    Returns True if the host operating system is a flavour of Linux.
    """
    return os.name == 'posix'


def check_properties_are_not_none(obj: Any, ignore: Optional[List[str]] = None) -> None:
    """
    Checks to make sure the provided object has no properties that have a None value assigned.
    """
    if ignore is not None:
        none_props = [k for k, v in vars(obj).items() if v is None and k not in ignore]
        if len(none_props) > 0:
            raise ValueError("Properties had None value: {}".format(none_props))


def path_to_namespace(path: Path, root: PathOrString = fixed_paths.repository_root_directory()) -> str:
    """
    Given a path (in form R/A/B/C) and an optional root directory R, create a namespace A.B.C.
    If root is provided, then path must be a relative child to it.

    :param path: Path to convert to namespace
    :param root: Path prefix to remove from namespace (default is project root)
    :return:
    """
    return ".".join([Path(x).stem for x in path.relative_to(root).parts])


def remove_file_or_directory(pth: Path) -> None:
    """
    Remove a directory and its contents, or a file.

    :param pth: the Path to the file or directory to be removed
    """
    if pth.is_dir():
        for child in pth.glob('*'):
            if child.is_file():
                child.unlink()
            else:
                remove_file_or_directory(child)
        pth.rmdir()
    elif pth.exists():
        pth.unlink()


@contextmanager
def change_working_directory(path_or_str: PathOrString) -> Generator:
    """
    Context manager for changing the current working directory to the value provided. Outside of the context
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


def get_all_environment_files(project_root: Path) -> List[Path]:
    """
    Returns a list of all Conda environment files that should be used. For now this is just an
    environment.yml file that lives at the project root folder.

    :param project_root: The root folder of the code that starts the present training run.
    :return: A list with 1 entry that is the root level repo's conda environment files.
    """
    project_yaml = project_root / fixed_paths.ENVIRONMENT_YAML_FILE_NAME
    if project_yaml.exists():
        return [project_yaml]
    return []


def get_all_pip_requirements_files() -> List[Path]:
    """
    If the root level hi-ml directory is available (e.g. it has been installed as a submodule or
    downloaded directly into a parent repo) then we must add it's pip requirements to any environment
    definition. This function returns a list of the necessary pip requirements files. If the hi-ml
    root directory does not exist (e.g. hi-ml has been installed as a pip package, this is not necessary
    and so this function returns None)

    :return: An list list of pip requirements files in the hi-ml and hi-ml-azure packages if relevant,
        or else an empty list
    """
    files = []
    himl_root_dir = fixed_paths.himl_root_dir()
    if himl_root_dir is not None:
        himl_yaml = himl_root_dir / "hi-ml" / "run_requirements.txt"
        himl_az_yaml = himl_root_dir / "hi-ml-azure" / "run_requirements.txt"
        files.append(himl_yaml)
        files.append(himl_az_yaml)
        return files
    return []


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
