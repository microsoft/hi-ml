#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------
import datetime
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Generator, Optional, Union

from health_azure.utils import ENV_LOCAL_RANK, check_is_any_of, is_global_rank_zero

logging_stdout_handler: Optional[logging.StreamHandler] = None
logger = logging.getLogger(__name__)


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
        if is_global_rank_zero():
            print("Setting up logging to stdout.")
        # At startup, logging has one handler set, that writes to stderr, with a log level of 0 (logging.NOTSET)
        if len(logger.handlers) == 1:
            logger.removeHandler(logger.handlers[0])
        logging_stdout_handler = logging.StreamHandler(stream=sys.stdout)
        _add_formatter(logging_stdout_handler)
        logger.addHandler(logging_stdout_handler)
    if is_global_rank_zero():
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


def format_time_from_seconds(time_in_seconds: float) -> str:
    """Formats a time in seconds as a string, e.g. 1.5 hours, 2.5 minutes, 3.5 seconds.

    :param time_in_seconds: time in seconds.
    :return: string expressing the time. If the time is more than an hour, it is expressed in hours, to 2 decimal
        places. If the time is more than a minute, it is expressed in minutes, to 2 decimal places. Otherwise, it is
        rounded to 2 decimal places and expressed in seconds.
    """
    if time_in_seconds >= 3600:
        time_expr = f"{time_in_seconds / 3600:0.2f} hours"
    elif time_in_seconds >= 60:
        time_expr = f"{time_in_seconds / 60:0.2f} minutes"
    else:
        time_expr = f"{time_in_seconds:0.2f} seconds"
    return time_expr


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

    logger.info("")
    msg = f"**** STARTING: {gerund} "
    logger.info(msg + (100 - len(msg)) * "*")
    logger.info("")
    start_time = time()
    yield
    elapsed = time() - start_time
    logger.info("")
    time_expr = format_time_from_seconds(elapsed)
    msg = f"**** FINISHED: {gerund} after {time_expr} "
    logger.info(msg + (100 - len(msg)) * "*")
    logger.info("")


def print_message_with_rank_pid(message: str = '', level: str = 'DEBUG') -> None:
    """Prints a message with the rank and PID of the current process.

    :param message: message to print.
    :param level: logging level to use.
    """
    print(f"{datetime.datetime.utcnow()} {level}    Rank {os.getenv(ENV_LOCAL_RANK)} - PID {os.getpid()} - {message}")


@contextmanager
def elapsed_timer(message: str, format_seconds: bool = False) -> Generator:
    """Context manager to print the elapsed time for a block of code in addition to its local rank and PID.
    Usage:
    with elapsed_timer("doing this and that"):
        print("doing this and that")
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    time_expr = format_time_from_seconds(elapsed) if format_seconds else f"{elapsed:0.2f} seconds"
    print_message_with_rank_pid(f"{message} took {time_expr}")
