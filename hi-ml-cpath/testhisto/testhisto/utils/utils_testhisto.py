#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence
import numpy as np
from pytest import MarkDecorator
import pytest
import torch
import torch.distributed
from torch import cuda
from PIL import Image


def assert_dicts_equal(d1: Mapping, d2: Mapping, exclude_keys: Collection[Any] = (),
                       rtol: float = 1e-5, atol: float = 1e-8) -> None:
    assert isinstance(d1, Mapping)
    assert isinstance(d2, Mapping)
    keys1 = [key for key in d1 if key not in exclude_keys]
    keys2 = [key for key in d2 if key not in exclude_keys]
    assert keys1 == keys2
    for key in keys1:
        msg = f"Dictionaries differ for key '{key}': {d1[key]} vs {d2[key]}"
        if isinstance(d1[key], torch.Tensor):
            assert torch.allclose(d1[key], d2[key], rtol=rtol, atol=atol, equal_nan=True), msg
        elif isinstance(d1[key], np.ndarray):
            assert np.allclose(d1[key], d2[key], rtol=rtol, atol=atol, equal_nan=True), msg
        else:
            assert d1[key] == d2[key], msg


def assert_file_exists(file_path: Path) -> None:
    """
    Checks if the given file exists.
    """
    assert file_path.exists(), f"File does not exist: {file_path}"


def assert_binary_files_match(actual_file: Path, expected_file: Path) -> None:
    """
    Checks if two files contain exactly the same bytes. If PNG files mismatch, additional diagnostics is printed.
    """
    # Uncomment this line to batch-update all result files that use this assert function
    # expected_file.write_bytes(actual_file.read_bytes())
    assert_file_exists(actual_file)
    assert_file_exists(expected_file)
    actual = actual_file.read_bytes()
    expected = expected_file.read_bytes()
    if actual == expected:
        return
    if actual_file.suffix == ".png" and expected_file.suffix == ".png":
        actual_image = Image.open(actual_file)
        expected_image = Image.open(expected_file)
        actual_size = actual_image.size
        expected_size = expected_image.size
        assert actual_size == expected_size, f"Image sizes don't match: actual {actual_size}, expected {expected_size}"
        assert np.allclose(np.array(actual_image), np.array(expected_image)), "Image pixel data does not match."
        return
    assert False, f"File contents does not match: len(actual)={len(actual)}, len(expected)={len(expected)}"


def full_ml_test_data_path(suffix: str = "") -> Path:
    """
    Returns the path to a folder named "test_data" / <suffix>  in testhisto

    :param suffix: The name of the folder to create in "test_data". If not provided,
    the path to test_data will be returned
    :return: The full absolute path of the directory
    """
    root = Path(os.path.realpath(__file__)).parent.parent.parent
    test_data_dir = root / "test_data"
    return test_data_dir / suffix


def _run_distributed_process(rank: int, world_size: int, fn: Callable[..., None], args: Sequence[Any] = (),
                             backend: str = 'nccl') -> None:
    """Run a function in the current subprocess within a PyTorch Distributed context.

    This function should be called with :py:func:`torch.multiprocessing.spawn()`.

    Reference: https://pytorch.org/tutorials/intermediate/dist_tuto.html

    :param rank: Rank of the current process.
    :param world_size: Total number of distributed subprocesses.
    :param fn: Function to execute in each subprocess, accepting least the following keyword arguments:

        * `rank` (`int`): The (global) rank assigned to the subprocess executing the function.
        * `world_size` (`int`): Total number of distributed subprocesses.
        * `device` (`str`): CUDA device allocated to this process (e.g. `'cuda:1'`).

    :param args: Any positional arguments to be passed to `fn`, which will be called as
        ``fn(*args, rank=..., world_size=..., device=...)``.
    :param backend: Distributed communication backend (default: `'nccl'`).
    """
    # TODO: Add timeouts and proper handling of CUDA/NCCL errors
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    fn(*args, rank=rank, world_size=world_size, device=device)
    torch.distributed.destroy_process_group()


def run_distributed(fn: Callable[..., None], args: Sequence[Any] = (), world_size: int = 1) -> None:
    """Run a function in multiple subprocesses using PyTorch Distributed.

    Reference: https://pytorch.org/tutorials/intermediate/dist_tuto.html

    :param fn: Function to execute in each subprocess, accepting least the following keyword arguments:

        * `rank` (`int`): The (global) rank assigned to the subprocess executing the function.
        * `world_size` (`int`): Total number of distributed subprocesses.
        * `device` (`str`): CUDA device allocated to this process (e.g. `'cuda:1'`).

    :param args: Any positional arguments to be passed to `fn`, which will be called as
        ``fn(*args, rank=..., world_size=..., device=...)``.
    :param world_size: Total number of distributed subprocesses to spawn.
    """
    torch.multiprocessing.spawn(_run_distributed_process, args=(world_size, fn, args), nprocs=world_size)


def wait_until_file_exists(filename: Path, timeout_sec: float = 10.0, sleep_sec: float = 0.1) -> None:
    """Wait until the given file exists. If the file does not exist after the given timeout, an exception is raised.

    :param filename: The file to wait for.
    :param timeout_sec: The maximum time to wait until the file exists.
    :param sleep_sec: The time to sleep between repeated checks if the file exists already.
    :raises TimeoutError: If the file does not exist after the given timeout."""
    current_time = time.time()
    while not filename.exists():
        logging.info(f"Waiting for file {filename}. Total wait time so far: {time.time() - current_time} seconds.")
        time.sleep(sleep_sec)
        if time.time() - current_time > timeout_sec:
            raise TimeoutError(f"File {filename} still does not exist after waiting for {timeout_sec} seconds")


def skipif_no_gpu() -> MarkDecorator:
    """Convenience for pytest.mark.skipif() in case no GPU is available.

    :return: A Pytest skipif mark decorator.
    """
    has_gpu = cuda.is_available() and cuda.device_count() > 0
    return pytest.mark.skipif(not has_gpu, reason="No GPU available")
