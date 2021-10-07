#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Utilities for testing run_upload_file, run_upload_files, and run_upload_folder.
"""
import shutil
from pathlib import Path
from typing import List, Optional, Set
from uuid import uuid4

from azureml.core.run import Run
from azureml.exceptions import AzureMLException

from health.azure.azure_util import hash_file, run_download_file_name


# The folder where the test files have been created. Since they are in a subfolder of the script, they should
# have been uploaded.
_base_data_folder_name = "base_data"
_base_data_folder = Path("outputs") / _base_data_folder_name

# A file to store the ordered list of test files.
_filenames_file_name = "filenames.txt"
_filenames_file = _base_data_folder / _filenames_file_name

# Create a new folder to upload files from
test_upload_folder_name = "test_data"


def create_test_files(subfolder: Optional[Path], range: range) -> None:
    """
    Create a set of test files in a target folder.

    If subfolder is not empty then the target folder will be "outputs" / "base_data" / subfolder, otherwise it will
    be "outputs" / "base_data". This folder will be created if it does not exist.

    :param path: Base folder, intended to be a temporary folder.
    :param subfolder: Optional subfolder of "base_data", where to create the test files.
    :param range: Range of suffixes to apply to the filenames.
    :return: None.
    """
    folder = _base_data_folder / subfolder if subfolder is not None else _base_data_folder
    if not folder.exists():
        folder.mkdir(parents=True)
    filenames = [folder / f"test_file{i}.txt" for i in range]
    # Populate the dummy text files with some unique text
    for filename in filenames:
        filename.write_text(f"some test data: {uuid4().hex}")
    relative_filenames = [str(f.relative_to(_base_data_folder)) for f in filenames]
    existing_filenames = _filenames_file.read_text() + "\n" if _filenames_file.exists() else ""
    _filenames_file.write_text(existing_filenames + "\n".join(relative_filenames))


def get_test_file_names() -> List[str]:
    """
    Return a list of filenames in the base_data folder, relative to that folder.

    :return: List of file names stored in the filenames.txt file.
    """
    return _filenames_file.read_text().split("\n")


def copy_test_file_name_set(test_upload_folder: Path, test_file_name_set: Set[str]) -> None:
    """
    Copy a set of test files from the base_data folder to the test_data folder, making sure parent folders exist.

    :param test_upload_folder: Folder containing test files.
    :param test_file_name_set: Set of files to copy.
    :return: None.
    """
    for f in test_file_name_set:
        target_folder = (test_upload_folder / f).parent
        if not target_folder.exists():
            target_folder.mkdir(parents=True)

        shutil.copyfile(_base_data_folder / f, test_upload_folder / f)


def rm_test_file_name_set(test_upload_folder: Path) -> None:
    """
    Completely remove the test_data folder and recreate it.

    :param test_upload_folder: Folder containing test files.
    :return: None.
    """
    if test_upload_folder.exists():
        shutil.rmtree(test_upload_folder)
    test_upload_folder.mkdir()


def assert_folder_contents(filenames: Set[str], download_folder: Path) -> None:
    """
    Compare hashes of two sets of files.

    :param good_filenames: Set of filenames that should have been uploaded without an error.
    :param download_folder: Folder with downloaded files.
    :return: None.
    """
    for filename in filenames:
        uploaded_file = _base_data_folder / filename
        uploaded_file_hash = hash_file(uploaded_file)
        downloaded_file = download_folder / filename
        downloaded_file_hash = hash_file(downloaded_file)
        assert uploaded_file_hash == downloaded_file_hash


def check_file(run: Run,
               name: str,
               good_filename: str,
               bad_filename: str,
               step: int,
               upload_folder_name: str) -> None:
    """
    Check that a file that has been uploaded to the run is as expected.

    :param run: AzureML run.
    :param name: Name of file on AzureML run.
    :param good_filename: Source filename that was expected to have been uploaded without an error.
    :param bad_filename: Source filename that was expected to have been uploaded with an error.
    :param step: Which step of the test, used for logging and creating temporary folders.
    :param upload_folder_name: Upload folder name.
    :return: None.
    """
    print_prefix = f"check_file_{step}_{upload_folder_name}:"
    print(f"{print_prefix} name:{name}")
    print(f"{print_prefix} good_filename:{good_filename}")
    print(f"{print_prefix} bad_filename:{bad_filename}")
    filename = good_filename or bad_filename

    # Download all the file names for this run with the expected name
    run_file_names = {f for f in run.get_file_names() if f == name}
    print(f"{print_prefix} run_file_names:{sorted(run_file_names)}")
    # This should be just the name, since duplicates are not allowed.
    assert run_file_names == {name}

    # Make a folder to download it
    download_folder_all = Path(f"outputs/download_file_{step}_{upload_folder_name}")
    download_folder_all.mkdir()

    if not bad_filename:
        # With no bad filename, it should be possible to just download it.
        run.download_file(name=name,
                          output_file_path=str(download_folder_all))
    else:
        # With bad filenames, run.download_file will raise an exception.
        try:
            run.download_file(name=name,
                              output_file_path=str(download_folder_all))
        except AzureMLException as ex:
            print(f"Expected error in download_file: {str(ex)}")
            assert "Failed to flush task queue within 120 seconds" in str(ex)

    # Glob the list of all the files that have been downloaded relative to the download folder.
    downloaded_all_local_files = {str(f.relative_to(download_folder_all))
                                  for f in download_folder_all.rglob("*") if f.is_file()}
    print(f"{print_prefix} downloaded_all_local_files:{sorted(downloaded_all_local_files)}")
    # This should be the same as the filename.
    assert downloaded_all_local_files == {run_download_file_name(name)}

    uploaded_file = _base_data_folder / filename
    uploaded_file_hash = hash_file(uploaded_file)
    downloaded_file = download_folder_all / run_download_file_name(name)
    downloaded_file_hash = hash_file(downloaded_file)
    assert uploaded_file_hash == downloaded_file_hash


def check_folder(run: Run,
                 good_filenames: Set[str],
                 bad_filenames: Set[str],
                 step: int,
                 upload_folder_name: str) -> None:
    """
    Check that the list of files that have been uploaded to the run for this folder name are as expected.

    :param run: AzureML run.
    :param good_filenames: Set of filenames that should have been uploaded without an error.
    :param bad_filenames: Set of filenames that are expected to have been uploaded with an error.
    :param step: Which step of the test, used for logging and creating temporary folders.
    :param upload_folder_name: Upload folder name.
    :return: None.
    """
    print_prefix = f"check_files_{step}_{upload_folder_name}:"
    print(f"{print_prefix} good_filenames:{sorted(good_filenames)}")
    print(f"{print_prefix} bad_filenames:{sorted(bad_filenames)}")
    all_filenames = good_filenames.union(bad_filenames)

    # Download all the file names for this upload_folder_name, stripping off the leading upload_folder_name and /
    run_file_names = {f[len(upload_folder_name) + 1:]
                      for f in run.get_file_names() if f.startswith(f"{upload_folder_name}/")}
    print(f"{print_prefix} run_file_names:{sorted(run_file_names)}")
    # This should be the same as the list of good and bad filenames combined.
    assert run_file_names == all_filenames

    # Make a folder to download them all at once
    download_folder_all = Path(f"outputs/download_folder_all_{step}_{upload_folder_name}")
    download_folder_all.mkdir()

    if len(bad_filenames) == 0:
        # With no bad filenames, it should be possible to just download them all at once
        # The option 'append_prefix' actually removes the upload_folder_name.
        run.download_files(prefix=upload_folder_name,
                           output_directory=str(download_folder_all),
                           append_prefix=False)
    else:
        # With bad filenames, run.download_files will raise an exception.
        try:
            run.download_files(prefix=upload_folder_name,
                               output_directory=str(download_folder_all),
                               append_prefix=False)
        except AzureMLException as ex:
            print(f"Expected error in download_files: {str(ex)}")
            assert "Failed to flush task queue within 120 seconds" in str(ex)

    # Glob the list of all the files that have been downloaded relative to the download folder.
    downloaded_all_local_files = {str(f.relative_to(download_folder_all))
                                  for f in download_folder_all.rglob("*") if f.is_file()}
    print(f"{print_prefix} downloaded_all_local_files:{sorted(downloaded_all_local_files)}")
    # This should be the same as the list of good and bad filenames.
    assert downloaded_all_local_files == all_filenames
    assert_folder_contents(good_filenames, download_folder_all)

    # Make a folder to download them individually
    download_folder_ind = Path(f"outputs/download_folder_ind_{step}_{upload_folder_name}")
    download_folder_ind.mkdir()

    for f in all_filenames:
        # Each file may be in a sub folder, make sure it exists before trying download.
        target_folder = (download_folder_ind / f).parent
        if not target_folder.exists():
            target_folder.mkdir(parents=True)

        try:
            # Try to download each file
            run.download_file(name=f"{upload_folder_name}/{f}",
                              output_file_path=str(target_folder))
        except AzureMLException as ex:
            if f in bad_filenames:
                # If this file is in the list of bad_filenames this is expected to raise an exception.
                print(f"Expected error in download_file: {f}: {str(ex)}")
                assert "Download of file failed with error: The specified blob does not exist. " \
                       "ErrorCode: BlobNotFound" in str(ex)
            else:
                print(f"Unexpected error in download_file: {f}: {str(ex)}")
                # Otherwise, reraise the exception to terminate the run.
                raise ex

    # Glob the list of all the files that have been downloaded relative to the download folder.
    downloaded_ind_local_files = {str(f.relative_to(download_folder_ind))
                                  for f in download_folder_ind.rglob("*") if f.is_file()}
    print(f"{print_prefix} downloaded_ind_local_files:{sorted(downloaded_ind_local_files)}")
    # This should be the same as the list of good and bad filenames.
    assert downloaded_ind_local_files == all_filenames
    assert_folder_contents(good_filenames, download_folder_ind)
