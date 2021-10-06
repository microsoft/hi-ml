#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Testing run_upload_file.
"""

from pathlib import Path
from typing import Set

from azureml.core.run import Run

import health.azure.azure_util as util

try:
    import upload_util
except Exception:
    import testazure.test_data.simple.upload_util as upload_util  # type: ignore


def init_test(tmp_path: Path) -> None:
    """
    Create test files.

    :param tmp_path: Folder to create test files in.
    """
    upload_util.create_test_files(tmp_path, None, range(0, 6))


def run_test(run: Run) -> None:
    """
    Run a set of tests against run.upload_file and run_upload_file.
    """
    # Extract the list of test file names
    filenames = upload_util.get_test_file_names()

    def amlupload_file(run: Run,
                       name: str,
                       path_or_stream: str) -> None:
        """
        Upload a folder using AzureML directly.
        """
        run.upload_file(name, path_or_stream)

    def himlupload_file(run: Run,
                        name: str,
                        path_or_stream: str) -> None:
        """
        Upload a folder using the HI-ML wrapper function.
        """
        util.run_upload_file(run, name, path_or_stream)

    # Test against two different methods. AzureML directly and using the HI-ML wrapper
    upload_datas = [
        # Test against AzureML. This takes a long time because of two minute timeouts trying to download
        # corrupted files.
        # ("upload_file_aml", amlupload_file, True),
        # Test against HI-ML wrapper function.
        ("upload_file_himl", himlupload_file, False)
    ]

    alias = "test_file0_txt.txt"

    # List of pairs names and files
    test_file_name_sets = [
        (f"{alias}", filenames[0]),
        (f"sub1/{alias}", filenames[1]),
        (f"sub1/sub2/sub3/{alias}", filenames[2]),
        (f"{alias}", filenames[3]),
        (f"sub1/{alias}", filenames[4]),
        (f"sub1/sub2/sub3/{alias}", filenames[5]),
    ]

    test_upload_folder = Path(upload_util.test_upload_folder_name)

    # Step 1, upload the first file
    upload_file_aliases: Set[str] = set()
    upload_util.copy_test_file_name_set(test_upload_folder, set(filenames))

    for upload_folder_name, upload_fn, errors in upload_datas:
        for suffix, filename in test_file_name_sets[0:3]:
            name = f"{upload_folder_name}/{suffix}"
            print(f"Upload the first file: {upload_folder_name}, {filename}")
            upload_fn(run=run,
                      name=name,
                      path_or_stream=str(test_upload_folder / filename))

            upload_file_aliases = upload_file_aliases.union(suffix)
            upload_util.check_files(run, upload_file_aliases, set(), 1, upload_folder_name)

    # Step 2, upload the first file again
    for upload_folder_name, upload_fn, errors in upload_datas:
        for suffix, filename in test_file_name_sets[0:3]:
            name = f"{upload_folder_name}/{suffix}"
            print(f"Upload the first file again: {upload_folder_name}, {filename}, \
                this should fail since first file already there")
            try:
                upload_fn(run=run,
                          name=name,
                          path_or_stream=str(test_upload_folder / filename))
            except Exception as ex:
                print(f"Expected error in run.upload_file: {str(ex)}")
                if errors:
                    assert "UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.test_script_" in str(ex)
                    for f in upload_file_aliases:
                        assert f"{f} already exists" in str(ex)
                else:
                    # File is the same, so nothing should have happened
                    raise ex

            upload_util.check_files(run, upload_file_aliases, set(), 2, upload_folder_name)

    # Step 3, upload a second file with the same alias as the first
    for upload_folder_name, upload_fn, errors in upload_datas:
        for suffix, filename in test_file_name_sets[3:6]:
            name = f"{upload_folder_name}/{suffix}"
            print(f"Upload a second file with same name as the first: {upload_folder_name}, {filename}, \
                this should fail since first file already there")
            try:
                upload_fn(run=run,
                          name=name,
                          path_or_stream=str(test_upload_folder / filename))
            except Exception as ex:
                print(f"Expected error in run.upload_file: {str(ex)}")
                if errors:
                    assert "UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.test_script_" in str(ex)
                    for f in upload_file_aliases:
                        assert f"{f} already exists" in str(ex)
                else:
                    assert f"Trying to upload file {name} but that file already exists in the run." \
                            "in str(ex)"

            upload_util.check_files(run, upload_file_aliases, set(), 3, upload_folder_name)
