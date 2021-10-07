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

    # List of pairs of name prefixes and files
    test_file_name_names = [
        ("", filenames[0]),
        ("sub1/", filenames[1]),
        ("sub1/sub2/sub3/", filenames[2]),
        ("", filenames[3]),
        ("sub1/", filenames[4]),
        ("sub1/sub2/sub3/", filenames[5]),
    ]

    test_upload_folder = Path(upload_util.test_upload_folder_name)

    # Step 1, upload the first set of three files
    upload_util.copy_test_file_name_set(test_upload_folder, set(filenames))

    step = 0
    for upload_folder_name, upload_fn, errors in upload_datas:
        for i in range(0, 3):
            prefix, filename = test_file_name_names[i]
            name = f"{prefix}{upload_folder_name}_{alias}"
            print(f"Upload the first files: {upload_folder_name}, {name}={filename}")
            upload_fn(run=run,
                      name=name,
                      path_or_stream=str(test_upload_folder / filename))

            step = step + 1
            upload_util.check_file(run=run,
                                   name=name,
                                   good_filename=filename,
                                   bad_filename="",
                                   step=step,
                                   upload_folder_name=upload_folder_name)

    # Step 2, upload the first set of three files again
    for upload_folder_name, upload_fn, errors in upload_datas:
        for i in range(0, 3):
            prefix, filename = test_file_name_names[i]
            name = f"{prefix}{upload_folder_name}_{alias}"
            print(f"Upload the first files again: {upload_folder_name}, {name}={filename}, \
                  this should fail since first file already there")
            try:
                upload_fn(run=run,
                          name=name,
                          path_or_stream=str(test_upload_folder / filename))
            except Exception as ex:
                print(f"Expected error in run.upload_file: {str(ex)}")
                if errors:
                    assert "UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.test_script_" in str(ex)
                    assert f"{name} already exists" in str(ex)
                else:
                    # File is the same, so nothing should have happened
                    raise ex

            step = step + 1
            upload_util.check_file(run=run,
                                   name=name,
                                   good_filename=(filename if not errors else ""),
                                   bad_filename=("" if not errors else filename),
                                   step=step,
                                   upload_folder_name=upload_folder_name)

    # Step 3, upload a second set of three files with the same alias as the first
    for upload_folder_name, upload_fn, errors in upload_datas:
        for i in range(3, 6):
            prefix, filename = test_file_name_names[i]
            name = f"{prefix}{upload_folder_name}_{alias}"
            print(f"Upload a second files with same name as the first: {upload_folder_name}, {name}={filename}, \
                  this should fail since first file already there")
            try:
                upload_fn(run=run,
                          name=name,
                          path_or_stream=str(test_upload_folder / filename))
            except Exception as ex:
                print(f"Expected error in run.upload_file: {str(ex)}")
                if errors:
                    assert "UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.test_script_" in str(ex)
                    assert f"{name} already exists" in str(ex)
                else:
                    assert f"Trying to upload file {name} but that file already exists in the run." \
                            "in str(ex)"

            _, existing_filename = test_file_name_names[i - 3]

            step = step + 1
            upload_util.check_file(run=run,
                                   name=name,
                                   good_filename=(existing_filename if not errors else ""),
                                   bad_filename=("" if not errors else filename),
                                   step=step,
                                   upload_folder_name=upload_folder_name)

