#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Testing run_upload_file.
"""

from pathlib import Path

from azureml.core.run import Run

import health.azure.azure_util as util

try:
    import upload_util
except Exception:
    import testazure.test_data.simple.upload_util as upload_util


def init_test(tmp_path: Path) -> None:
    """
    Create test files.

    :param tmp_path: Folder to create test files in.
    """
    upload_util.create_test_files(tmp_path, None, range(0, 2))


def run_test(run: Run) -> None:
    """
    Run a set of tests against run.upload_file and run_upload_file.
    """
    # Extract the list of test file names
    filenames = upload_util.get_base_data_filenames()

    test_file_name_sets = [
        {filenames[0]},
        {filenames[1]}
    ]

    test_file_name_alias = "test_file0_txt.txt"

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
        ("upload_file_aml", amlupload_file, True),
        # Test against HI-ML wrapper function.
        ("upload_file_himl", himlupload_file, False)
    ]

    test_upload_folder = Path(upload_util.test_upload_folder_name)

    # Step 1, upload the first file
    upload_files = test_file_name_sets[0]
    upload_file_aliases = {test_file_name_alias}
    upload_util.copy_test_file_name_set(test_file_name_sets[0])

    for upload_folder_name, upload_fn, errors in upload_datas:
        print(f"Upload the first file: {upload_folder_name}, {upload_files}")
        upload_fn(run=run,
                  name=f"{upload_folder_name}/{test_file_name_alias}",
                  path_or_stream=str(test_upload_folder / filenames[0]))

        upload_util.check_files(run, upload_file_aliases, set(), 1, upload_folder_name)

    # Step 2, upload the first file again
    for upload_folder_name, upload_fn, errors in upload_datas:
        print(f"Upload the first file again: {upload_folder_name}, {upload_files}, \
              this should fail since first file already there")
        try:
            upload_fn(run=run,
                      name=f"{upload_folder_name}/{test_file_name_alias}",
                      path_or_stream=str(test_upload_folder / filenames[0]))
        except Exception as ex:
            print(f"Expected error in run.upload_file: {str(ex)}")
            if errors:
                assert "UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.test_script_" in str(ex)
                for f in upload_file_aliases:
                    print(f"f is: {f}")
                    assert f"{f} already exists" in str(ex)
            else:
                # File is the same, so nothing should have happened
                raise ex

        upload_util.check_files(run, upload_file_aliases, set(), 2, upload_folder_name)

    # Step 3, upload a second file with the same alias as the first
    upload_files = test_file_name_sets[1]
    upload_util.copy_test_file_name_set(test_file_name_sets[1])

    for upload_folder_name, upload_fn, errors in upload_datas:
        print(f"Upload a second file with same name as the first: {upload_folder_name}, {upload_files}, \
              this should fail since first file already there")
        name = f"{upload_folder_name}/{test_file_name_alias}"
        try:
            upload_fn(run=run,
                      name=name,
                      path_or_stream=str(test_upload_folder / filenames[1]))
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
