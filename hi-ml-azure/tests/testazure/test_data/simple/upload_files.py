#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Testing run_upload_files.
"""

from pathlib import Path

from azureml.core.run import Run

import health.azure.azure_util as util

try:
    import upload_util
except Exception:
    import testazure.test_data.simple.upload_util as upload_util  # type: ignore


def run_test(run: Run) -> None:
    """
    Run a set of tests against run_upload_files.

    :param run: AzureML run.
    """
    # Create test files.
    upload_util.create_test_files(None, range(0, 12))

    # Extract the list of test file names
    filenames = upload_util.get_test_file_names()

    # List of name prefixes
    prefixes = [
        "",
        "sub1/",
        "sub1/sub2/sub3/",
    ]

    # List of name suffixes
    suffixes = [
        "test_file0_txt.txt",
        "test_file1_txt.txt",
        "test_file2_txt.txt",
    ]

    test_upload_folder = Path(upload_util.test_upload_folder_name)
    upload_util.copy_test_file_name_set(test_upload_folder, set(filenames))

    # Step 1, upload the first set of three x three files
    step = 0
    names = [f"{prefix}{suffix}" for prefix in prefixes for suffix in suffixes]
    filenames1 = filenames[:9]
    paths = [test_upload_folder / f for f in filenames1]

    print(f"Upload the first file set: {names}={paths}")
    util.run_upload_files(run=run,
                          names=names,
                          paths=paths)

    step = step + 1
    upload_util.check_files(run=run,
                            names=names,
                            filenames=filenames1,
                            step=step)

    # Step 2, upload the first set of three x three files again
    print(f"Upload the first file again: {names}={paths}, "
          "these files should be silently ignored since they are the same.")
    util.run_upload_files(run=run,
                          names=names,
                          paths=paths)

    step = step + 1
    upload_util.check_files(run=run,
                            names=names,
                            filenames=filenames1,
                            step=step)

    # Step 3, upload a second set of three x three files with the same names as the first
    for i in range(0, 3):
        new_paths = paths.copy()
        new_paths[3 * i] = test_upload_folder / filenames[9 + i]

        print(f"Upload a second file with same name as the first: {names}={new_paths}, "
              "this should fail since first file already there")
        try:
            util.run_upload_files(run=run,
                                  names=names,
                                  paths=new_paths)
        except Exception as ex:
            print(f"Expected error in run_upload_files: {str(ex)}")
            name = names[3 * i]
            assert f"Trying to upload file {name} but that file already exists in the run." in str(ex)

    step = step + 1
    upload_util.check_files(run=run,
                            names=names,
                            filenames=filenames1,  # Check that the original files are still there
                            step=step)
