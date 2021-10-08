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
    import testazure.test_data.simple.upload_util as upload_util  # type: ignore


def run_test(run: Run) -> None:
    """
    Run a set of tests against run_upload_file.
    """
    # Create test files.
    upload_util.create_test_files(None, range(0, 6))

    # Extract the list of test file names
    filenames = upload_util.get_test_file_names()

    # List of name prefixes
    prefixes = [
        "",
        "sub1/",
        "sub1/sub2/sub3/",
    ]

    # Common name suffix
    suffix = "test_file0_txt.txt"

    test_upload_folder = Path(upload_util.test_upload_folder_name)
    upload_util.copy_test_file_name_set(test_upload_folder, set(filenames))

    step = 0
    for i, prefix in enumerate(prefixes):
        name = f"{prefix}{suffix}"
        filename = filenames[i]

        # Step 1, upload the first set of three files
        print(f"Upload the first file: {name}={filename}")
        util.run_upload_file(run=run,
                             name=name,
                             path_or_stream=test_upload_folder / filename)

        step = step + 1
        upload_util.check_file(run=run,
                               name=name,
                               filename=filename,
                               step=step)

        # Step 2, upload the first set of three files again
        print(f"Upload the first file again: {name}={filename}, "
              "these files should be silently ignored since they are the same.")
        util.run_upload_file(run=run,
                             name=name,
                             path_or_stream=test_upload_folder / filename)

        step = step + 1
        upload_util.check_file(run=run,
                               name=name,
                               filename=filename,
                               step=step)

        # Step 3, upload a second set of three files with the same suffix as the first
        new_filename = filenames[i + 3]
        print(f"Upload a second file with same name as the first: {name}={new_filename}, "
              "this should fail since first file already there")
        try:
            util.run_upload_file(run=run,
                                 name=name,
                                 path_or_stream=test_upload_folder / new_filename)
        except Exception as ex:
            print(f"Expected error in run.upload_file: {str(ex)}")
            assert f"Trying to upload file {name} but that file already exists in the run." in str(ex)

        step = step + 1
        upload_util.check_file(run=run,
                               name=name,
                               filename=filename,  # Check that the original file is still there.
                               step=step)
