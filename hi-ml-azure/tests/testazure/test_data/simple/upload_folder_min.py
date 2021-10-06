#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Testing run_upload_folder, minimal version.
"""

from pathlib import Path

from azureml.core.run import Run

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
    Run a minimal set of tests against run.upload_folder.
    """

    upload_folder_name = "uploaded_folder"

    # Extract the list of test file names
    filenames = upload_util.get_base_data_filenames()

    test_file_name_sets = [
        {filenames[0]},
        {filenames[1]}
    ]

    test_upload_folder = Path(upload_util.test_upload_folder_name)

    # Step 1, upload the first file set
    upload_files = test_file_name_sets[0].copy()
    upload_util.copy_test_file_name_set(test_file_name_sets[0])

    print(f"Upload the first file set: {upload_files}")
    run.upload_folder(name=upload_folder_name, path=str(test_upload_folder))

    upload_util.check_files(run, test_file_name_sets[0], set(), 1, upload_folder_name)

    # Step 2, upload the second file set
    upload_files = upload_files.union(test_file_name_sets[1])
    upload_util.copy_test_file_name_set(test_file_name_sets[1])

    print(f"Upload the second file set: {upload_files}, \
          this should fail since first file set already there")
    try:
        run.upload_folder(name=upload_folder_name, path=str(test_upload_folder))
    except Exception as ex:
        assert "UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.test_script_" in str(ex)
        for f in test_file_name_sets[0]:
            assert f"{f} already exists" in str(ex)

    upload_util.check_files(run, test_file_name_sets[0], test_file_name_sets[1], 2, upload_folder_name)
