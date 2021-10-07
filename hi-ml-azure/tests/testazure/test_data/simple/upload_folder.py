#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Testing run_upload_folder.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from azureml.core.run import Run

import health.azure.azure_util as util

try:
    import upload_util
except Exception:
    import testazure.test_data.simple.upload_util as upload_util  # type: ignore


@dataclass
class TestUploadFolderData:
    """
    Class to track progress of uploading test file sets and tracking expected results.
    """
    # Name of folder
    folder_name: str
    # Function to use for upload
    upload_fn: Callable
    # Does this work?
    errors: bool
    # Set of files that should have been uploaded without an error
    good_files: set = field(default_factory=set)
    # Set of files that should have been uploaded with an error
    bad_files: set = field(default_factory=set)


def run_test(run: Run) -> None:
    """
    Run a set of tests against run.upload_folder and run_upload_folder.

    :param run: AzureML run.
    """
    # Create test files.
    # [0, 12) Create test files in the root of the base_data folder.
    upload_util.create_test_files(None, range(0, 12))

    # [12, 24) Create test files in a direct sub folder of the base_data folder.
    upload_util.create_test_files(Path("sub1"), range(12, 24))

    # [24, 36) Create test files in a sub sub sub folder of the base_data folder.
    upload_util.create_test_files(Path("sub1") / "sub2" / "sub3", range(24, 36))

    # [36, 45) Create test files in a direct sub folder of the base_data folder, with same filenames as the first set.
    upload_util.create_test_files(Path("sub1"), range(0, 9))

    # [45, 54) Create test files in a sub sub sub folder of the base_data folder, with same filenames as the first set.
    upload_util.create_test_files(Path("sub1") / "sub2" / "sub3", range(0, 9))

    # Extract the list of test file names
    filenames = upload_util.get_test_file_names()

    # Split into distinct sets for each stage of the test
    test_file_name_sets = [
        # 0. Base level files
        set(filenames[:3]),
        # 1. Second set of base level files, distinct from the first
        set(filenames[3:6]),
        # 2. sub1 level files to check folder handling
        set(filenames[12:15]),
        # 3. Second set of sub1 level files, distinct from the first
        set(filenames[15:18]),
        # 4. sub1/sub2/sub3 level files to check folder handling when an extra level inserted
        set(filenames[24:27]),
        # 5. Second set of sub1/sub2/sub3 level files, distinct from the first
        set(filenames[27:30]),
        # 6. Third set of sub1 level files, same filenames as the first
        set(filenames[36:39]),
        # 7. Third set of sub1/sub2/sub3 level files, same filenames as the third
        set(filenames[45:48]),
        # 8. Hold back base level files to test overlaps
        set(filenames[6:9]),
        # 9. Hold back sub1 level files to test overlaps
        set(filenames[18:21]),
        # 10. Hold back sub1/sub2/sub3 level files to test overlaps
        set(filenames[30:33]),
        # 11. Repeat already loaded files to get duplicates, with the same filenames
        set(filenames[0:3]).union(set(filenames[36:39])).union(set(filenames[45:48])),
        # 12. Hold back sub1, and sub1/sub2/sub3 level files to test overlaps, new files, with the same filenames
        set(filenames[9:12]).union(set(filenames[21:24])).union(set(filenames[33:36]))
    ]

    def amlupload_folder(run: Run,
                         name: str,
                         path: Path) -> None:
        """
        Upload a folder using AzureML directly.
        """
        run.upload_folder(name, str(path))

    def himlupload_folder(run: Run,
                          name: str,
                          path: Path) -> None:
        """
        Upload a folder using the HI-ML wrapper function.
        """
        util.run_upload_folder(run, name, str(path))

    # Test against two different methods. AzureML directly and using the HI-ML wrapper
    upload_datas = [
        # Test against AzureML. This takes a long time because of two minute timeouts trying to download
        # corrupted files.
        # TestUploadFolderData("uploaded_folder_aml", amlupload_folder, True),
        # Test against HI-ML wrapper function.
        TestUploadFolderData("uploaded_folder_himl", himlupload_folder, False)
    ]

    test_upload_folder = Path(upload_util.test_upload_folder_name)

    # Step 1, upload distinct file sets
    step = 0
    for i in range(0, 8):
        # Remove any existing test files
        upload_util.rm_test_file_name_set(test_upload_folder)
        # Copy in the new test file set
        upload_util.copy_test_file_name_set(test_upload_folder, test_file_name_sets[i])
        upload_files = test_file_name_sets[i]

        # Upload using each method and check the results
        for upload_data in upload_datas:
            upload_data.good_files = upload_data.good_files.union(test_file_name_sets[i])

            print(f"Upload file set {i}: {upload_data.folder_name}, {upload_files}")

            upload_data.upload_fn(run, upload_data.folder_name, test_upload_folder)

            step = step + 1
            upload_util.check_folder(run=run,
                                     good_filenames=upload_data.good_files,
                                     bad_filenames=upload_data.bad_files,
                                     step=step,
                                     upload_folder_name=upload_data.folder_name)

    # Step 2, upload the overlapping file sets
    for (k, i) in [(1, 8), (3, 9), (5, 10), (11, 12)]:
        upload_util.rm_test_file_name_set(test_upload_folder)
        upload_util.copy_test_file_name_set(test_upload_folder, test_file_name_sets[k])
        upload_util.copy_test_file_name_set(test_upload_folder, test_file_name_sets[i])
        upload_files = test_file_name_sets[k].union(test_file_name_sets[i])

        for upload_data in upload_datas:
            if upload_data.errors:
                print(f"Upload file sets {k} and {i}: {upload_data.folder_name}, {upload_files}, \n \
this should fail, since file set: {test_file_name_sets[k]} already uploaded")

                upload_data.bad_files = upload_data.bad_files.union(test_file_name_sets[i])

                try:
                    upload_data.upload_fn(run, upload_data.folder_name, test_upload_folder)
                except Exception as ex:
                    assert "UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.test_script_" in str(ex)
                    for f in test_file_name_sets[k]:
                        assert f"{f} already exists" in str(ex)

            else:
                print(f"Upload file sets {k} and {i}: {upload_data.folder_name}, {upload_files}, \n \
this should be fine, since overlaps handled")

                upload_data.good_files = upload_data.good_files.union(test_file_name_sets[i])

                upload_data.upload_fn(run, upload_data.folder_name, test_upload_folder)

            step = step + 1
            upload_util.check_folder(run=run,
                                     good_filenames=upload_data.good_files,
                                     bad_filenames=upload_data.bad_files,
                                     step=step,
                                     upload_folder_name=upload_data.folder_name)

    # Step 3, modify the original set
    for k in [1, 3, 5]:
        upload_util.rm_test_file_name_set(test_upload_folder)
        upload_util.copy_test_file_name_set(test_upload_folder, test_file_name_sets[k])
        upload_files = test_file_name_sets[k]

        random_file = list(test_file_name_sets[k])[0]
        random_upload_file = test_upload_folder / random_file
        existing_text = random_upload_file.read_text()
        random_upload_file.write_text("modified... " + existing_text)

        for upload_data in upload_datas:
            if upload_data.errors:
                print(f"Upload file set {k}: {upload_data.folder_name}, {upload_files}, \n \
this should fail, since file set: {test_file_name_sets[k]} already uploaded")

                upload_data.bad_files = upload_data.bad_files.union(test_file_name_sets[k])

                try:
                    upload_data.upload_fn(run, upload_data.folder_name, test_upload_folder)
                except Exception as ex:
                    print(f"Expected error in upload_folder: {str(ex)}")
                    assert "UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.test_script_" in str(ex)
                    for f in test_file_name_sets[k]:
                        assert f"{f} already exists" in str(ex)

            else:
                print(f"Upload file set {k}: {upload_data.folder_name}, {upload_files}, \n \
this should be raise an exception since one of the files has changed")

                try:
                    upload_data.upload_fn(run, upload_data.folder_name, test_upload_folder)
                except Exception as ex:
                    print(f"Expected error in upload_folder: {str(ex)}")
                    assert f"Trying to upload file {random_upload_file} but that file already exists in the run." \
                           "in str(ex)"

            step = step + 1
            upload_util.check_folder(run=run,
                                     good_filenames=upload_data.good_files,
                                     bad_filenames=upload_data.bad_files,
                                     step=step,
                                     upload_folder_name=upload_data.folder_name)
