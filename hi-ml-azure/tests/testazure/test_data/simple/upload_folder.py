#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Testing run_upload_folder.
"""

from pathlib import Path
import random

from azureml.core.run import Run

import health.azure.azure_util as util

try:
    import upload_util
except Exception:
    import testazure.test_data.simple.upload_util as upload_util  # type: ignore


def run_test(run: Run) -> None:
    """
    Run a set of tests against run_upload_folder.

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
        # 0. A set of files at base, sub1, and sub1/sub2/sub3 level to check
        #  file and folder handling.
        set(filenames[:3]).union(set(filenames[12:15])).union(set(filenames[24:27])),
        # 1. A distinct set of files to 0, with different filenames, in the same folders to
        #  check adding distinct files to existing folders.
        set(filenames[3:6]).union(set(filenames[15:18])).union(set(filenames[27:30])),
        # 2. A distinct set of files to 0 and 1, but with the same filenames as 0, in different
        #  folders to check repeating the filename but in a subfolder
        set(filenames[36:39]).union(set(filenames[45:48])),
        # 3. A distinct set of files to 0, 1 and 2, with different filenames to check uploading some
        #  duplicate files and some new files.
        set(filenames[6:9]).union(set(filenames[18:21])).union(set(filenames[30:33])),
        # 4. A set of files that all have the same filenames but different folders, that have already been uploaded
        set(filenames[0:3]).union(set(filenames[36:39])).union(set(filenames[45:48])),
        # 5. A distinct set of files, all with the same filename, but in different folders.
        set(filenames[9:12]).union(set(filenames[21:24])).union(set(filenames[33:36]))
    ]

    folder_name = "uploaded_folder"
    # Set of files that should have been uploaded without an error
    good_filenames = set()

    test_upload_folder = Path(upload_util.test_upload_folder_name)

    # Step 1, upload distinct file sets
    step = 0
    for i in range(0, 3):
        # Remove any existing test files
        upload_util.rm_test_file_name_set(test_upload_folder)
        # Copy in the new test file set
        upload_util.copy_test_file_name_set(test_upload_folder, test_file_name_sets[i])

        upload_files = test_file_name_sets[i]
        good_filenames = good_filenames.union(test_file_name_sets[i])

        print(f"Upload file set {i}: {upload_files}")

        util.run_upload_folder(run=run,
                               name=folder_name,
                               path=test_upload_folder)

        step = step + 1
        upload_util.check_folder(run=run,
                                 good_filenames=good_filenames,
                                 bad_filenames=set(),
                                 step=step,
                                 upload_folder_name=folder_name)

    # Step 2, upload the overlapping file sets
    for (k, i) in [(0, 3), (4, 5)]:
        upload_util.rm_test_file_name_set(test_upload_folder)
        upload_util.copy_test_file_name_set(test_upload_folder, test_file_name_sets[k])
        upload_util.copy_test_file_name_set(test_upload_folder, test_file_name_sets[i])

        upload_files = test_file_name_sets[k].union(test_file_name_sets[i])
        good_filenames = good_filenames.union(test_file_name_sets[i])

        print(f"Upload file sets {k} and {i}: {upload_files}, \n "
              "this should be fine, since overlaps handled")

        util.run_upload_folder(run=run,
                               name=folder_name,
                               path=test_upload_folder)

        step = step + 1
        upload_util.check_folder(run=run,
                                 good_filenames=good_filenames,
                                 bad_filenames=set(),
                                 step=step,
                                 upload_folder_name=folder_name)

    # Step 3, modify the original set
    for k in range(0, 3):
        upload_util.rm_test_file_name_set(test_upload_folder)
        upload_util.copy_test_file_name_set(test_upload_folder, test_file_name_sets[0])

        upload_files = test_file_name_sets[0]

        random_file = random.choice(test_file_name_sets[0])
        random_upload_file = test_upload_folder / random_file
        existing_text = random_upload_file.read_text()
        random_upload_file.write_text("modified... " + existing_text)

        print(f"Upload file set {k}: {upload_files}, \n"
              "this should be raise an exception since one of the files has changed")

        try:
            util.run_upload_folder(run=run,
                                   name=folder_name,
                                   path=test_upload_folder)
        except Exception as ex:
            print(f"Expected error in upload_folder: {str(ex)}")
            assert f"Trying to upload file {random_upload_file} but that file already exists in the run." \
                   "in str(ex)"

        step = step + 1
        upload_util.check_folder(run=run,
                                 good_filenames=good_filenames,
                                 bad_filenames=set(),
                                 step=step,
                                 upload_folder_name=folder_name)
