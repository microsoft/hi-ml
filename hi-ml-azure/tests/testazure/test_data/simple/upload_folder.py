#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Testing run_upload_folder.
"""

from pathlib import Path

from azureml.core.run import Run

import upload_util
import health.azure.azure_util as util


def run_test(run: Run) -> None:
    """
    Run a set of tests against run.upload_folder and run_upload_folder.
    """
    # Extract the list of test file names
    filenames = upload_util.get_base_data_filenames()

    # Split into distinct sets for each stage of the test
    test_file_name_sets = [
        # 0. Base level files
        set(filenames[:3]),
        # 1. Second set of base level files, distinct from the first
        set(filenames[3:6]),
        # 2. sub1 level files to check folder handling
        set(filenames[9:12]),
        # 3. Second set of sub1 level files, distinct from the first
        set(filenames[12:15]),
        # 4. sub1/sub2/sub3 level files to check folder handling when an extra level inserted
        set(filenames[18:21]),
        # 5. Second set of sub1/sub2/sub3 level files, distinct from the first
        set(filenames[21:24]),
        # 6. Hold back base level files to test overlaps
        set(filenames[6:9]),
        # 7. Hold back sub1 level files to test overlaps
        set(filenames[15:18]),
        # 8. Hold back sub1/sub2/sub3 level files to test overlaps
        set(filenames[24:27]),
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
        # upload_util.TestUploadFolderData("uploaded_folder_aml", amlupload_folder, True),
        # Test against HI-ML wrapper function.
        upload_util.TestUploadFolderData("uploaded_folder_himl", himlupload_folder, False)
    ]

    test_upload_folder = Path(upload_util.test_upload_folder_name)

    # Step 1, upload distinct file sets
    for i in range(0, 6):
        # Remove any existing test files
        upload_util.rm_test_file_name_set()
        # Copy in the new test file set
        upload_util.copy_test_file_name_set(test_file_name_sets[i])

        # Upload using each method and check the results
        for upload_data in upload_datas:
            upload_data.upload_files = upload_data.upload_files.union(test_file_name_sets[i])
            upload_data.good_files = upload_data.good_files.union(test_file_name_sets[i])
            upload_data.bad_files = upload_data.bad_files.union(set())

            print(f"Upload file set {i}: {upload_data.folder_name}, {upload_data.upload_files}")

            upload_data.upload_fn(run, upload_data.folder_name, test_upload_folder)
            upload_util.check_files(run, upload_data.good_files, upload_data.bad_files, i, upload_data.folder_name)

    # Step 2, upload the overlapping file sets
    for (i, j) in [(1, 6), (3, 7), (5, 8)]:
        upload_util.rm_test_file_name_set()
        upload_util.copy_test_file_name_set(test_file_name_sets[i])
        upload_util.copy_test_file_name_set(test_file_name_sets[j])

        for upload_data in upload_datas:
            upload_data.upload_files = upload_data.upload_files.union(test_file_name_sets[j])

            if upload_data.errors:
                print(f"Upload file sets {i} and {j}: {upload_data.folder_name}, {upload_data.upload_files}, \n \
this should fail, since file set: {test_file_name_sets[i]} already uploaded")

                upload_data.good_files = upload_data.good_files.union()
                upload_data.bad_files = upload_data.bad_files.union(test_file_name_sets[j])

                try:
                    upload_data.upload_fn(run, upload_data.folder_name, test_upload_folder)
                except Exception as ex:
                    assert "UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.test_script_" in str(ex)
                    for f in test_file_name_sets[i]:
                        assert f"{f} already exists" in str(ex)

            else:
                print(f"Upload file sets {i} and {j}: {upload_data.folder_name}, {upload_data.upload_files}, \n \
this should be fine, since overlaps handled")

                upload_data.good_files = upload_data.good_files.union(test_file_name_sets[j])
                upload_data.bad_files = upload_data.bad_files.union(set())

                upload_data.upload_fn(run, upload_data.folder_name, test_upload_folder)

            upload_util.check_files(run, upload_data.good_files, upload_data.bad_files, j, upload_data.folder_name)

    # Step 3, modify the original set
    for i in [1, 3, 5]:
        upload_util.rm_test_file_name_set()
        upload_util.copy_test_file_name_set(test_file_name_sets[i])
        random_file = list(test_file_name_sets[i])[0]
        random_upload_file = test_upload_folder / random_file
        existing_text = random_upload_file.read_text()
        random_upload_file.write_text("modified... " + existing_text)

        for upload_data in upload_datas:
            upload_data.good_files = upload_data.good_files.union(set())
            upload_data.upload_files = upload_data.upload_files.union(set())

            if upload_data.errors:
                print(f"Upload file set {i}: {upload_data.folder_name}, {upload_data.upload_files}, \n \
this should fail, since file set: {test_file_name_sets[i]} already uploaded")

                upload_data.bad_files = upload_data.bad_files.union(test_file_name_sets[i])

                try:
                    upload_data.upload_fn(run, upload_data.folder_name, test_upload_folder)
                except Exception as ex:
                    assert "UserError: Resource Conflict: ArtifactId ExperimentRun/dcid.test_script_" in str(ex)
                    for f in test_file_name_sets[i]:
                        assert f"{f} already exists" in str(ex)

            else:
                print(f"Upload file set {i}: {upload_data.folder_name}, {upload_data.upload_files}, \n \
this should be raise an exception since one of the files has changed")

                upload_data.bad_files = upload_data.bad_files.union(set())

                try:
                    upload_data.upload_fn(run, upload_data.folder_name, test_upload_folder)
                except Exception as ex:
                    print(f"Expected error in upload_folder: {str(ex)}")
                    assert f"Trying to upload file {random_upload_file} but that file already exists in the run." \
                           "in str(ex)"

            j = j + 1
            upload_util.check_files(run, upload_data.good_files, upload_data.bad_files, j, upload_data.folder_name)
