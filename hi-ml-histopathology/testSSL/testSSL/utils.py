#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pydicom
import SimpleITK as sitk
from SSL.data.io_util import PhotometricInterpretation
from health_azure.utils import PathOrString

ENV_RESOURCE_GROUP = "HIML_RESOURCE_GROUP"
ENV_SUBSCRIPTION_ID = "HIML_SUBSCRIPTION_ID"
ENV_WORKSPACE_NAME = "HIML_WORKSPACE_NAME"
WORKSPACE_CONFIG_JSON = "config.json"


TEST_OUTPUTS_PATH = Path(__file__).parent.parent / "test_outputs"


def tests_root_directory(path: Optional[PathOrString] = None) -> Path:
    """
    Gets the full path to the root directory that holds the tests.
    If a relative path is provided then concatenate it with the absolute path
    to the repository root.

    :return: The full path to the repository's root directory, with symlinks resolved if any.
    """
    root = Path(os.path.realpath(__file__)).parent.parent
    return root / path if path else root


def write_test_dicom(array: np.ndarray, path: Path, is_monochrome2: bool = True,
                     bits_stored: Optional[int] = None) -> None:
    """
    This saves the input array as a Dicom file.
    This function DOES NOT create a usable Dicom file and is meant only for testing: tags are set to
    random/default values so that pydicom does not complain when reading the file.
    """

    # Write a file directly with pydicom is cumbersome (all tags need to be set by hand). Hence using simpleITK to
    # create the file. However SimpleITK does not let you set the tags directly, so using pydicom so set them after.
    image = sitk.GetImageFromArray(array)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path))
    writer.Execute(image)

    ds = pydicom.dcmread(path)
    ds.PhotometricInterpretation = PhotometricInterpretation.MONOCHROME2.value if is_monochrome2 else \
        PhotometricInterpretation.MONOCHROME1.value
    if bits_stored is not None:
        ds.BitsStored = bits_stored
    ds.save_as(path)


def get_shared_config_json() -> Path:
    """
    Gets the path to the config.json file that should exist for running tests locally (outside github build agents).
    """
    return tests_root_directory() / WORKSPACE_CONFIG_JSON


@contextmanager
def check_config_json() -> Generator:
    """
    Create a workspace config.json file in the folder where we expect the test scripts. This is either copied
    from the repository root folder (this should be the case when executing a test on a dev machine), or create
    it from environment variables (this should trigger in builds on the github agents).
    """
    target_config_json =  get_shared_config_json()
    if not target_config_json.exists():
        logging.info(f"Creating {str(target_config_json)} from environment variables.")
        subscription_id = os.getenv(ENV_SUBSCRIPTION_ID, "")
        resource_group = os.getenv(ENV_RESOURCE_GROUP, "")
        workspace_name = os.getenv(ENV_WORKSPACE_NAME, "")
        if subscription_id and resource_group and workspace_name:
            with open(str(target_config_json), 'w', encoding="utf-8") as file:
                config = {
                    "subscription_id": os.getenv(ENV_SUBSCRIPTION_ID, ""),
                    "resource_group": os.getenv(ENV_RESOURCE_GROUP, ""),
                    "workspace_name": os.getenv(ENV_WORKSPACE_NAME, "")
                }
                json.dump(config, file)
        else:
            raise ValueError("Either a shared config.json must be present, or all 3 environment variables for "
                             "workspace creation must exist.")
    try:
        yield
    finally:
        target_config_json.unlink()

