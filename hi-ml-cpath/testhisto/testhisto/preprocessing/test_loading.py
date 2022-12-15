#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import shutil
from unittest.mock import patch
import numpy as np
import pytest
from pathlib import Path
from typing import List, Tuple
from monai.transforms import LoadImaged
from monai.data.wsi_reader import CuCIMWSIReader, OpenSlideWSIReader, WSIReader
from health_cpath.datasets.default_paths import PANDA_DATASET_ID
from health_cpath.datasets.panda_dataset import PandaDataset
from health_cpath.preprocessing.loading import BaseLoadROId, LoadingParams, ROIType, WSIBackend, LoadROId, LoadMaskROId
from health_cpath.scripts.mount_azure_dataset import mount_dataset
from health_cpath.utils.naming import SlideKey
from health_ml.utils.common_utils import is_gpu_available
from testhiml.utils_testhiml import DEFAULT_WORKSPACE

no_gpu = not is_gpu_available()


@pytest.mark.parametrize("roi_type", [r for r in ROIType])
@pytest.mark.parametrize("backend", [b for b in WSIBackend])
def test_get_load_roid_transform(backend: WSIBackend, roi_type: ROIType) -> None:
    loading_params = LoadingParams(backend=backend, roi_type=roi_type)
    transform = loading_params.get_load_roid_transform()
    transform_type = {ROIType.MASK: LoadMaskROId, ROIType.FOREGROUND: LoadROId, ROIType.WHOLE: LoadImaged}
    assert isinstance(transform, transform_type[roi_type])
    reader_type = {WSIBackend.CUCIM: CuCIMWSIReader, WSIBackend.OPENSLIDE: OpenSlideWSIReader}
    if roi_type in [ROIType.MASK, ROIType.FOREGROUND]:
        assert isinstance(transform, BaseLoadROId)
        assert isinstance(transform.reader, WSIReader)  # type: ignore
        assert isinstance(transform.reader.reader, reader_type[backend])  # type: ignore


@pytest.mark.skip(reason="This test is failing because of issue #655")
@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_load_slide(tmp_path: Path) -> None:
    _ = mount_dataset(dataset_id=PANDA_DATASET_ID, tmp_root=str(tmp_path), aml_workspace=DEFAULT_WORKSPACE.workspace)
    root_path = tmp_path / PANDA_DATASET_ID

    def _check_load_roi_transforms(
        backend: WSIBackend, expected_keys: List[SlideKey], expected_shape: Tuple[int, int, int]
    ) -> None:
        loading_params.backend = backend
        load_transform = loading_params.get_load_roid_transform()
        sample = PandaDataset(root_path)[0]
        slide_dict = load_transform(sample)
        assert all([k in slide_dict for k in expected_keys])
        assert slide_dict[SlideKey.IMAGE].shape == expected_shape

    # WSI ROIType
    loading_params = LoadingParams(roi_type=ROIType.WHOLE, level=2)
    wsi_expected_keys = [SlideKey.IMAGE, SlideKey.SLIDE_ID]
    wsi_expected_shape = (3, 1840, 1728)
    for backend in [WSIBackend.CUCIM, WSIBackend.OPENSLIDE]:
        _check_load_roi_transforms(backend, wsi_expected_keys, wsi_expected_shape)

    # Foreground ROIType
    loading_params = LoadingParams(roi_type=ROIType.FOREGROUND, level=2)
    foreground_expected_keys = [SlideKey.ORIGIN, SlideKey.SCALE, SlideKey.FOREGROUND_THRESHOLD, SlideKey.IMAGE]
    foreground_expected_shape = (3, 1342, 340)
    for backend in [WSIBackend.CUCIM, WSIBackend.OPENSLIDE]:
        _check_load_roi_transforms(backend, foreground_expected_keys, foreground_expected_shape)

    # Mask ROI transforms
    loading_params = LoadingParams(roi_type=ROIType.MASK, level=2)
    mask_expected_keys = [SlideKey.ORIGIN, SlideKey.SCALE, SlideKey.IMAGE]
    mask_expected_shape = (3, 1344, 341)
    for backend in [WSIBackend.CUCIM, WSIBackend.OPENSLIDE]:
        _check_load_roi_transforms(backend, mask_expected_keys, mask_expected_shape)


@pytest.mark.parametrize("roi_type", [ROIType.FOREGROUND, ROIType.MASK])
@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_failed_to_estimate_foreground(
    roi_type: ROIType, mock_panda_slides_root_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("WARNING")
    loading_params = LoadingParams(roi_type=roi_type, level=2)
    load_transform = loading_params.get_load_roid_transform()
    sample = PandaDataset(mock_panda_slides_root_dir)[0]
    if roi_type == ROIType.MASK:
        os.makedirs(Path(sample[SlideKey.MASK]).parent, exist_ok=True)
        shutil.copy(sample[SlideKey.IMAGE], sample[SlideKey.MASK])  # copy image to mask
    with patch.object(load_transform, "_get_foreground_mask", return_value=np.zeros((24, 24))):  # empty mask
        with patch.object(load_transform, "_get_whole_slide_bbox") as mock_get_wsi_bbox:
            _ = load_transform(sample)
            mock_get_wsi_bbox.assert_called_once()
            assert "Failed to estimate bounding box for slide _0: The input mask is empty" in caplog.text
