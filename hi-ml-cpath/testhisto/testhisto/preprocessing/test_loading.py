#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import shutil
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from _pytest.capture import SysCapture
from pathlib import Path
from typing import List, Optional, Tuple
from monai.transforms import LoadImaged
from monai.data.wsi_reader import CuCIMWSIReader, OpenSlideWSIReader, WSIReader
from health_cpath.datasets.default_paths import PANDA_DATASET_ID
from health_cpath.datasets.panda_dataset import PandaDataset
from health_cpath.preprocessing.loading import (
    BaseLoadROId, LoadingParams, ROIType, WSIBackend, LoadROId, LoadMaskROId, LoadMaskSubROId
)
from health_cpath.scripts.mount_azure_dataset import mount_dataset
from health_cpath.utils.naming import SlideKey
from health_ml.utils.common_utils import is_gpu_available
from PIL import Image
from testhiml.utils_testhiml import DEFAULT_WORKSPACE


no_gpu = not is_gpu_available()


@pytest.mark.parametrize("roi_type", [r for r in ROIType])
@pytest.mark.parametrize("backend", [b for b in WSIBackend])
def test_get_load_roid_transform(backend: WSIBackend, roi_type: ROIType) -> None:
    loading_params = LoadingParams(backend=backend, roi_type=roi_type)
    transform = loading_params.get_load_roid_transform()
    transform_type = {
        ROIType.MASK: LoadMaskROId, ROIType.FOREGROUND: LoadROId, ROIType.WHOLE: LoadImaged,
        ROIType.MASKSUBROI: LoadMaskSubROId,
    }
    assert isinstance(transform, transform_type[roi_type])
    reader_type = {WSIBackend.CUCIM: CuCIMWSIReader, WSIBackend.OPENSLIDE: OpenSlideWSIReader}
    if roi_type in [ROIType.MASK, ROIType.FOREGROUND]:
        assert isinstance(transform, BaseLoadROId)
        assert isinstance(transform.reader, WSIReader)  # type: ignore
        assert isinstance(transform.reader.reader, reader_type[backend])  # type: ignore


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
    roi_type: ROIType, mock_panda_slides_root_dir: Path, capsys: SysCapture
) -> None:
    loading_params = LoadingParams(roi_type=roi_type, level=2)
    load_transform: BaseLoadROId = loading_params.get_load_roid_transform()  # type: ignore
    sample = PandaDataset(mock_panda_slides_root_dir)[0]
    if roi_type == ROIType.MASK:
        os.makedirs(Path(sample[SlideKey.MASK]).parent, exist_ok=True)
        shutil.copy(sample[SlideKey.IMAGE], sample[SlideKey.MASK])  # copy image to mask, we just need a dummy mask
    with patch.object(load_transform, "_get_foreground_mask", return_value=np.zeros((24, 24))):  # empty mask
        with patch.object(load_transform, "_get_whole_slide_bbox") as mock_get_wsi_bbox:
            with patch.object(load_transform.reader, "get_data", return_value=(MagicMock(), MagicMock())):
                _ = load_transform(sample)
                mock_get_wsi_bbox.assert_called_once()
                stdout: str = capsys.readouterr().out  # type: ignore
                assert "Failed to estimate bounding box for slide _0: The input mask is empty" in stdout


@pytest.mark.parametrize("roi_label", [None, 1])
def test_load_mask_sub_roid_roi_label(roi_label: Optional[int]) -> None:
    loading_params = LoadingParams(roi_type=ROIType.MASKSUBROI, level=2, roi_label=roi_label)
    load_transform = loading_params.get_load_roid_transform()
    assert isinstance(load_transform, LoadMaskSubROId)
    assert load_transform.roi_label == roi_label
    mask = np.random.randint(0, 3, size=(4, 4))
    np.random.seed(0)
    section_label = load_transform._get_sub_section_label(mask)
    if roi_label is None:
        assert section_label in [0, 1, 2]
    else:
        assert section_label == roi_label


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_load_mask_sub_roid(mock_panda_slides_root_dir_diagonal: Path, tmp_path: Path) -> None:
    sample = PandaDataset(mock_panda_slides_root_dir_diagonal)[0]
    loading_params = LoadingParams(roi_type=ROIType.MASKSUBROI, level=0, roi_label=1, mask_mag=10.0)
    load_transform = loading_params.get_load_roid_transform()

    # Creat a fake mask
    mask = np.zeros((224, 224, 3))
    mask[56:112, :56, :] = 1  # Set pixels to 1 (foreground), they are all white in the image
    # write a mask as png
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask.astype(np.uint8)).save(mask_path)

    sample[SlideKey.MASK] = str(mask_path)
    slide_dict = load_transform(sample)
    assert SlideKey.IMAGE in slide_dict
    assert slide_dict[SlideKey.IMAGE].shape == (3, 56, 56)
    assert slide_dict[SlideKey.IMAGE].sum() == 255 * 3 * 56 * 56  # all pixels are white
