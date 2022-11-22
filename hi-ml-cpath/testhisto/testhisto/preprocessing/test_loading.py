#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
from pathlib import Path
from typing import List, Tuple
from health_cpath.datasets.default_paths import PANDA_DATASET_ID
from health_cpath.datasets.panda_dataset import PandaDataset
from health_cpath.preprocessing.loading import (
    BaseLoadROId, CucimLoadMaskROId, CucimLoadROId, LoadingParams,
    OpenSlideLoadMaskROId, OpenSlideLoadROId, ROIType, WSIBackend
)
from health_cpath.scripts.mount_azure_dataset import mount_dataset
from health_cpath.utils.naming import SlideKey
from health_ml.utils.common_utils import is_gpu_available
from testhiml.utils_testhiml import DEFAULT_WORKSPACE

no_gpu = not is_gpu_available()


@pytest.mark.parametrize("roi_type", [r for r in ROIType])
def test_get_traansform_args(roi_type: str) -> None:
    loading_params = LoadingParams(roi_type=roi_type)
    args = loading_params.get_transform_args()
    if roi_type == ROIType.MASK:
        assert "mask_key" in args
    elif roi_type == ROIType.FOREGROUND:
        assert "foreground_threshold" in args


@pytest.mark.parametrize("roi_type", [r for r in ROIType])
@pytest.mark.parametrize("backend", [b for b in WSIBackend])
def test_get_load_roid_transform(backend: str, roi_type: str) -> None:
    loading_params = LoadingParams(backend=backend, roi_type=roi_type)
    transform = loading_params.get_load_roid_transform()
    if backend == WSIBackend.CUCIM:
        assert "Cucim" in transform.__class__.__name__
    elif backend == WSIBackend.OPENSLIDE:
        assert "OpenSlide" in transform.__class__.__name__
    if roi_type == ROIType.MASK:
        assert "Mask" in transform.__class__.__name__
    elif roi_type == ROIType.FOREGROUND:
        assert "Mask" not in transform.__class__.__name__


@pytest.mark.skip(reason="This test is failing because of issue #655")
@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
def test_load_slide(tmp_path: Path) -> None:
    _ = mount_dataset(dataset_id=PANDA_DATASET_ID, tmp_root=str(tmp_path), aml_workspace=DEFAULT_WORKSPACE.workspace)
    root_path = tmp_path / PANDA_DATASET_ID

    def _check_load_roi_transforms(
        load_transform: BaseLoadROId, expected_keys: List[SlideKey], expected_shape: Tuple[int, int, int]
    ) -> None:
        sample = PandaDataset(root_path)[0]
        slide_dict = load_transform(sample)
        assert all([k in slide_dict for k in expected_keys])
        assert slide_dict[SlideKey.IMAGE].shape == expected_shape

    # Foreground ROI transforms
    foreground_expected_keys = [SlideKey.ORIGIN, SlideKey.SCALE, SlideKey.FOREGROUND_THRESHOLD, SlideKey.IMAGE]
    foreground_expected_shape = (3, 1342, 340)
    _check_load_roi_transforms(OpenSlideLoadROId(), foreground_expected_keys, foreground_expected_shape)
    _check_load_roi_transforms(CucimLoadROId(), foreground_expected_keys, foreground_expected_shape)

    # Mask ROI transforms
    mask_expected_keys = [SlideKey.ORIGIN, SlideKey.SCALE, SlideKey.IMAGE]
    mask_expected_shape = (3, 1344, 341)
    _check_load_roi_transforms(CucimLoadMaskROId(), mask_expected_keys, mask_expected_shape)
    _check_load_roi_transforms(OpenSlideLoadMaskROId(), mask_expected_keys, mask_expected_shape)
