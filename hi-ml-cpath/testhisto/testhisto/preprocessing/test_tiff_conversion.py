#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from health_cpath.preprocessing.tiff_conversion import ConvertWSIToTiffd


def test_convert_wsi_to_tiff(mock_panda_slides_root_dir: Path, tmp_path: Path) -> None:
    """Test the conversion of the cyted dataset from ndpi to tiff."""
    wsi_path = mock_panda_slides_root_dir / "train" / "_1.tiff"
    transform = ConvertWSIToTiffd(dest_dir=tmp_path, target_magnifications=[])
    transform({"image": wsi_path})
    tiff_wsi = tmp_path / "_1.tiff"
    # assert tiff_wsi == tmp_path / "_1.tiff"
    assert tiff_wsi.exists()
