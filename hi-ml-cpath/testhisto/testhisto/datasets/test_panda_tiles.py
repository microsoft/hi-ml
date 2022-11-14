from pathlib import Path
import pytest
import pandas as pd

from health_cpath.datasets.panda_tiles_dataset import PandaTilesDataset
from health_cpath.utils.naming import TileKey
from testhisto.mocks.base_data_generator import MockHistoDataType
from testhisto.mocks.tiles_generator import MockPandaTilesGenerator


@pytest.mark.parametrize("tiling_version", [0, 1])
def test_panda_tiles_dataset(tiling_version: int, tmp_path: Path) -> None:
    """Test the PandaTilesDataset class.

    :param tiling_version: The version of the tiles dataset, defaults to 0. This is used to support both the old
    and new tiling scheme where coordinates are stored as tile_x and tile_y in v0 and as tile_left and tile_top
    in v1.
    :param tmp_path: The temporary path where to store the mock dataset.
    """
    _ = MockPandaTilesGenerator(
        dest_data_path=tmp_path,
        mock_type=MockHistoDataType.FAKE,
        n_tiles=4,
        n_slides=10,
        n_channels=3,
        tile_size=28,
        img_size=224,
        tiling_version=tiling_version,
    )
    base_df = pd.read_csv(tmp_path / PandaTilesDataset.DEFAULT_CSV_FILENAME).set_index(PandaTilesDataset.TILE_ID_COLUMN)
    dataset = PandaTilesDataset(root=tmp_path)

    coordinates_columns_v0 = {PandaTilesDataset.TILE_X_COLUMN, PandaTilesDataset.TILE_Y_COLUMN}
    coordinates_columns_v1 = {TileKey.TILE_LEFT, TileKey.TILE_TOP}
    dataset_columns = set(dataset.dataset_df.columns)
    base_df_columns = set(base_df.columns)

    assert coordinates_columns_v0.issubset(dataset_columns)   # v0 columns are always present

    if tiling_version == 0:
        assert coordinates_columns_v0.issubset(dataset_columns)
        assert not coordinates_columns_v1.issubset(dataset_columns)
        assert base_df_columns == dataset_columns
    elif tiling_version == 1:
        assert coordinates_columns_v1.issubset(dataset_columns)
        assert not coordinates_columns_v0.issubset(base_df_columns)
        assert dataset_columns == base_df_columns.union(coordinates_columns_v0)
