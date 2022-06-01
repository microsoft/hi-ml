import torch
import pytest
import numpy as np

from typing import Any, Dict, List
from typing import Sequence
from histopathology.utils.naming import SlideKey
from histopathology.utils.wsi_utils import image_collate
from torch.utils.data import Dataset


class MockTiledWSIDataset(Dataset):
    def __init__(self,
                 n_tiles: int,
                 n_slides: int,
                 n_classes: int,
                 tile_size: Sequence[int],
                 random_n_tiles: bool) -> None:

        self.n_tiles = n_tiles
        self.n_slides = n_slides
        self.tile_size = tile_size
        self.n_classes = n_classes
        self.random_n_tiles = random_n_tiles
        self.slide_ids = torch.arange(self.n_slides)

    def __len__(self) -> int:
        return self.n_slides

    def __getitem__(self, index: int) -> List[Dict[SlideKey, Any]]:
        tile_count = np.random.randint(self.n_tiles) if self.random_n_tiles else self.n_tiles
        label = np.random.choice(self.n_classes)
        return [{SlideKey.SLIDE_ID: self.slide_ids[index],
                 SlideKey.IMAGE: np.random.randint(0, 255, size=self.tile_size),
                 SlideKey.IMAGE_PATH: f"slide_{self.slide_ids[index]}.tiff",
                 SlideKey.LABEL: label
                 } for _ in range(tile_count)
                ]


@pytest.mark.parametrize("random_n_tiles", [False, True])
def test_image_collate(random_n_tiles: bool) -> None:
    # random_n_tiles accounts for both train and inference settings where the number of tiles is fixed (during
    # training) and None during inference (validation and test)
    dataset = MockTiledWSIDataset(n_tiles=20,
                                  n_slides=10,
                                  n_classes=4,
                                  tile_size=(1, 4, 4),
                                  random_n_tiles=random_n_tiles)

    batch_size = 5
    samples_list = [dataset[idx] for idx in range(batch_size)]

    batch: dict = image_collate(samples_list)

    assert isinstance(batch, Dict)
    assert batch.keys() == samples_list[0].keys()  # type: ignore

    for key, value_list in batch.items():
        assert isinstance(value_list, List)
        assert len(value_list) == batch_size
        if key == SlideKey.IMAGE_PATH:
            assert all((value_list[idx] == samples_list[idx][key]) for idx in range(batch_size))
        else:
            assert all(torch.equal(value_list[idx], samples_list[idx][key]) for idx in range(batch_size))
