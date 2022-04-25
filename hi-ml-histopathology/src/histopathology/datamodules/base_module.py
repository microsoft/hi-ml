#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, Sequence, Tuple, TypeVar, Union

from monai.data.dataset import CacheDataset, Dataset, PersistentDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from health_ml.utils.bag_utils import BagDataset, multibag_collate
from health_ml.utils.common_utils import _create_generator

from histopathology.utils.wsi_utils import image_collate
from histopathology.models.transforms import LoadTilesBatchd
from histopathology.datasets.base_dataset import SlidesDataset, TilesDataset

from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.apps.pathology.transforms import TileOnGridd
from monai.data.image_reader import WSIReader

_SlidesOrTilesDataset = TypeVar('_SlidesOrTilesDataset', SlidesDataset, TilesDataset)


class CacheMode(Enum):
    NONE = "none"
    MEMORY = "memory"
    DISK = "disk"


class CacheLocation(Enum):
    NONE = "none"
    CPU = "cpu"
    SAME = "same"


class HistoDataModule(LightningDataModule, Generic[_SlidesOrTilesDataset]):
    """Base class to load a histopathology dataset as train, val, test sets"""

    def __init__(
        self,
        root_path: Path,
        batch_size: int = 1,
        seed: Optional[int] = None,
        transform: Optional[Callable] = None,
        crossval_count: int = 0,
        crossval_index: int = 0,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param root_path: Root directory of the source dataset.
        :param batch_size: Number of slides to load per batch.
        :param seed: pseudorandom number generator seed to use for shuffling instances and bags. Note that randomness in
        train/val/test splits is handled independently in `get_splits()`. (default: `None`)
        :param transform: A transform to apply to the source tiles dataset, or a composition of
        transforms using `monai.transforms.Compose`. By default (`None`).
        :param crossval_count: Number of folds to perform.
        :param crossval_index: Index of the cross validation split to be performed.
        :param dataloader_kwargs: Additional keyword arguments for the training, validation, and test dataloaders.
        """

        super().__init__()

        self.root_path = root_path
        self.transform = transform
        self.batch_size = batch_size
        self.crossval_count = crossval_count
        self.crossval_index = crossval_index
        self.train_dataset: _SlidesOrTilesDataset
        self.val_dataset: _SlidesOrTilesDataset
        self.test_dataset: _SlidesOrTilesDataset
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_splits()
        self.class_weights = self.train_dataset.get_class_weights()
        self.seed = seed
        self.dataloader_kwargs = dataloader_kwargs or {}

    def get_splits(self) -> Tuple[_SlidesOrTilesDataset, _SlidesOrTilesDataset, _SlidesOrTilesDataset]:
        """Create the training, validation, and test datasets"""
        raise NotImplementedError


class TilesDataModule(HistoDataModule[TilesDataset]):
    """Base class to load the tiles of a dataset as train, val, test sets"""

    def __init__(
        self,
        max_bag_size: int = 0,
        max_bag_size_inf: int = 0,
        cache_mode: CacheMode = CacheMode.NONE,
        precache_location: CacheLocation = CacheLocation.NONE,
        cache_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        """
        :param max_bag_size: Upper bound on number of tiles in each loaded bag during training stage. If 0 (default),
        will return all samples in each bag. If > 0 , bags larger than `max_bag_size` will yield
        random subsets of instances.
        :param max_bag_size_inf: Upper bound on number of tiles in each loaded bag during validation and test stages.
        If 0 (default), will return all samples in each bag. If > 0 , bags larger than `max_bag_size_inf` will yield
        random subsets of instances.
        :param cache_mode: The type of caching to perform, i.e. whether the results of all
        transforms up to the first randomised one should be computed only once and reused in
        subsequent iterations:
          - `MEMORY`: MONAI CacheDataset is used, the entire transformed dataset is kept in memory for fastest access;
          - `DISK`: MONAI PersistentDataset is used, each transformed sample is saved to disk and loaded on-demand;
          - `NONE` (default): standard MONAI dataset is used, no caching is performed.
        :param precache_location: Whether to pre-cache the entire transformed dataset upfront and save
        it to disk. This is done once in `prepare_data()` only on the local rank-0 process, so
        multiple processes can afterwards access the same cache without contention in DDP settings.
        This parameter also allows us to choose if the cache will be re-loaded into CPU or GPU memory:
          - `NONE (default)`: no pre-cache is performed;
          - `CPU`: each transformed sample is saved to disk and, if cache_mode is `MEMORY`, reloaded into CPU;
          - `SAME`: each transformed sample is saved to disk and, if cache_mode is `MEMORY`, reloaded on the same
          device it was saved from;
        If cache_mode is `DISK` precache_location `CPU` and `GPU` are equivalent.
        :param cache_dir: The directory onto which to cache data if caching is enabled.
        """
        if precache_location is not CacheLocation.NONE and cache_mode is CacheMode.NONE:
            raise ValueError("Can only pre-cache if caching is enabled")
        if precache_location is not CacheLocation.NONE and cache_dir is None:
            raise ValueError("A cache directory is required for pre-caching")
        if cache_mode is CacheMode.DISK and cache_dir is None:
            raise ValueError("A cache directory is required for on-disk caching")

        self.max_bag_size = max_bag_size
        self.max_bag_size_inf = max_bag_size_inf
        self.cache_mode = cache_mode
        self.precache_location = precache_location
        self.cache_dir = cache_dir

        super().__init__(**kwargs)

    def prepare_data(self) -> None:
        if self.precache_location != CacheLocation.NONE:
            self._load_dataset(self.train_dataset, stage="train", shuffle=True)
            self._load_dataset(self.val_dataset, stage="val", shuffle=True)
            self._load_dataset(self.test_dataset, stage="test", shuffle=True)

    def _dataset_pickle_path(self, stage: str) -> Optional[Path]:
        if self.cache_dir is None or self.cache_mode == CacheMode.NONE:
            return None
        return self.cache_dir / f"{stage}_dataset.pt"

    def _get_transformed_dataset(self, dataset: Dataset, transform: Union[Sequence[Callable], Callable]) -> Dataset:
        if self.cache_mode is CacheMode.MEMORY:
            dataset = CacheDataset(dataset, transform, num_workers=1)  # type: ignore
        elif self.cache_mode is CacheMode.DISK:
            dataset = PersistentDataset(dataset, transform, cache_dir=self.cache_dir)  # type: ignore
            if self.precache_location != CacheLocation.NONE:
                import tqdm  # TODO: Make optional

                for i in tqdm.trange(len(dataset), desc="Loading dataset"):
                    dataset[i]  # empty loop to pre-compute all transformed samples
        else:
            dataset = Dataset(dataset, transform)  # type: ignore
        return dataset

    def _load_dataset(self, tiles_dataset: TilesDataset, stage: str, shuffle: bool) -> Dataset:
        dataset_pickle_path = self._dataset_pickle_path(stage)

        if dataset_pickle_path and dataset_pickle_path.is_file():
            if self.precache_location == CacheLocation.CPU:
                memory_location = torch.device("cpu")
                print(f"Loading dataset from {dataset_pickle_path} into {memory_location}")
            else:
                # by default torch.load will reload on the same device it was saved from
                memory_location = None  # type: ignore

            with dataset_pickle_path.open("rb") as f:
                return torch.load(f, map_location=memory_location)

        generator = _create_generator(self.seed)

        if stage in ["val", "test"]:
            eff_max_bag_size = self.max_bag_size_inf
        else:
            eff_max_bag_size = self.max_bag_size

        bag_dataset = BagDataset(
            tiles_dataset,  # type: ignore
            bag_ids=tiles_dataset.slide_ids,
            max_bag_size=eff_max_bag_size,
            shuffle_samples=shuffle,
            generator=generator,
        )
        transform = self.transform or LoadTilesBatchd(tiles_dataset.IMAGE_COLUMN)

        # Save and restore PRNG state for consistency across (pre-)caching options
        generator_state = generator.get_state()
        transformed_bag_dataset = self._get_transformed_dataset(bag_dataset, transform)  # type: ignore
        generator.set_state(generator_state)

        # Dataset is saved if cache_dir is True, regardless of CacheMode
        if dataset_pickle_path:
            dataset_pickle_path.parent.mkdir(parents=True, exist_ok=True)
            with dataset_pickle_path.open("wb") as f:
                torch.save(transformed_bag_dataset, f)

        return transformed_bag_dataset

    def _get_dataloader(self, dataset: TilesDataset, stage: str, shuffle: bool, **dataloader_kwargs: Any) -> DataLoader:
        transformed_bag_dataset = self._load_dataset(dataset, stage=stage, shuffle=shuffle)
        bag_dataset: BagDataset = transformed_bag_dataset.data  # type: ignore
        generator = bag_dataset.bag_sampler.generator
        return DataLoader(
            transformed_bag_dataset,
            batch_size=self.batch_size,
            collate_fn=multibag_collate,
            shuffle=shuffle,
            generator=generator,
            **dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, "train", shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, "val", shuffle=True, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dataset, "test", shuffle=True, **self.dataloader_kwargs)


class SlidesDataModule(HistoDataModule[SlidesDataset]):
    """
    Base class to load the slides of a dataset as train, val, test sets. The slide data module performs tiling on the
    fly by default. One can specify the tiling strategies (background removal, overlapping tiles, padding, ...) through
    the class parameters.
    """

    def __init__(
        self,
        level: Optional[int] = 0,
        tile_count: Optional[int] = None,
        tile_size: Optional[int] = 224,
        step: Optional[int] = None,
        random_offset: Optional[bool] = True,
        pad_full: Optional[bool] = False,
        background_val: Optional[int] = 255,
        filter_mode: Optional[str] = "min",
        **kwargs: Any,
    ) -> None:
        """
        :param level: the whole slide image level at which the image is extracted, defaults to 0
        this param is passed to the LoadImaged monai transform that loads a WSI with cucim backend
        :param tile_count: number of tiles to extract, if None extracts all non-background tiles, defaults to None
        this param is passed to TileOnGridd monai transform for tiling on the fly.
        :param tile_size: size of the square tile, defaults to 224
        this param is passed to TileOnGridd monai transform for tiling on the fly.
        :param step: step size to create overlapping tiles, defaults to None (same as tile_size)
        Use a step < tile_size to create overlapping tiles, analogousely a step > tile_size will skip some chunks in
        the wsi. This param is passed to TileOnGridd monai transform for tiling on the fly.
        :param random_offset: randomize position of the grid, instead of starting from the top-left corner,
        defaults to True. This param is passed to TileOnGridd monai transform for tiling on the fly.
        :param pad_full: pad image to the size evenly divisible by tile_size, defaults to False
        This param is passed to TileOnGridd monai transform for tiling on the fly.
        :param background_val: the background constant to ignore background tiles (e.g. 255 for white background),
        defaults to 255. This param is passed to TileOnGridd monai transform for tiling on the fly.
        :param filter_mode: mode must be in ["min", "max", "random"]. If total number of tiles is greater than
        tile_count, then sort by intensity sum, and take the smallest (for min), largest (for max) or random (for
        random) subset, defaults to "min" (which assumes background is high value). This param is passed to TileOnGridd
        monai transform for tiling on the fly.
        """
        super().__init__(**kwargs)
        self.level = level
        self.tile_count = tile_count
        self.tile_size = tile_size
        self.step = step
        self.random_offset = random_offset
        self.pad_full = pad_full
        self.background_val = background_val
        self.filter_mode = filter_mode
        if self.tile_count is None:
            assert self.batch_size == 1, "batch_size > 1 not supported if tiles_count=None 'for now'"

    def _load_dataset(self, slides_dataset: SlidesDataset) -> Dataset:
        base_transform = Compose(
            [
                LoadImaged(
                    keys=slides_dataset.IMAGE_COLUMN,
                    reader=WSIReader,
                    backend="cuCIM",
                    dtype=np.float32,
                    level=self.level,
                    image_only=True,
                ),
                TileOnGridd(
                    keys=slides_dataset.IMAGE_COLUMN,
                    tile_count=self.tile_count,
                    tile_size=self.tile_size,
                    step=self.step,
                    random_offset=self.random_offset,
                    pad_full=self.pad_full,
                    background_val=self.background_val,
                    filter_mode=self.filter_mode,
                    return_list_of_dicts=True,
                ),
            ]
        )
        transforms = Compose([base_transform, self.transform]).flatten() if self.transform else base_transform
        return Dataset(slides_dataset, transforms)

    def _get_dataloader(self, dataset: SlidesDataset, shuffle: bool, **dataloader_kwargs: Any) -> DataLoader:
        transformed_slides_dataset = self._load_dataset(dataset)
        generator = _create_generator(self.seed)
        return DataLoader(
            transformed_slides_dataset,
            batch_size=self.batch_size,
            collate_fn=image_collate,
            shuffle=shuffle,
            generator=generator,
            **dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, shuffle=True, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dataset, shuffle=True, **self.dataloader_kwargs)
