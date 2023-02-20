#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import torch
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

from pytorch_lightning import LightningDataModule
from pytorch_lightning.overrides.distributed import UnrepeatedDistributedSampler
from torch.utils.data import DataLoader, DistributedSampler
from health_cpath.preprocessing.loading import LoadingParams

from health_ml.utils.bag_utils import BagDataset, multibag_collate
from health_ml.utils.common_utils import _create_generator

from health_cpath.utils.wsi_utils import TilingParams, image_collate
from health_cpath.models.transforms import LoadTilesBatchd
from health_cpath.datasets.base_dataset import SlidesDataset, TilesDataset
from health_cpath.utils.naming import ModelKey

from monai.data.dataset import CacheDataset, Dataset, PersistentDataset
from monai.transforms import Compose


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
        batch_size_inf: Optional[int] = None,
        max_bag_size: int = 0,
        max_bag_size_inf: int = 0,
        seed: Optional[int] = None,
        transforms_dict: Optional[Dict[ModelKey, Union[Callable, None]]] = None,
        crossval_count: int = 0,
        crossval_index: int = 0,
        pl_replace_sampler_ddp: bool = True,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        dataframe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param root_path: Root directory of the source dataset.
        :param batch_size: Number of slides to load per batch.
        :param batch_size_inf: Number of slides to load per batch during inference. If None, use batch_size.
        :param max_bag_size: Upper bound on number of tiles in each loaded bag during training stage. If 0 (default),
        will return all samples in each bag. If > 0 , bags larger than `max_bag_size` will yield
        random subsets of instances. For SlideDataModule, this parameter is used in Rand/GridPatchd Transform to set the
        num_patches used for tiling on the fly at training time.
        :param max_bag_size_inf: Upper bound on number of tiles in each loaded bag during validation and test stages.
        If 0 (default), will return all samples in each bag. If > 0 , bags larger than `max_bag_size_inf` will yield
        random subsets of instances. For SlideDataModule, this parameter is used in Rand/GridPatchd Transform to set the
        num_patches used for tiling on the fly at validation and test time.
        :param seed: pseudorandom number generator seed to use for shuffling instances and bags. Note that randomness in
        train/val/test splits is handled independently in `get_splits()`. (default: `None`)
        :param transforms_dict: A dictionary that contains transform, or a composition of transforms using
        `monai.transforms.Compose`, to apply to the source dataset at training, validation and testing time.
        By default (`None`).
        :param crossval_count: Number of folds to perform.
        :param crossval_index: Index of the cross validation split to be performed.
        :param pl_replace_sampler_ddp: If True, replace the sampler with a DistributedSampler when using DDP.
        :param dataloader_kwargs: Additional keyword arguments for the training, validation, and test dataloaders.
        :param dataframe_kwargs: Keyword arguments to pass to `pd.read_csv()` when loading the dataset CSV.
        """

        batch_size_inf = batch_size_inf or batch_size
        super().__init__()

        self.root_path = root_path
        self.transforms_dict = transforms_dict
        self.batch_sizes = {ModelKey.TRAIN: batch_size, ModelKey.VAL: batch_size_inf, ModelKey.TEST: batch_size_inf}
        self.bag_sizes = {ModelKey.TRAIN: max_bag_size, ModelKey.VAL: max_bag_size_inf, ModelKey.TEST: max_bag_size_inf}
        self.crossval_count = crossval_count
        self.crossval_index = crossval_index
        self.pl_replace_sampler_ddp = pl_replace_sampler_ddp
        self.train_dataset: _SlidesOrTilesDataset
        self.val_dataset: _SlidesOrTilesDataset
        self.test_dataset: _SlidesOrTilesDataset
        self.dataframe_kwargs = dataframe_kwargs or {}
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_splits()
        self.class_weights = self.train_dataset.get_class_weights()
        self.seed = seed
        self.dataloader_kwargs = dataloader_kwargs or {}

    def get_splits(self) -> Tuple[_SlidesOrTilesDataset, _SlidesOrTilesDataset, _SlidesOrTilesDataset]:
        """Create the training, validation, and test datasets"""
        raise NotImplementedError

    def _get_dataloader(
        self, dataset: _SlidesOrTilesDataset, stage: ModelKey, shuffle: bool, **dataloader_kwargs: Any
    ) -> DataLoader:
        raise NotImplementedError

    def _get_ddp_sampler(self, dataset: Dataset, stage: ModelKey) -> Optional[DistributedSampler]:
        is_distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        if is_distributed and not self.pl_replace_sampler_ddp:
            assert self.seed is not None, "seed must be set when using distributed training for reproducibility"
            if stage == ModelKey.TRAIN:
                logging.info("pl_replace_sampler_ddp is False, setting DistributedSampler for training dataloader.")
                return DistributedSampler(dataset, shuffle=True, seed=self.seed)
            else:
                logging.info("pl_replace_sampler_ddp is False, setting UnrepeatedDistributedSampler for validation and "
                             "test dataloaders. This will ensure that each process gets a unique set of samples. "
                             "If you want to use DistributedSampler, set pl_replace_sampler_ddp to True.")
                return UnrepeatedDistributedSampler(dataset, shuffle=False, seed=self.seed)
        return None

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset,  # type: ignore
                                    shuffle=True,
                                    stage=ModelKey.TRAIN,
                                    **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset,  # type: ignore
                                    shuffle=False,
                                    stage=ModelKey.VAL,
                                    **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dataset,  # type: ignore
                                    shuffle=False,
                                    stage=ModelKey.TEST,
                                    **self.dataloader_kwargs)


class TilesDataModule(HistoDataModule[TilesDataset]):
    """Base class to load the tiles of a dataset as train, val, test sets. Note that tiles are always shuffled by
    default. This means that we sample a random subset of tiles from each bag at each epoch. This is different from
    slides shuffling that is switched on during training time only. This is done to avoid overfitting to the order of
    the tiles in each bag.
    """

    def __init__(
        self,
        cache_mode: CacheMode = CacheMode.NONE,
        precache_location: CacheLocation = CacheLocation.NONE,
        cache_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        """
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

        self.cache_mode = cache_mode
        self.precache_location = precache_location
        self.cache_dir = cache_dir

        super().__init__(**kwargs)

    def prepare_data(self) -> None:
        if self.precache_location != CacheLocation.NONE:
            self._load_dataset(self.train_dataset, stage=ModelKey.TRAIN, shuffle=True)
            self._load_dataset(self.val_dataset, stage=ModelKey.VAL, shuffle=True)
            self._load_dataset(self.test_dataset, stage=ModelKey.TEST, shuffle=True)

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

    def _load_dataset(self, tiles_dataset: TilesDataset, stage: ModelKey, shuffle: bool) -> Dataset:
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

        bag_dataset = BagDataset(
            tiles_dataset,  # type: ignore
            bag_ids=tiles_dataset.slide_ids,
            max_bag_size=self.bag_sizes[stage],
            shuffle_samples=True,
            generator=generator,
        )
        if self.transforms_dict and self.transforms_dict[stage]:
            transform = self.transforms_dict[stage]
        else:
            transform = LoadTilesBatchd(tiles_dataset.IMAGE_COLUMN)

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

    def _get_dataloader(self, dataset: TilesDataset, stage: ModelKey, shuffle: bool,
                        **dataloader_kwargs: Any) -> DataLoader:
        transformed_bag_dataset = self._load_dataset(dataset, stage=stage, shuffle=shuffle)
        bag_dataset: BagDataset = transformed_bag_dataset.data  # type: ignore
        generator = bag_dataset.bag_sampler.generator
        sampler = self._get_ddp_sampler(transformed_bag_dataset, stage)
        return DataLoader(
            transformed_bag_dataset,
            batch_size=self.batch_sizes[stage],
            collate_fn=multibag_collate,
            sampler=sampler,
            # sampler option is mutually exclusive with shuffle
            shuffle=shuffle if sampler is None else None,  # type: ignore
            generator=generator,
            **dataloader_kwargs,
        )


class SlidesDataModule(HistoDataModule[SlidesDataset]):
    """
    Base class to load the slides of a dataset as train, val, test sets. The slide data module performs tiling on the
    fly by default. One can specify the tiling strategies (background removal, overlapping tiles, padding, ...) through
    the class parameters.
    """

    def __init__(
        self,
        loading_params: LoadingParams,
        tiling_params: TilingParams,
        **kwargs: Any,
    ) -> None:
        """
        :param tiling_params: the tiling on the fly parameters.
        :param loading_params: the loading parameters.
        :param kwargs: additional parameters to pass to the parent class HistoDataModule
        """
        super().__init__(**kwargs)
        self.tiling_params = tiling_params
        self.loading_params = loading_params

    def _load_dataset(self, slides_dataset: SlidesDataset, stage: ModelKey) -> Dataset:
        base_transform = Compose(self.get_tiling_transforms(stage=stage))
        if self.transforms_dict and self.transforms_dict[stage]:
            transforms = Compose([base_transform, self.transforms_dict[stage]]).flatten()
        else:
            transforms = base_transform
        # The tiling transform is randomized. Make them deterministic. This call needs to be
        # done on the final Compose, not at the level of the individual randomized transforms.
        transforms.set_random_state(seed=self.seed)
        return Dataset(slides_dataset, transforms)

    def _get_dataloader(self, dataset: SlidesDataset, stage: ModelKey, shuffle: bool,
                        **dataloader_kwargs: Any) -> DataLoader:
        transformed_slides_dataset = self._load_dataset(dataset, stage)
        generator = _create_generator(self.seed)
        sampler = self._get_ddp_sampler(transformed_slides_dataset, stage)
        return DataLoader(
            transformed_slides_dataset,
            batch_size=self.batch_sizes[stage],
            collate_fn=image_collate,
            sampler=sampler,
            # sampler option is mutually exclusive with shuffle
            shuffle=shuffle if not sampler else None,  # type: ignore
            generator=generator,
            **dataloader_kwargs,
        )

    def get_tiling_transforms(self, stage: ModelKey) -> List[Callable]:
        """Returns the list of transforms to apply to the dataset to perform tiling on the fly. The transforms are
        applied in the order they are returned by this method. To add additional transforms, override this method."""
        return [
            self.loading_params.get_load_roid_transform(),
            self.tiling_params.get_tiling_transform(bag_size=self.bag_sizes[stage], stage=stage),
            self.tiling_params.get_extract_coordinates_transform(),
            self.tiling_params.get_split_transform(),
        ]
