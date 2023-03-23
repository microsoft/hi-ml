from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import default_collate
from math import ceil

from health_ml.utils.common_utils import _create_generator


class BagSampler(Sampler[List[int]]):
    """A batch sampler that iterates samples in specified groups (bags).

    This is useful in multiple-instance learning contexts, where samples in a 'bag' must be
    processed simultaneously. The sampler should be passed to a `DataLoader` as the `batch_sampler`
    argument, for example:

    >>> sampler = BagSampler([1, 1, 0, 2, 0, 1])
    >>> list(sampler)
    [[2, 4], [0, 1, 5], [3]]
    >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        bag_ids: Sequence,
        shuffle_bags: bool = False,
        shuffle_samples: bool = False,
        max_bag_size: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        :param bag_ids: The bag IDs for each sample, of the same length as the dataset.
        :param shuffle_bags: Whether the bags should be iterated in random order.
        :param shuffle_samples: Whether the instances in each bag should be shuffled.
        :param max_bag_size: Upper bound on number of instances in each loaded bag. If 0 (default),
            will return all samples in each bag. If > 0 and `shuffle_samples=True`, bags larger than
            `max_bag_size` will yield random subsets of instances. Note that setting `max_bag_size > 0`
            with `shuffle_samples=False` will return fixed subsets and may completely exclude some
            samples from the iteration.
        :param generator: The pseudorandom number generator to use for shuffling. By default, creates one with a random
            seed.
        """
        self.unique_bag_ids, self.bag_indices = np.unique(bag_ids, return_inverse=True)
        self.shuffle_bags = shuffle_bags
        self.shuffle_samples = shuffle_samples
        self.max_bag_size = max_bag_size
        self.generator = generator

    def __iter__(self) -> Iterator[List[int]]:
        generator = self.generator or _create_generator()
        n_bags = len(self.unique_bag_ids)
        bag_sequence = torch.randperm(n_bags, generator=generator).tolist() if self.shuffle_bags else range(n_bags)
        for bag_idx in bag_sequence:
            yield self.get_bag(bag_idx, generator)

    def get_bag(self, bag_index: int, generator: Optional[torch.Generator] = None) -> List[int]:
        (bag,) = np.where(self.bag_indices == bag_index)
        if self.shuffle_samples:
            if generator is None:
                generator = self.generator or _create_generator()
            perm = torch.randperm(len(bag), generator=generator)
            bag = np.atleast_1d(bag[perm])  # pytorch squeezes singleton tensors
        if self.max_bag_size > 0:
            bag = bag[: self.max_bag_size]
        return bag.tolist()

    def __len__(self) -> int:
        return len(self.unique_bag_ids)

    def __getstate__(self) -> Dict:
        # torch.Generator is not pickleable, so we need to serialise only its internal state.
        # This is useful e.g. for caching and restoring a BagDataset.
        d = self.__dict__.copy()
        if self.generator is not None:
            generator_state = self.generator.get_state() if self.generator is not None else None
            d['generator'] = generator_state
        return d

    def __setstate__(self, d: Dict) -> None:
        # Same here for restoring the torch.Generator state
        self.__dict__ = d
        generator = None
        if d['generator'] is not None:
            generator = torch.Generator()
            generator.set_state(d['generator'])
        self.generator = generator


class BagDataset(Dataset):
    """A wrapper dataset that iterates a base dataset in user-specified 'bags'."""

    def __init__(
        self,
        base_dataset: Sequence,
        bag_ids: Sequence[int],
        shuffle_samples: bool = False,
        max_bag_size: int = 0,
        generator: Optional[torch.Generator] = None,
        collate_fn: Callable[[List], Any] = default_collate,
    ) -> None:
        """
        :param base_dataset: The source dataset whose samples will be grouped in bags.
        :param bag_ids: The bag IDs for each sample, of the same length as the dataset.
        :param shuffle_samples: Whether the instances in each bag should be shuffled.
        :param max_bag_size: Upper bound on number of instances in each loaded bag. If 0 (default),
        will return all samples in each bag. If > 0 and `shuffle_samples=True`, bags larger than
        `max_bag_size` will yield random subsets of instances. Note that setting `max_bag_size > 0`
        with `shuffle_samples=False` will return fixed subsets and may completely exclude some
        samples from the iteration.
        :param generator: The pseudorandom number generator to use for shuffling. By default, creates
        one with a random seed.
        :param collate_fn: Function to aggregate individual samples into a batch. Uses the PyTorch
        default if unspecified, which stacks tensors along their first dimension.
        More details in https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
        """
        if len(base_dataset) != len(bag_ids):
            raise ValueError(
                f"Base dataset and bag IDs must have the same length, " f"got {len(base_dataset)} and {len(bag_ids)}"
            )

        self.base_dataset = base_dataset
        self.bag_sampler = BagSampler(
            bag_ids,
            shuffle_bags=False,  # bag shuffling is handled by dataloader
            shuffle_samples=shuffle_samples,
            max_bag_size=max_bag_size,
            generator=generator,
        )
        self.collate_fn = collate_fn
        self.bag_ids = bag_ids

    def __len__(self) -> int:
        return len(self.bag_sampler)

    def __getitem__(self, index: int) -> Any:
        bag_indices = self.bag_sampler.get_bag(index)
        bag_samples = [self.base_dataset[i] for i in bag_indices]
        return self.collate_fn(bag_samples)


class BatchedDataset(Dataset):
    """A wrapper class that aggregates multiple bags in a batch"""

    # TODO: Just a stub for now; extend to enable shuffling and/or other batching strategies

    def __init__(self, base_dataset: Sequence, batch_size: int) -> None:
        self.base_dataset = base_dataset
        self.batch_size = batch_size

    def __len__(self) -> int:
        return ceil(len(self.base_dataset) / self.batch_size)

    def __getitem__(self, index: int) -> Any:
        start_index = index * self.batch_size
        stop_index = start_index + self.batch_size
        # list of len batch, each element is a dict, dict values are lists or tensors,
        # len of lists/tensors is variable -> we can't use default collate
        multi_bag_list = [self.base_dataset[i] for i in range(start_index, stop_index)]
        return multi_bag_list


def multibag_collate(batch: List) -> Any:
    """Turn list of batches into batch of lists"""
    # TODO: Enable padding tensors to the same fixed dimensions, e.g. using a callable class
    elem = batch[0]

    if isinstance(elem, Mapping):
        elem_keys = elem.keys()
        if not all(other.keys() == elem_keys for other in batch[1:]):
            raise RuntimeError("Every element in the batch should have the same keys")
        return {key: [d[key] for d in batch] for key in elem}

    elif isinstance(elem, Sequence):
        elem_len = len(elem)
        if not all(len(other) == elem_len for other in batch[1:]):
            raise RuntimeError("Every element in the batch should be of equal size")
        return tuple(zip(*batch))

    return batch  # return other types as a plain list


def create_bag_dataloader(
    base_dataset: Sequence,
    bag_ids: Sequence,
    *,  # make following arguments keyword-only to avoid confusion
    shuffle_bags: bool,
    shuffle_samples: bool,
    max_bag_size: int = 0,
    batch_size: Optional[int] = 1,
    collate_fn: Callable[[List], Any] = default_collate,
    generator: Optional[torch.Generator] = None,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """Create a DataLoader that consumes the given dataset in batches grouped by bag ID.

    :param base_dataset: The source dataset whose samples will be grouped in bags.
    :param bag_ids: The bag IDs for each sample, of the same length as the dataset.
    :param shuffle_bags: Whether the bags should be iterated in random order.
    :param shuffle_samples: Whether the instances in each bag should be shuffled.
    :param max_bag_size: Upper bound on number of instances in each loaded bag. If 0 (default),
    will return all samples in each bag. If > 0 and `shuffle_samples=True`, bags larger than
    `max_bag_size` will yield random subsets of instances. Note that setting `max_bag_size > 0`
    with `shuffle_samples=False` will return fixed subsets and may completely exclude some
    samples from the iteration.
    :param batch_size: Number of bags per batch to load. If a number, each field in a batch
    (e.g. dictionary entries or tuple elements) will contain a list of length `batch_size`
    with the corresponding values for every bag. If `None`, will iterate individual bags
    without intermediate lists.
    :param collate_fn: Function to aggregate individual samples into a bag. Uses the PyTorch
    default if unspecified, which stacks tensors along their first dimension.
    More details in https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    :param generator: The pseudorandom number generator to use for shuffling. By default,
    creates one with a random seed.
    :param dataloader_kwargs: Further keyword arguments to be passed to the `DataLoader`, e.g.
    `num_workers`, `pin_memory`, etc.
    :return: The `DataLoader` configured to iterate one bag at a time.
    """
    generator = generator or _create_generator()
    bag_dataset = BagDataset(
        base_dataset,
        bag_ids=bag_ids,
        max_bag_size=max_bag_size,
        shuffle_samples=shuffle_samples,
        generator=generator,
        collate_fn=collate_fn,
    )
    return DataLoader(
        bag_dataset,
        shuffle=shuffle_bags,
        generator=generator,
        batch_size=batch_size,
        collate_fn=None if batch_size is None else multibag_collate,
        **dataloader_kwargs,
    )
