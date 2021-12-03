import os
import pytest
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset


from health_ml.utils.bag_utils import (BagSampler, create_bag_dataloader, multibag_collate)

# Run GPU tests only if available
GPUS = [0, -1] if torch.cuda.is_available() else [0]  # type: ignore


def get_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


@pytest.mark.parametrize('shuffle_bags', [False, True])
@pytest.mark.parametrize('shuffle_samples', [False, True])
@pytest.mark.parametrize('max_bag_size', [0, 2])
def test_bag_sampler(shuffle_bags: bool, shuffle_samples: bool, max_bag_size: int) -> None:
    bag_ids = list('aabeefecc')
    n_samples = len(bag_ids)
    n_bags = len(set(bag_ids))
    limited_bag_size = max_bag_size > 0

    sampler = BagSampler(bag_ids,
                         shuffle_bags=shuffle_bags,
                         shuffle_samples=shuffle_samples,
                         max_bag_size=max_bag_size)
    assert len(sampler) == n_bags

    sampled_bags = list(sampler)
    assert len(sampled_bags) == n_bags

    # test sampled indices
    idx_counter = Counter()  # type: ignore
    max_expected_bag_size = max_bag_size if limited_bag_size else n_samples
    for bag in sampled_bags:
        assert 0 < len(bag) <= max_expected_bag_size
        assert all(isinstance(idx, int) for idx in bag)
        idx_counter.update(bag)
    assert all(count == 1 for count in idx_counter.values()), "There were repeated indices"
    if not limited_bag_size:
        assert set(idx_counter) == set(range(n_samples)), \
            "Sampled indices do not match input indices"

    # test sampled bag IDs
    bag_id_counter = Counter()  # type: ignore
    for bag in sampled_bags:
        ids_in_bag = set(bag_ids[idx] for idx in bag)
        assert len(ids_in_bag) == 1, "Bag has mixed IDs"
        bag_id_counter.update(ids_in_bag)
    assert all(count == 1 for count in bag_id_counter.values()), "There were repeated bag IDs"
    assert set(bag_id_counter) == set(bag_ids), "Sampled bag IDs do not match input bag IDs"


@pytest.mark.parametrize('shuffle_bags', [False, True])
@pytest.mark.parametrize('shuffle_samples', [False, True])
@pytest.mark.parametrize('seed1', [None, 0])
@pytest.mark.parametrize('seed2', [None, 0, 1])
def test_bag_sampler_seeding(shuffle_bags: bool, shuffle_samples: bool,
                             seed1: Optional[int], seed2: Optional[int]) -> None:

    bag_ids = list('aabeefecc')

    sampler1 = BagSampler(bag_ids,
                          shuffle_bags=shuffle_bags,
                          shuffle_samples=shuffle_samples,
                          generator=get_generator(seed1))

    sampler2 = BagSampler(bag_ids,
                          shuffle_bags=shuffle_bags,
                          shuffle_samples=shuffle_samples,
                          generator=get_generator(seed2))

    same_seed = (seed1 is not None) and (seed2 is not None) and seed1 == seed2
    same_bag_order = same_seed and not shuffle_bags
    same_sample_order = same_seed and not shuffle_samples

    def get_comparable_outputs(bags: Iterable[Iterable[int]]) -> Iterable[Iterable[int]]:
        # if same order is expected, convert to tuple, otherwise to frozenset
        bags = map(tuple, bags) if same_sample_order else map(frozenset, bags)
        bags = tuple(bags) if same_bag_order else frozenset(bags)
        return bags

    sampled_bags1 = get_comparable_outputs(sampler1)
    sampled_bags2 = get_comparable_outputs(sampler2)

    assert sampled_bags1 == sampled_bags2


def test_bag_sampler_pickling() -> None:
    import pickle

    generator = get_generator(seed=0)
    original_sampler = BagSampler([0, 0, 1, 1, 1], generator=generator)

    # Force a change in the generator state
    torch.rand([1000], generator=original_sampler.generator)

    restored_sampler = pickle.loads(pickle.dumps(original_sampler))

    assert torch.equal(torch.rand([1000], generator=original_sampler.generator),
                       torch.rand([1000], generator=restored_sampler.generator))


class MockMILDataset(Dataset):
    # TODO: Extend tests to also support tuple datasets (e.g. TensorDataset)
    def __init__(self, n_samples: int, n_classes: int, n_bags: int, input_shape: Sequence[int]) -> None:
        self.inputs = torch.rand(n_samples, *input_shape)
        self.labels = torch.randint(n_classes, size=(n_samples,))
        self.instance_ids = torch.arange(n_samples)
        self.bag_ids = torch.randint(n_bags, size=(n_samples,))

    def __len__(self) -> int:
        return len(self.instance_ids)

    def __getitem__(self, index: int) -> Dict:
        return {'input': self.inputs[index],
                'label': self.labels[index],
                'instance_id': self.instance_ids[index],
                'bag_id': self.bag_ids[index]}


def test_multibag_collate() -> None:
    # Use MockMILDataset as a minimal example of dictionary dataset
    dataset = MockMILDataset(n_samples=100,
                             n_classes=10,
                             n_bags=8,
                             input_shape=(1, 4, 4))

    batch_size = 5
    samples_list = [dataset[idx] for idx in range(batch_size)]
    assert all(isinstance(sample, Dict) for sample in samples_list)

    batch = multibag_collate(samples_list)

    assert isinstance(batch, Dict)
    assert batch.keys() == samples_list[0].keys()

    for key, value_list in batch.items():
        assert isinstance(value_list, List)
        assert len(value_list) == batch_size
        assert all(torch.equal(value_list[idx], samples_list[idx][key])
                   for idx in range(batch_size))


@pytest.mark.parametrize('shuffle_bags', [False, True])
@pytest.mark.parametrize('shuffle_samples', [False, True])
@pytest.mark.parametrize('seed', [None, 0])
@pytest.mark.parametrize('batch_size', [None, 1, 2])
def test_bag_dataloader(shuffle_bags: bool, shuffle_samples: bool, batch_size: int,
                        seed: Optional[int]) -> None:
    n_samples = 100
    n_classes = 10
    n_bags = 8
    input_shape = (1, 4, 4)
    dataset = MockMILDataset(n_samples, n_classes, n_bags, input_shape)

    loader = create_bag_dataloader(
        dataset,  # type: ignore
        dataset.bag_ids.tolist(),
        shuffle_bags=shuffle_bags,
        shuffle_samples=shuffle_samples,
        batch_size=batch_size,
        generator=get_generator(seed)
    )

    if batch_size is None:
        for batch in loader:
            assert isinstance(batch, Dict)
            bag_size = batch['input'].shape[0]
            assert batch['input'].shape == (bag_size, *input_shape)
            assert batch['label'].shape == (bag_size,)
            assert batch['instance_id'].shape == (bag_size,)
            assert batch['bag_id'].shape == (bag_size,)
    else:
        for batch_idx, batch in enumerate(loader):
            assert isinstance(batch, Dict)
            actual_batch_size = len(batch['input'])
            if batch_idx != len(loader) - 1:
                assert actual_batch_size == batch_size
            else:  # last batch can be smaller
                assert actual_batch_size <= batch_size

            for key, value in batch.items():
                assert isinstance(value, List)
                assert len(value) == actual_batch_size, f"'{key}' does not match batch size"
            for bag_idx in range(actual_batch_size):
                bag_size = batch['input'][bag_idx].shape[0]
                assert batch['input'][bag_idx].shape == (bag_size, *input_shape)
                assert batch['label'][bag_idx].shape == (bag_size,)
                assert batch['instance_id'][bag_idx].shape == (bag_size,)
                assert batch['bag_id'][bag_idx].shape == (bag_size,)


class InstrumentedLightningModule(LightningModule):
    def __init__(self, base_dataset: Sequence, bag_ids: Sequence[int], batch_size: Optional[int],
                 shuffle_bags: bool, shuffle_samples: bool,
                 get_instance_id: Callable[[Any], Sequence],
                 get_bag_id: Callable[[Any], Sequence],
                 seed: Optional[int] = None, **dataloader_kwargs: Any) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.bag_ids = bag_ids
        self.batch_size = batch_size
        self.shuffle_bags = shuffle_bags
        self.shuffle_samples = shuffle_samples
        self.get_instance_id = get_instance_id
        self.get_bag_id = get_bag_id
        self.dataloader_kwargs = dataloader_kwargs

        self.expected_bag_sizes = Counter(self.bag_ids)
        self.generator = get_generator(seed)

    def configure_optimizers(self) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return create_bag_dataloader(self.base_dataset, self.bag_ids,
                                     batch_size=self.batch_size,
                                     shuffle_bags=self.shuffle_bags,
                                     shuffle_samples=self.shuffle_samples,
                                     generator=self.generator,
                                     **self.dataloader_kwargs)

    def on_train_epoch_start(self) -> None:  # type: ignore
        # mypy can't infer the parametrised type of empty collections
        self.instance_id_counter = Counter()  # type: ignore
        self.bag_id_counter = Counter()  # type: ignore
        self.unique_bag_id_counter = Counter()  # type: ignore
        self.n_bags_seen = 0

    def training_step(self, batch: Any, batch_idx: int) -> None:  # type: ignore
        def to_list(x: Union[torch.Tensor, Iterable]) -> List:
            if isinstance(x, torch.Tensor):
                return x.tolist()
            return list(x)

        batch_instance_ids = self.get_instance_id(batch)
        batch_bag_ids = self.get_bag_id(batch)
        if self.batch_size is not None:
            n_bags_in_batch = len(batch_instance_ids)
            batch_instance_ids = sum(map(to_list, batch_instance_ids), [])  # flatten list of lists
            batch_bag_ids = sum(map(to_list, batch_bag_ids), [])  # flatten list of lists
        else:
            n_bags_in_batch = 1
            batch_instance_ids = to_list(batch_instance_ids)
            batch_bag_ids = to_list(batch_bag_ids)

        self.instance_id_counter.update(batch_instance_ids)

        batch_bag_sizes = Counter(batch_bag_ids)
        for bag_id in batch_bag_sizes:
            assert batch_bag_sizes[bag_id] == self.expected_bag_sizes[bag_id]
        self.bag_id_counter.update(batch_bag_sizes)

        unique_batch_bag_ids = set(batch_bag_ids)
        n_unique_batch_bag_ids = len(unique_batch_bag_ids)
        if self.batch_size is None or self.batch_size == 1:
            assert n_unique_batch_bag_ids == 1, "Batch has mixed bag IDs"
        else:
            assert n_unique_batch_bag_ids == n_bags_in_batch
            assert n_unique_batch_bag_ids <= self.batch_size, "Too many bags in a batch"
        self.unique_bag_id_counter.update(unique_batch_bag_ids)
        self.n_bags_seen += n_bags_in_batch

    def on_train_epoch_end(self) -> None:  # type: ignore
        assert all(count == 1 for count in self.instance_id_counter.values()), \
            "There were repeated instances"
        assert len(set(self.instance_id_counter)) == len(self.base_dataset)

        assert set(self.bag_id_counter) == set(self.unique_bag_id_counter), \
            "Inconsistent test state - this should not happen"
        assert self.bag_id_counter == self.expected_bag_sizes

        assert all(count == 1 for count in self.unique_bag_id_counter.values()), \
            "There were repeated bag IDs"
        unique_bag_ids = set(self.bag_ids)
        assert set(self.unique_bag_id_counter) == unique_bag_ids, \
            "Sampled bag IDs do not match input bag IDs"
        assert self.n_bags_seen == len(unique_bag_ids)


@pytest.mark.parametrize('shuffle_bags', [False, True])
@pytest.mark.parametrize('shuffle_samples', [False, True])
@pytest.mark.parametrize('gpus', GPUS)
def test_lightning_mock_dataset(shuffle_bags: bool, shuffle_samples: bool, gpus: int) -> None:
    n_samples = 100
    n_classes = 10
    n_bags = 5
    input_shape = (1, 4, 4)
    base_dataset = MockMILDataset(n_samples, n_classes, n_bags, input_shape)
    module = InstrumentedLightningModule(
        base_dataset,  # type: ignore
        bag_ids=base_dataset.bag_ids.tolist(),
        batch_size=2,
        shuffle_bags=shuffle_bags,
        shuffle_samples=shuffle_samples,
        get_instance_id=lambda batch: batch['instance_id'],
        get_bag_id=lambda batch: batch['bag_id'],
        seed=0,
        num_workers=os.cpu_count())
    trainer = Trainer(max_epochs=2, gpus=gpus)
    trainer.fit(module)
