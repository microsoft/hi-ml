#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch
import pytest
import numpy as np

from unittest.mock import patch
from typing import Dict, List, Any, Set
from testhisto.utils.utils_testhisto import run_distributed
from health_cpath.utils.naming import ResultsKey, SlideKey
from health_cpath.utils.plots_utils import TilesSelector, SlideNode


def _create_mock_data(n_samples: int, n_tiles: int = 3, device: str = "cpu") -> Dict:
    """Generates a mock pretiled slides data dictionary.

    :param n_samples: The number of whole slide images to generate.
    :param n_tiles: The minimum number of tiles in each slide, defaults to 3
    :param device: torch device where tensors should be created, defaults to "cpu"
    :return: A dictioanry containing randomly generated mock data.
    """
    n_tiles = 3
    tile_size = (1, 4, 4)
    diff_n_tiles = [n_tiles + i for i in range(n_samples)]
    mock_data = {
        SlideKey.SLIDE_ID: np.array([f"slide_{i}" for i in range(n_samples)]),
        SlideKey.IMAGE: [torch.randint(0, 255, (diff_n_tiles[i], *tile_size), device=device) for i in range(n_samples)],
    }
    return mock_data


def _create_mock_results(n_samples: int, n_tiles: int = 3, n_classes: int = 2, device: str = "cpu") -> Dict:
    """Generates mock results data dictionary.

    :param n_samples: The number of whole slide images.
    :param n_tiles: The minimum number of tiles in each slide, defaults to 3
    :param n_classes: the number of class labels in the dataset, defaults to 2
    :param device: torch device where tensors should be created, defaults to "cpu"
    :return: A dictioanry containing randomly generated mock results.
    """
    diff_n_tiles = [n_tiles + i for i in range(n_samples)]
    probs = torch.rand((n_samples, n_classes), device=device)
    probs = probs / probs.sum(dim=1, keepdim=True)
    mock_results = {
        ResultsKey.SLIDE_ID: np.array([f"slide_{i}" for i in range(n_samples)]),
        ResultsKey.TRUE_LABEL: torch.randint(n_classes, size=(n_samples,), device=device),
        ResultsKey.PRED_LABEL: torch.argmax(probs, dim=1),
        ResultsKey.BAG_ATTN: [torch.rand(size=(1, diff_n_tiles[i]), device=device) for i in range(n_samples)],
        ResultsKey.CLASS_PROBS: probs,
    }
    return mock_results


def _batch_data(data: Dict, batch_idx: int, batch_size: int) -> Dict:
    """Helper function to generate smaller batches from a dictionary."""
    batch = {}
    for k in data:
        batch[k] = data[k][batch_idx * batch_size: (batch_idx + 1) * batch_size]
    return batch


def _create_and_update_top_bottom_tiles_selector(
    data: Dict,
    results: Dict,
    num_top_slides: int,
    num_top_tiles: int,
    n_classes: int,
    rank: int = 0,
    batch_size: int = 2,
    n_batches: int = 10,
) -> TilesSelector:
    """Create a top and bottom tiles selector and update its top and bottom slides/tiles while looping through the data
    available for the current rank

    :param data: The data dictionary containing the entire small dataset.
    :param results: The results dictionary containing mock resulst for all data.
    :param num_top_slides: The number of slides to use to select top and bottom slides.
    :param num_top_tiles: The number of tiles to select as top and bottom tiles for each top/bottom slide.
    :param n_classes: The number of class labels.
    :param rank: The identifier of the current process within the ddp context, defaults to 0
    :param batch_size: The number of samples in a batch, defaults to 2
    :param n_batches: The number of batches, defaults to 10
    :return: A top bottom tiles selector with selected top and bottom slides and corresponding top and bottom slides.
    """

    tiles_selector = TilesSelector(n_classes, num_slides=num_top_slides, num_tiles=num_top_tiles)

    for i in range(rank * n_batches, (rank + 1) * n_batches):
        batch_data = _batch_data(data, batch_idx=i, batch_size=batch_size)
        batch_results = _batch_data(results, batch_idx=i, batch_size=batch_size)
        tiles_selector.update_slides_selection(batch_data, batch_results)

    return tiles_selector


def _get_expected_slides_by_probability(
    results: Dict[ResultsKey, Any], num_top_slides: int = 2, label: int = 1, top: bool = True
) -> Set[str]:
    """Select top or bottom slides according to their probability scores from the entire dataset.

    :param results: The results dictionary for the entire dataset.
    :param num_top_slides: The number of slides to use to select top and bottom slides, defaults to 5
    :param label: The current label to process given that top and bottom are grouped by class label, defaults to 1
    :param top: A flag to select top or bottom slides with highest (respetively, lowest) prob scores, defaults to True
    :return: A set of selected slide ids.
    """

    class_indices = (results[ResultsKey.TRUE_LABEL].squeeze() == label).nonzero().squeeze(1)
    class_prob = results[ResultsKey.CLASS_PROBS][class_indices, label]
    assert class_prob.shape == (len(class_indices),)
    num_top_slides = min(num_top_slides, len(class_prob))

    _, sorting_indices = class_prob.topk(num_top_slides, largest=top)
    sorted_class_indices = class_indices[sorting_indices]

    def _selection_condition(index: int) -> bool:
        if top:
            return results[ResultsKey.PRED_LABEL][index] == results[ResultsKey.TRUE_LABEL][index]
        else:
            return results[ResultsKey.PRED_LABEL][index] != results[ResultsKey.TRUE_LABEL][index]

    return {results[ResultsKey.SLIDE_ID][i] for i in sorted_class_indices if _selection_condition(i)}


@pytest.mark.parametrize("num_top_slides", [2, 10])
@pytest.mark.parametrize("n_classes", [2, 3])  # n_classes=2 represents the binary case.
def test_aggregate_shallow_slide_nodes(
    n_classes: int, num_top_slides: int, uneven_samples: bool = False, rank: int = 0, world_size: int = 1,
    device: str = "cpu"
) -> None:
    """This test ensures that shallow copies of slide nodes are gathered properlyy across devices in a ddp context."""
    n_tiles = 3
    batch_size = 2
    n_batches = 10
    total_batches = n_batches * world_size
    num_top_tiles = 2
    n_samples = batch_size * total_batches - int(uneven_samples)
    torch.manual_seed(42)
    data = _create_mock_data(n_samples=n_samples, n_tiles=n_tiles, device=device)
    results = _create_mock_results(n_samples=n_samples, n_tiles=n_tiles, n_classes=n_classes, device=device)

    tiles_selector = _create_and_update_top_bottom_tiles_selector(
        data, results, num_top_slides, num_top_tiles, n_classes, rank, batch_size, n_batches
    )

    shallow_top_slides_heaps = tiles_selector._shallow_copy_slides_heaps(tiles_selector.top_slides_heaps)
    shallow_bottom_slides_heaps = tiles_selector._shallow_copy_slides_heaps(tiles_selector.bottom_slides_heaps)

    if torch.distributed.is_initialized():
        if world_size > 1:
            shallow_top_slides_heaps = tiles_selector._aggregate_shallow_slides_heaps(
                world_size, shallow_top_slides_heaps
            )
            shallow_bottom_slides_heaps = tiles_selector._aggregate_shallow_slides_heaps(
                world_size, shallow_bottom_slides_heaps
            )

    if rank == 0:
        for label in range(n_classes):

            assert all(slide_node.pred_label == slide_node.true_label for slide_node in shallow_top_slides_heaps[label])
            selected_top_slides_ids = {slide_node.slide_id for slide_node in shallow_top_slides_heaps[label]}
            expected_top_slides_ids = _get_expected_slides_by_probability(results, num_top_slides, label, top=True)
            assert expected_top_slides_ids == selected_top_slides_ids

            assert all(
                slide_node.pred_label != slide_node.true_label for slide_node in shallow_bottom_slides_heaps[label]
            )
            selected_bottom_slides_ids = {slide_node.slide_id for slide_node in shallow_bottom_slides_heaps[label]}
            expected_bottom_slides_ids = _get_expected_slides_by_probability(results, num_top_slides, label, top=False)
            assert expected_bottom_slides_ids == selected_bottom_slides_ids

            # Make sure that the top and bottom slides are disjoint.
            assert not selected_top_slides_ids.intersection(selected_bottom_slides_ids)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_aggregate_shallow_slide_nodes_distributed() -> None:
    """These tests need to be called sequentially to prevent them to be run in parallel."""
    # test with n_classes = 2, n_slides = 2
    run_distributed(test_aggregate_shallow_slide_nodes, [2, 2, False], world_size=1)
    run_distributed(test_aggregate_shallow_slide_nodes, [2, 2, False], world_size=2)
    run_distributed(test_aggregate_shallow_slide_nodes, [2, 2, True], world_size=2)
    # test with n_classes = 3, n_slides = 2
    run_distributed(test_aggregate_shallow_slide_nodes, [3, 2], world_size=1)
    run_distributed(test_aggregate_shallow_slide_nodes, [3, 2], world_size=2)


def assert_equal_top_bottom_attention_tiles(
    slide_ids: Set[str], data: Dict, results: Dict, num_top_tiles: int, slide_nodes: List[SlideNode]
) -> None:
    """Asserts that top and bottom tiles selected on the fly by the top bottom tiles selector are equal to the expected
    top and bottom tiles in the mock dataset.

    :param slide_ids: A set of expected slide ids0
    :param data: A dictionary containing the entire dataset.
    :param results: A dictionary of data results.
    :param num_top_tiles: The number of tiles to select as top and bottom tiles for each top/bottom slide.
    :param slide_nodes: The top or bottom slide nodes selected on the fly by the selector.
    """

    slide_nodes_dict = {slide_node.slide_id: slide_node for slide_node in slide_nodes}

    for slide_id in slide_ids:
        slide_batch_idx = int(slide_id.split("_")[1])
        tiles = data[SlideKey.IMAGE][slide_batch_idx]

        expected_top_attns, top_tiles_ids = torch.topk(
            results[ResultsKey.BAG_ATTN][slide_batch_idx].squeeze(), k=num_top_tiles, largest=True, sorted=True
        )
        expected_bottom_attns, bottom_tiles_ids = torch.topk(
            results[ResultsKey.BAG_ATTN][slide_batch_idx].squeeze(), k=num_top_tiles, largest=False, sorted=True
        )

        expected_top_tiles: List[torch.Tensor] = [tiles[tile_id] for tile_id in top_tiles_ids]
        expected_bottom_tiles: List[torch.Tensor] = [tiles[tile_id] for tile_id in bottom_tiles_ids]

        top_tiles = slide_nodes_dict[slide_id].top_tiles
        bottom_tiles = slide_nodes_dict[slide_id].bottom_tiles

        for j, expected_top_tile in enumerate(expected_top_tiles):
            assert torch.equal(expected_top_tile.cpu(), top_tiles[j].data)
            assert expected_top_attns[j].item() == top_tiles[j].attn

        for j, expected_bottom_tile in enumerate(expected_bottom_tiles):
            assert torch.equal(expected_bottom_tile.cpu(), bottom_tiles[j].data)
            assert expected_bottom_attns[j].item() == bottom_tiles[j].attn


@pytest.mark.parametrize("num_top_slides", [2, 10])
@pytest.mark.parametrize("n_classes", [2, 3])  # n_classes=2 represents the binary case.
def test_select_k_top_bottom_tiles_on_the_fly(
    n_classes: int, num_top_slides: int, uneven_samples: bool = False, rank: int = 0, world_size: int = 1,
    device: str = "cpu"
) -> None:
    """This tests checks that k top and bottom tiles are selected properly `on the fly`:
        1- Create a mock dataset and corresponding mock results that are small enough to fit in memory
        2- Create a tiles selector that is only exposed to a subset of the data distributed across devices. This
           selector updates its top and bottom slides and tiles sequentially as we processes smaller batches of data.
        3- Gather top and bottom tiles if ddp context
        4- Select expected top slides from the entire dataset using torch.topk given that it's a small set that fits
           entirely in memory.
        5- Assert that the top slides selected on the fly are equal to the expected top slides selected from the entire
           dataset for both ddp and single device runs.
        6- Assert that corresponding top and bottom tiles are equal as well
        7- Repeat steps 4, 5 and 6 for bottom slides.
    """

    n_tiles = 3
    batch_size = 2
    n_batches = 10
    total_batches = n_batches * world_size
    num_top_tiles = 2
    num_top_slides = 2
    n_samples = batch_size * total_batches - int(uneven_samples)
    torch.manual_seed(42)
    data = _create_mock_data(n_samples=n_samples, n_tiles=n_tiles, device=device)
    results = _create_mock_results(n_samples=n_samples, n_tiles=n_tiles, n_classes=n_classes, device=device)

    tiles_selector = _create_and_update_top_bottom_tiles_selector(
        data, results, num_top_slides, num_top_tiles, n_classes, rank, batch_size, n_batches
    )
    tiles_selector.gather_selected_tiles_across_devices()

    if rank == 0:
        for label in range(n_classes):

            assert all(
                slide_node.pred_label == slide_node.true_label for slide_node in tiles_selector.top_slides_heaps[label]
            )
            selected_top_slides_ids = {slide_node.slide_id for slide_node in tiles_selector.top_slides_heaps[label]}
            expected_top_slides_ids = _get_expected_slides_by_probability(results, num_top_slides, label, top=True)
            assert expected_top_slides_ids == selected_top_slides_ids
            assert_equal_top_bottom_attention_tiles(
                expected_top_slides_ids, data, results, num_top_tiles, tiles_selector.top_slides_heaps[label]
            )

            assert all(
                slide_node.pred_label != slide_node.true_label
                for slide_node in tiles_selector.bottom_slides_heaps[label]
            )
            selected_bottom_slides_ids = {
                slide_node.slide_id for slide_node in tiles_selector.bottom_slides_heaps[label]
            }
            expected_bottom_slides_ids = _get_expected_slides_by_probability(results, num_top_slides, label, top=False)
            assert expected_bottom_slides_ids == selected_bottom_slides_ids

            assert_equal_top_bottom_attention_tiles(
                expected_bottom_slides_ids, data, results, num_top_tiles, tiles_selector.bottom_slides_heaps[label]
            )

            # Make sure that the top and bottom slides are disjoint.
            assert not selected_top_slides_ids.intersection(selected_bottom_slides_ids)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_select_k_top_bottom_tiles_on_the_fly_distributed() -> None:
    """These tests need to be called sequentially to prevent them to be run in parallel"""
    # test with n_classes = 2, n_slides = 2
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [2, 2, False], world_size=1)
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [2, 2, False], world_size=2)
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [2, 2, True], world_size=2)
    # test with n_classes = 3, n_slides = 2
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [3, 2], world_size=1)
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [3, 2], world_size=2)


def test_disable_top_bottom_tiles_selector() -> None:
    with pytest.raises(ValueError) as ex:
        _ = TilesSelector(n_classes=2, num_slides=2, num_tiles=0)
    assert "You should use `num_top_tiles>0` to be able to select top and bottom tiles" in str(ex)


def test_tiles_are_selected_only_with_non_zero_num_top_slides(
    uneven_samples: bool = False, rank: int = 0, world_size: int = 1, device: str = "cpu"
) -> None:
    n_tiles = 3
    batch_size = 1
    n_batches = 2
    total_batches = n_batches * world_size
    num_top_tiles = 2
    num_top_slides = 0
    n_classes = 2
    n_samples = batch_size * total_batches - int(uneven_samples)
    torch.manual_seed(42)
    data = _create_mock_data(n_samples=n_samples, n_tiles=n_tiles, device=device)
    results = _create_mock_results(n_samples=n_samples, n_tiles=n_tiles, n_classes=n_classes, device=device)
    tiles_selector = TilesSelector(n_classes, num_slides=num_top_slides, num_tiles=num_top_tiles)

    with patch.object(tiles_selector, "_update_label_slides") as mock_update_label_slides:
        for i in range(rank * n_batches, (rank + 1) * n_batches):
            batch_data = _batch_data(data, batch_idx=i, batch_size=batch_size)
            batch_results = _batch_data(results, batch_idx=i, batch_size=batch_size)
            tiles_selector.update_slides_selection(batch_data, batch_results)
    mock_update_label_slides.assert_not_called()

    for class_id in range(n_classes):
        assert len(tiles_selector.top_slides_heaps[class_id]) == 0
        assert len(tiles_selector.bottom_slides_heaps[class_id]) == 0

    with patch.object(tiles_selector, "_shallow_copy_slides_heaps") as mock_shallow_copy_slides_heaps:
        tiles_selector.gather_selected_tiles_across_devices()
    mock_shallow_copy_slides_heaps.assert_not_called()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_tiles_are_selected_only_with_non_zero_num_top_slides_distributed() -> None:
    """These tests need to be called sequentially to prevent them to be run in parallel"""
    run_distributed(test_tiles_are_selected_only_with_non_zero_num_top_slides, [False], world_size=1)
    run_distributed(test_tiles_are_selected_only_with_non_zero_num_top_slides, [False], world_size=2)
    run_distributed(test_tiles_are_selected_only_with_non_zero_num_top_slides, [True], world_size=2)
