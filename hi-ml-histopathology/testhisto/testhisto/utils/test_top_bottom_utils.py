#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

from pathlib import Path
from typing import Dict, List, Any

from health_ml.utils.common_utils import is_windows
from histopathology.utils.viz_utils import save_figure
from testhisto.utils.utils_testhisto import run_distributed
from histopathology.utils.naming import ResultsKey, SlideKey
from health_ml.utils.fixed_paths import OutputFolderForTests
from testhisto.utils.utils_testhisto import assert_binary_files_match, full_ml_test_data_path
from histopathology.utils.top_bottom_tiles_utils import TileNode, TopBottomTilesHandler, SlideNode


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
    mock_results = {
        ResultsKey.SLIDE_ID: np.array([f"slide_{i}" for i in range(n_samples)]),
        ResultsKey.TRUE_LABEL: torch.randint(2, size=(n_samples,), device=device),
        ResultsKey.BAG_ATTN: [torch.rand(size=(1, diff_n_tiles[i]), device=device) for i in range(n_samples)],
        ResultsKey.CLASS_PROBS: torch.rand((n_samples, n_classes), device=device),
    }
    return mock_results


def _batch_data(data: Dict, batch_idx: int, batch_size: int) -> Dict:
    """Helper function to generate smaller batches from a dictionary."""
    batch = {}
    for k in data:
        batch[k] = data[k][batch_idx * batch_size: (batch_idx + 1) * batch_size]
    return batch


def _create_and_update_top_bottom_tiles_handler(
    data: Dict,
    results: Dict,
    num_top_slides: int,
    num_top_tiles: int,
    n_classes: int,
    rank: int = 0,
    batch_size: int = 2,
    n_batches: int = 10,
) -> TopBottomTilesHandler:
    """Create a top and bottom tiles handler and update its top and bottom slides/tiles while looping through the data
    available for the current rank

    :param data: The data dictionary containing the entire small dataset.
    :param results: The results dictionary containing mock resulst for all data.
    :param num_top_slides: The number of slides to use to select top and bottom slides.
    :param num_top_tiles: The number of tiles to select as top and bottom tiles for each top/bottom slide.
    :param n_classes: The number of class labels.
    :param rank: The identifier of the current process within the ddp context, defaults to 0
    :param batch_size: The number of samples in a batch, defaults to 2
    :param n_batches: The number of batches, defaults to 10
    :return: A top bottom tiles handler with selected top and bottom slides and corresponding top and bottom slides.
    """

    handler = TopBottomTilesHandler(n_classes, num_top_slides=num_top_slides, num_top_tiles=num_top_tiles)

    for i in range(rank * n_batches, (rank + 1) * n_batches):
        batch_data = _batch_data(data, batch_idx=i, batch_size=batch_size)
        batch_results = _batch_data(results, batch_idx=i, batch_size=batch_size)
        handler.update_slides_selection(batch_data, batch_results)

    return handler


def _get_expected_slides_by_probability(
    results: Dict[ResultsKey, Any], num_top_slides: int = 2, label: int = 1, top: bool = True
) -> List[str]:
    """Select top or bottom slides according to their probability scores from the entire dataset.

    :param results: The results dictionary for the entire dataset.
    :param num_top_slides: The number of slides to use to select top and bottom slides, defaults to 5
    :param label: The current label to process given that top and bottom are grouped by class label, defaults to 1
    :param top: A flag to select top or bottom slides with highest (respetively, lowest) prob scores, defaults to True
    :return: A list of selected slide ids.
    """

    class_indices = (results[ResultsKey.TRUE_LABEL].squeeze() == label).nonzero().squeeze(1)
    class_prob = results[ResultsKey.CLASS_PROBS][class_indices, label]
    assert class_prob.shape == (len(class_indices),)
    num_top_slides = min(num_top_slides, len(class_prob))

    _, sorting_indices = class_prob.topk(num_top_slides, largest=top, sorted=True)
    sorted_class_indices = class_indices[sorting_indices]

    return [results[ResultsKey.SLIDE_ID][i] for i in sorted_class_indices][::-1]  # the order is inversed in the heaps


def get_expected_top_slides_by_probability(
    results: Dict[ResultsKey, Any], num_top_slides: int = 5, label: int = 1
) -> List[str]:
    """Calls `_get_expected_slides_by_probability` with `top=True` to select expected top slides for the entire dataset
    in one go. """
    return _get_expected_slides_by_probability(results, num_top_slides, label, top=True)


def get_expected_bottom_slides_by_probability(
    results: Dict[ResultsKey, Any], num_top_slides: int = 5, label: int = 1
) -> List[str]:
    """Calls `_get_expected_slides_by_probability` with `top=False` to select expected bottom slides for the entire
    dataset in one go. """
    return _get_expected_slides_by_probability(results, num_top_slides, label, top=False)


@pytest.mark.parametrize("n_classes", [2, 3])  # n_classes=2 represents the binary case.
def test_aggregate_shallow_slide_nodes(n_classes: int, rank: int = 0, world_size: int = 1, device: str = "cpu") -> None:
    """This test ensures that shallow copies of slide nodes are gathered properlyy across devices in a ddp context."""
    n_tiles = 3
    batch_size = 2
    n_batches = 10
    total_batches = n_batches * world_size
    num_top_tiles = 2
    num_top_slides = 2

    torch.manual_seed(42)
    data = _create_mock_data(n_samples=batch_size * total_batches, n_tiles=n_tiles, device=device)
    results = _create_mock_results(
        n_samples=batch_size * total_batches, n_tiles=n_tiles, n_classes=n_classes, device=device
    )

    handler = _create_and_update_top_bottom_tiles_handler(
        data, results, num_top_slides, num_top_tiles, n_classes, rank, batch_size, n_batches
    )

    shallow_top_slides_heaps = handler._shallow_copy_slides_heaps(handler.top_slides_heaps)
    shallow_bottom_slides_heaps = handler._shallow_copy_slides_heaps(handler.bottom_slides_heaps)

    if torch.distributed.is_initialized():
        if world_size > 1:
            shallow_top_slides_heaps = handler._aggregate_shallow_slides_heaps(world_size, shallow_top_slides_heaps)
            shallow_bottom_slides_heaps = handler._aggregate_shallow_slides_heaps(
                world_size, shallow_bottom_slides_heaps
            )

    if rank == 0:
        for label in range(n_classes):
            expected_top_slides_ids = _get_expected_slides_by_probability(results, num_top_slides, label, top=True)
            assert expected_top_slides_ids == [slide_node.slide_id for slide_node in shallow_top_slides_heaps[label]]

            expected_bottom_slides_ids = _get_expected_slides_by_probability(results, num_top_slides, label, top=False)
            assert expected_bottom_slides_ids == [
                slide_node.slide_id for slide_node in shallow_bottom_slides_heaps[label]
            ]


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_aggregate_shallow_slide_nodes_distributed() -> None:
    """These tests need to be called sequentially to prevent them to be run in parallel."""
    # test with n_classes = 2
    run_distributed(test_aggregate_shallow_slide_nodes, [2], world_size=1)
    run_distributed(test_aggregate_shallow_slide_nodes, [2], world_size=2)
    # test with n_classes = 3
    run_distributed(test_aggregate_shallow_slide_nodes, [3], world_size=1)
    run_distributed(test_aggregate_shallow_slide_nodes, [3], world_size=2)


def assert_equal_top_bottom_attention_tiles(
    slide_ids: List[str], data: Dict, results: Dict, num_top_tiles: int, slide_nodes: List[SlideNode]
) -> None:
    """Asserts that top and bottom tiles selected on the fly by the top bottom tiles handler are equal to the expected
    top and bottom tiles in the mock dataset.

    :param slide_ids: A list of expected slide ids0
    :param data: A dictionary containing the entire dataset.
    :param results: A dictionary of data results.
    :param num_top_tiles: The number of tiles to select as top and bottom tiles for each top/bottom slide.
    :param slide_nodes: The top or bottom slide nodes selected on the fly by the handler.
    """
    for i, slide_id in enumerate(slide_ids):
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

        top_tiles = slide_nodes[i].top_tiles
        bottom_tiles = slide_nodes[i].bottom_tiles

        for j, expected_top_tile in enumerate(expected_top_tiles):
            assert torch.equal(expected_top_tile.cpu(), top_tiles[j].data)
            assert expected_top_attns[j].item() == top_tiles[j].attn

        for j, expected_bottom_tile in enumerate(expected_bottom_tiles):
            assert torch.equal(expected_bottom_tile.cpu(), bottom_tiles[j].data)
            assert expected_bottom_attns[j].item() == bottom_tiles[j].attn


@pytest.mark.parametrize("n_classes", [2, 3])  # n_classes=2 represents the binary case.
def test_select_k_top_bottom_tiles_on_the_fly(
    n_classes: int, rank: int = 0, world_size: int = 1, device: str = "cpu"
) -> None:
    """This tests checks that k top and bottom tiles are selected properly `on the fly`:
        1- Create a mock dataset and corresponding mock results that are small enough to fit in memory
        2- Create a handler that is only exposed to a subset of the data distributed across devices. This handler
           updates its top and bottom slides and tiles sequentially as we processes smaller batches of data.
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

    torch.manual_seed(42)
    data = _create_mock_data(n_samples=batch_size * total_batches, n_tiles=n_tiles, device=device)
    results = _create_mock_results(
        n_samples=batch_size * total_batches, n_tiles=n_tiles, n_classes=n_classes, device=device
    )

    handler = _create_and_update_top_bottom_tiles_handler(
        data, results, num_top_slides, num_top_tiles, n_classes, rank, batch_size, n_batches
    )
    handler.gather_selected_tiles_across_devices()

    if rank == 0:
        for label in range(n_classes):
            expected_top_slides_ids = get_expected_top_slides_by_probability(results, num_top_slides, label)
            assert expected_top_slides_ids == [slide_node.slide_id for slide_node in handler.top_slides_heaps[label]]
            assert_equal_top_bottom_attention_tiles(
                expected_top_slides_ids, data, results, num_top_tiles, handler.top_slides_heaps[label]
            )

            expected_bottom_slides_ids = get_expected_bottom_slides_by_probability(results, num_top_slides, label)
            assert expected_bottom_slides_ids == [
                slide_node.slide_id for slide_node in handler.bottom_slides_heaps[label]
            ]
            assert_equal_top_bottom_attention_tiles(
                expected_bottom_slides_ids, data, results, num_top_tiles, handler.bottom_slides_heaps[label]
            )


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_select_k_top_bottom_tiles_on_the_fly_distributed() -> None:
    """These tests need to be called sequentially to prevent them to be run in parallel"""
    # test with n_classes = 2
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [2], world_size=1)
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [2], world_size=2)
    # test with n_classes = 3
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [3], world_size=1)
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [3], world_size=2)


@pytest.fixture
def slide_node() -> SlideNode:
    """Fixture to create a mock slide node with corresponding top and bottom tiles."""
    torch.manual_seed(42)
    tile_size = (3, 224, 224)
    num_top_tiles = 12
    slide_node = SlideNode(slide_id="slide_0", prob_score=0.5)
    top_attn_scores = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88]
    slide_node.top_tiles = [
        TileNode(attn=top_attn_scores[i], data=torch.randint(0, 255, tile_size)) for i in range(num_top_tiles)
    ]
    bottom_attn_scores = [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007]
    slide_node.bottom_tiles = [
        TileNode(attn=bottom_attn_scores[i], data=torch.randint(0, 255, tile_size)) for i in range(num_top_tiles)
    ]
    return slide_node


def assert_plot_tiles_figure(tiles_fig: plt.Figure, fig_name: str, test_output_dirs: OutputFolderForTests) -> None:
    assert isinstance(tiles_fig, plt.Figure)
    file = Path(test_output_dirs.root_dir) / fig_name
    save_figure(fig=tiles_fig, figpath=file)
    assert file.exists()
    expected = full_ml_test_data_path("top_bottom_tiles") / fig_name
    # To update the stored results, uncomment this line:
    # expected.write_bytes(file.read_bytes())
    assert_binary_files_match(file, expected)


@pytest.mark.skipif(is_windows(), reason="Rendering is different on Windows")
def test_plot_top_bottom_tiles(slide_node: SlideNode, test_output_dirs: OutputFolderForTests) -> None:
    top_tiles_fig = slide_node.plot_attention_tiles(tile_nodes=slide_node.top_tiles, case="TP")
    bottom_tiles_fig = slide_node.plot_attention_tiles(tile_nodes=slide_node.bottom_tiles, case="FN")
    assert_plot_tiles_figure(top_tiles_fig, "slide_0_top.png", test_output_dirs)
    assert_plot_tiles_figure(bottom_tiles_fig, "slide_0_bottom.png", test_output_dirs)


@pytest.mark.parametrize("num_top_slides, num_top_tiles", [(0, 0), (0, 1), (2, 0)])
def test_disable_top_bottom_tiles_handler(num_top_slides: int, num_top_tiles: int) -> None:
    try:
        _ = TopBottomTilesHandler(n_classes=2, num_top_slides=num_top_slides, num_top_tiles=num_top_tiles)
    except Exception as err:
        assert num_top_slides > 0 and num_top_tiles == 0
        assert isinstance(err, AssertionError)


def test_tiles_are_selected_only_with_non_zero_num_top_slides(
    rank: int = 0, world_size: int = 1, device: str = "cpu"
) -> None:
    n_tiles = 3
    batch_size = 1
    n_batches = 2
    total_batches = n_batches * world_size
    num_top_tiles = 2
    num_top_slides = 0
    n_classes = 2

    torch.manual_seed(42)
    data = _create_mock_data(n_samples=batch_size * total_batches, n_tiles=n_tiles, device=device)
    results = _create_mock_results(
        n_samples=batch_size * total_batches, n_tiles=n_tiles, n_classes=n_classes, device=device
    )

    handler = TopBottomTilesHandler(n_classes, num_top_slides=num_top_slides, num_top_tiles=num_top_tiles)

    with patch.object(handler, "_update_label_slides") as mock_update_label_slides:
        for i in range(rank * n_batches, (rank + 1) * n_batches):
            batch_data = _batch_data(data, batch_idx=i, batch_size=batch_size)
            batch_results = _batch_data(results, batch_idx=i, batch_size=batch_size)
            handler.update_slides_selection(batch_data, batch_results)
    mock_update_label_slides.assert_not_called()

    for class_id in range(n_classes):
        assert len(handler.top_slides_heaps[class_id]) == 0

    with patch.object(handler, "_shallow_copy_slides_heaps") as mock_shallow_copy_slides_heaps:
        handler.gather_selected_tiles_across_devices()
    mock_shallow_copy_slides_heaps.assert_not_called()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_tiles_are_selected_only_with_non_zero_num_top_slides_distributed() -> None:
    """These tests need to be called sequentially to prevent them to be run in parallel"""
    run_distributed(test_tiles_are_selected_only_with_non_zero_num_top_slides, world_size=1)
    run_distributed(test_tiles_are_selected_only_with_non_zero_num_top_slides, world_size=2)
