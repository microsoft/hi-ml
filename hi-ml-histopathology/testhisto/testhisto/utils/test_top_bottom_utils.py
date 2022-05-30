import torch
import pytest
import numpy as np
from typing import Dict, Generator, List, Tuple, Any

from testhisto.utils.utils_testhisto import run_distributed
from histopathology.utils.naming import ResultsKey, SlideKey
from histopathology.utils.top_bottom_tiles_utils import TopBottomTilesHandler, SlideNode


def _create_mock_data(n_samples: int, device: str = "cpu") -> Dict:
    n_tiles = 6
    tile_size = (1, 4, 4)
    diff_n_tiles = [n_tiles + i for i in range(n_samples)]
    mock_data = {
        SlideKey.SLIDE_ID: np.array([f"slide_{i}" for i in range(n_samples)]),
        SlideKey.IMAGE: [torch.randint(0, 255, (diff_n_tiles[i], *tile_size), device=device) for i in range(n_samples)],
    }
    return mock_data


def _create_mock_results(n_samples: int, n_classes: int = 2, device: str = "cpu") -> Dict:
    n_tiles: int = 3
    diff_n_tiles = [n_tiles + i for i in range(n_samples)]
    mock_results = {
        ResultsKey.SLIDE_ID: np.array([f"slide_{i}" for i in range(n_samples)]),
        ResultsKey.TRUE_LABEL: torch.randint(2, size=(n_samples,), device=device),
        ResultsKey.BAG_ATTN: [torch.rand(size=(1, diff_n_tiles[i]), device=device) for i in range(n_samples)],
        ResultsKey.CLASS_PROBS: torch.rand((n_samples, n_classes), device=device),
    }
    return mock_results


def _batch_data(data, batch_idx: int, batch_size: int) -> Generator:
    """Helper function to generate smaller batches from a dictionary."""
    batch = {}
    for k in data:
        batch[k] = data[k][batch_idx * batch_size: (batch_idx + 1) * batch_size]
    yield batch


def _select_slides_by_probability(
    results: Dict[ResultsKey, Any], n_top_slides: int = 5, label: int = 1, top: bool = True
) -> Tuple[List[str], torch.Tensor]:
    """Select top or bottom slides accoring to their probability scores."""
    class_indices = (results[ResultsKey.TRUE_LABEL].squeeze() == label).nonzero().squeeze(1)
    class_prob = results[ResultsKey.CLASS_PROBS][class_indices, label]
    assert class_prob.shape == (len(class_indices),)
    n_top_slides = min(n_top_slides, len(class_prob))

    _, sorting_indices = class_prob.topk(n_top_slides, largest=top, sorted=True)
    sorted_class_indices = class_indices[sorting_indices]
    return [results[ResultsKey.SLIDE_ID][i] for i in sorted_class_indices]


def _assert_equal_top_bottom_tiles(
    slide_ids: List[str], batches: Dict, results: Dict, n_top_tiles: int, slide_nodes: List[SlideNode]
) -> None:
    for i, slide_id in enumerate(slide_ids):
        slide_batch_idx = int(slide_id.split("_")[1])
        tiles = batches[SlideKey.IMAGE][slide_batch_idx]

        _, top_tiles_ids = torch.topk(
            results[ResultsKey.BAG_ATTN][slide_batch_idx].squeeze(), k=n_top_tiles, largest=True, sorted=True
        )
        _, bottom_tiles_ids = torch.topk(
            results[ResultsKey.BAG_ATTN][slide_batch_idx].squeeze(), k=n_top_tiles, largest=False, sorted=True
        )

        expected_top_tiles = [tiles[tile_id] for tile_id in top_tiles_ids]
        expected_bottom_tiles = [tiles[tile_id] for tile_id in bottom_tiles_ids]

        assert all(
            torch.equal(expected_top_tile, top_tile.data)
            for expected_top_tile, top_tile in zip(expected_top_tiles, slide_nodes[-(i + 1)].top_tiles)
        )
        assert all(
            torch.equal(expected_bottom_tile, bottom_tile.data)
            for expected_bottom_tile, bottom_tile in zip(expected_bottom_tiles, slide_nodes[-(i + 1)].bottom_tiles)
        )


@pytest.mark.parametrize("n_classes", [2, 3])
def test_gather_shallow_slide_nodes(n_classes: int, rank: int = 0, world_size: int = 1, device: str = "cpu") -> None:

    batch_size = 2
    n_batches = 10
    total_batches = n_batches * world_size

    n_top_tiles = 2
    n_top_slides = 2
    handler = TopBottomTilesHandler(n_classes, n_top_slides=n_top_slides, n_top_tiles=n_top_tiles)

    torch.manual_seed(42)
    data = _create_mock_data(n_samples=batch_size * total_batches, device=device)
    results = _create_mock_results(n_samples=batch_size * total_batches, n_classes=n_classes, device=device)

    for i in range(rank * n_batches, (rank + 1) * n_batches):
        batch_data = next(_batch_data(data, batch_idx=i, batch_size=batch_size))
        batch_results = next(_batch_data(results, batch_idx=i, batch_size=batch_size))
        handler.update_top_bottom_slides_heaps(batch_data, batch_results)

    shallow_top_slides_heaps = handler.shallow_copy_top_slides_heaps()
    shallow_bottom_slides_heaps = handler.shallow_copy_bottom_slides_heaps()

    if torch.distributed.is_initialized():
        if world_size > 1:
            shallow_top_slides_heaps = handler.gather_shallow_slides_heaps(world_size, shallow_top_slides_heaps)
            shallow_bottom_slides_heaps = handler.gather_shallow_slides_heaps(world_size, shallow_bottom_slides_heaps)

    for label in range(n_classes):
        top_slides_ids = _select_slides_by_probability(results, n_top_slides, label, top=True)
        assert top_slides_ids == [slide_node.slide_id for slide_node in shallow_top_slides_heaps[label]][::-1]

        bottom_slides_ids = _select_slides_by_probability(results, n_top_slides, label, top=False)
        assert bottom_slides_ids == [slide_node.slide_id for slide_node in shallow_bottom_slides_heaps[label]][::-1]


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_gather_shallow_slide_nodes_distributed() -> None:
    # These tests need to be called sequentially to prevent them to be run in parallel
    # test with n_classes = 2
    run_distributed(test_gather_shallow_slide_nodes, [2], world_size=1)
    run_distributed(test_gather_shallow_slide_nodes, [2], world_size=2)
    # test with n_classes = 3
    run_distributed(test_gather_shallow_slide_nodes, [3], world_size=1)
    run_distributed(test_gather_shallow_slide_nodes, [3], world_size=2)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_select_k_top_bottom_tiles_on_the_fly(
    n_classes: int, rank: int = 0, world_size: int = 1, device: str = "cpu"
) -> None:

    batch_size = 2
    n_batches = 10
    total_batches = n_batches * world_size

    n_top_tiles = 2
    n_top_slides = 2
    handler = TopBottomTilesHandler(n_classes, n_top_slides=n_top_slides, n_top_tiles=n_top_tiles)

    torch.manual_seed(42)
    data = _create_mock_data(n_samples=batch_size * total_batches, device=device)
    results = _create_mock_results(n_samples=batch_size * total_batches, n_classes=n_classes, device=device)

    for i in range(rank * n_batches, (rank + 1) * n_batches):
        batch_data = next(_batch_data(data, batch_idx=i, batch_size=batch_size))
        batch_results = next(_batch_data(results, batch_idx=i, batch_size=batch_size))
        handler.update_top_bottom_slides_heaps(batch_data, batch_results)

    handler.gather_top_bottom_tiles_for_top_bottom_slides()

    for label in range(n_classes):
        top_slides_ids = _select_slides_by_probability(results, n_top_slides, label, top=True)
        assert top_slides_ids == [slide_node.slide_id for slide_node in handler.top_slides_heaps[label]][::-1]
        _assert_equal_top_bottom_tiles(top_slides_ids, data, results, n_top_tiles, handler.top_slides_heaps[label])

        bottom_slides_ids = _select_slides_by_probability(results, n_top_slides, label, top=False)
        assert bottom_slides_ids == [slide_node.slide_id for slide_node in handler.bottom_slides_heaps[label]][::-1]
        _assert_equal_top_bottom_tiles(
            bottom_slides_ids, data, results, n_top_tiles, handler.bottom_slides_heaps[label]
        )


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Not enough GPUs available")
@pytest.mark.gpu
def test_select_k_top_bottom_tiles_on_the_fly_distributed() -> None:
    # These tests need to be called sequentially to prevent them to be run in parallel
    # test with n_classes = 2
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [2], world_size=1)
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [2], world_size=2)
    # test with n_classes = 3
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [3], world_size=1)
    run_distributed(test_select_k_top_bottom_tiles_on_the_fly, [3], world_size=2)
