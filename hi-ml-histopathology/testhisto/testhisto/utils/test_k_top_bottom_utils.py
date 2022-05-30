import torch
import pytest
import numpy as np

from torch import Tensor
from typing import Dict, Generator, List, Tuple, Any

from histopathology.utils.k_top_bottom_tiles_utils import KTopBottomTilesHandler, SlideNode
from histopathology.utils.naming import ResultsKey, SlideKey


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
        ResultsKey.TRUE_LABEL: Tensor([[i % n_classes] for i in range(n_samples)], device=device).int(),
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
    results: Dict[ResultsKey, Any], k_slides: int = 5, label: int = 1, top: bool = True
) -> Tuple[List[str], torch.Tensor]:
    """Select top or bottom slides accoring to their probability scores."""
    class_indices = (results[ResultsKey.TRUE_LABEL].squeeze() == label).nonzero().squeeze(1)
    class_prob = results[ResultsKey.CLASS_PROBS][class_indices, label]
    assert class_prob.shape == (len(class_indices),)
    k_slides = min(k_slides, len(class_prob))

    _, sorting_indices = class_prob.topk(k_slides, largest=top, sorted=True)
    sorted_class_indices = class_indices[sorting_indices]
    return [results[ResultsKey.SLIDE_ID][i] for i in sorted_class_indices]


def _assert_equal_top_bottom_tiles(
    slide_ids: List[str], batches: Dict, results: Dict, k_tiles: int, slide_nodes: List[SlideNode]
) -> None:
    for i, slide_id in enumerate(slide_ids):
        slide_batch_idx = int(slide_id.split("_")[1])
        tiles = batches[SlideKey.IMAGE][slide_batch_idx]

        _, top_tiles_ids = torch.topk(
            results[ResultsKey.BAG_ATTN][slide_batch_idx].squeeze(), k=k_tiles, largest=True, sorted=True
        )
        _, bottom_tiles_ids = torch.topk(
            results[ResultsKey.BAG_ATTN][slide_batch_idx].squeeze(), k=k_tiles, largest=False, sorted=True
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


@pytest.mark.fast
@pytest.mark.parametrize("n_classes", [2, 3])
def test_select_k_top_bottom_on_the_fly(n_classes: int) -> None:
    batch_size = 2
    n_batches = 10

    k_tiles = 2
    k_slides = 2
    handler = KTopBottomTilesHandler(n_classes, k_slides=k_slides, k_tiles=k_tiles)

    data = _create_mock_data(n_samples=batch_size * n_batches)
    results = _create_mock_results(n_samples=batch_size * n_batches, n_classes=n_classes)

    for i in range(n_batches):
        batch_data = next(_batch_data(data, batch_idx=i, batch_size=batch_size))
        batch_results = next(_batch_data(results, batch_idx=i, batch_size=batch_size))
        handler.update_top_bottom_slides_heaps(batch_data, batch_results)

    for label in range(n_classes):
        top_slides_ids = _select_slides_by_probability(results, k_slides, label, top=True)
        assert top_slides_ids == [slide_node.slide_id for slide_node in handler.top_slides_heaps[label]][::-1]
        _assert_equal_top_bottom_tiles(top_slides_ids, data, results, k_tiles, handler.top_slides_heaps[label])

        bottom_slides_ids = _select_slides_by_probability(results, k_slides, label, top=False)
        assert bottom_slides_ids == [slide_node.slide_id for slide_node in handler.bottom_slides_heaps[label]][::-1]
        _assert_equal_top_bottom_tiles(bottom_slides_ids, data, results, k_tiles, handler.bottom_slides_heaps[label])
