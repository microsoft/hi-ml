#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import math
import random
from pathlib import Path
from typing import List, Optional

import matplotlib
import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
from pytest import LogCaptureFixture
from torch.functional import Tensor

from health_ml.utils.common_utils import is_gpu_available, is_windows
from health_ml.utils.fixed_paths import OutputFolderForTests
from health_cpath.utils.viz_utils import plot_attention_tiles, plot_scores_hist, resize_and_save, plot_slide, \
    plot_heatmap_overlay, plot_normalized_confusion_matrix
from health_cpath.utils.naming import ResultsKey
from health_cpath.utils.heatmap_utils import location_selected_tiles
from health_cpath.utils.tiles_selection_utils import SlideNode, TileNode
from health_cpath.utils.viz_utils import save_figure
from testhisto.utils.utils_testhisto import assert_binary_files_match, full_ml_test_data_path


def set_random_seed(random_seed: int, caller_name: Optional[str] = None) -> None:
    """
    Set the seed for the random number generators of python, numpy, torch.random, and torch.cuda for all gpus.
    :param random_seed: random seed value to set.
    :param caller_name: name of the caller for logging purposes.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if is_gpu_available():
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(random_seed)  # type: ignore
    prefix = ""
    if caller_name is not None:
        prefix = caller_name + ": "
    logging.debug(f"{prefix}Random seed set to: {random_seed}")


def assert_equal_lists(pred: List, expected: List) -> None:
    assert len(pred) == len(expected)
    for i, slide in enumerate(pred):
        for j, value in enumerate(slide):
            if type(value) in [int, float]:
                assert math.isclose(value, expected[i][j], rel_tol=1e-06)
            elif (type(value) == Tensor) and (value.ndim >= 1):
                for k, idx in enumerate(value):
                    assert math.isclose(idx, expected[i][j][k], rel_tol=1e-06)
            elif isinstance(value, List):
                for k, idx in enumerate(value):
                    if type(idx) in [int, float]:
                        assert math.isclose(idx, expected[i][j][k], rel_tol=1e-06)
                    elif type(idx) == Tensor:
                        assert math.isclose(idx.item(), expected[i][j][k].item(), rel_tol=1e-06)
            else:
                raise TypeError("Unexpected list composition")


test_dict = {ResultsKey.SLIDE_ID: [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4],
                                   [5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
             ResultsKey.IMAGE_PATH: [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                                     [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
             ResultsKey.CLASS_PROBS: [Tensor([0.6, 0.4]), Tensor([0.3, 0.7]), Tensor([0.6, 0.4]), Tensor([0.0, 1.0]),
                                      Tensor([0.7, 0.3]), Tensor([0.8, 0.2]), Tensor([0.1, 0.9]), Tensor([0.01, 0.99])],
             ResultsKey.TRUE_LABEL: [0, 1, 1, 1, 1, 0, 0, 0],
             ResultsKey.BAG_ATTN:
                 [Tensor([[0.10, 0.00, 0.20, 0.15]]),
                  Tensor([[0.10, 0.18, 0.15, 0.13]]),
                  Tensor([[0.25, 0.23, 0.20, 0.21]]),
                  Tensor([[0.33, 0.31, 0.37, 0.35]]),
                  Tensor([[0.43, 0.01, 0.07, 0.25]]),
                  Tensor([[0.53, 0.11, 0.17, 0.55]]),
                  Tensor([[0.63, 0.21, 0.27, 0.05]]),
                  Tensor([[0.73, 0.31, 0.37, 0.15]])],
             ResultsKey.TILE_LEFT:
                 [Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424])],
             ResultsKey.TILE_TOP:
                 [Tensor([200, 424, 200, 424]),
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424])],
             ResultsKey.TILE_RIGHT:
                 [Tensor([200, 424, 424, 424]),
                  Tensor([200, 424, 424, 424]),
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 200, 424, 424])],
             ResultsKey.TILE_BOTTOM:
                 [Tensor([200, 424, 200, 424]),
                  Tensor([200, 200, 424, 424]),
                  Tensor([200, 424, 424, 424]),
                  Tensor([200, 200, 424, 424])],
             }


@pytest.mark.skipif(is_windows(), reason="Rendering is different on Windows")
def test_plot_scores_hist(test_output_dirs: OutputFolderForTests) -> None:
    fig = plot_scores_hist(test_dict)
    assert isinstance(fig, matplotlib.figure.Figure)
    file = Path(test_output_dirs.root_dir) / "plot_score_hist.png"
    resize_and_save(5, 5, file)
    assert file.exists()
    expected = full_ml_test_data_path("histo_heatmaps") / "score_hist.png"
    # To update the stored results, uncomment this line:
    # expected.write_bytes(file.read_bytes())
    assert_binary_files_match(file, expected)


@pytest.fixture
def slide_node() -> SlideNode:
    """Fixture to create a mock slide node with corresponding top and bottom tiles."""
    set_random_seed(0)
    tile_size = (3, 224, 224)
    num_top_tiles = 12
    slide_node = SlideNode(slide_id="slide_0", gt_prob_score=0.04, pred_prob_score=0.96, true_label=1, pred_label=0)
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
    top_tiles_fig = plot_attention_tiles(
        case="FN", slide_node=slide_node, top=True, num_columns=4, figsize=(10, 10)
    )
    assert top_tiles_fig is not None
    bottom_tiles_fig = plot_attention_tiles(
        case="FN", slide_node=slide_node, top=False, num_columns=4, figsize=(10, 10)
    )
    assert bottom_tiles_fig is not None
    assert_plot_tiles_figure(top_tiles_fig, "slide_0_top.png", test_output_dirs)
    assert_plot_tiles_figure(bottom_tiles_fig, "slide_0_bottom.png", test_output_dirs)


def test_plot_attention_tiles_below_min_rows(slide_node: SlideNode, caplog: LogCaptureFixture) -> None:
    expected_warning = "The number of selected top and bottom tiles is too low, plotting will be skipped."
    "Try debugging with a higher num_top_tiles and/or a higher number of batches."
    slide_node.bottom_tiles = []
    with caplog.at_level(logging.WARNING):
        bottom_tiles_fig = plot_attention_tiles(
            case="FN", slide_node=slide_node, top=False, num_columns=4, figsize=(10, 10)
        )
        assert bottom_tiles_fig is None
        assert expected_warning in caplog.text

    slide_node.top_tiles = []
    with caplog.at_level(logging.WARNING):
        top_tiles_fig = plot_attention_tiles(
            case="FN", slide_node=slide_node, top=True, num_columns=4, figsize=(10, 10)
        )
        assert top_tiles_fig is None
        assert expected_warning in caplog.text


@pytest.mark.parametrize(
    "scale, gt_prob, pred_prob, gt_label, pred_label, case",
    [
        (0.1, 0.99, 0.99, 1, 1, "TP"),
        (1.2, 0.95, 0.95, 0, 0, "TN"),
        (2.4, 0.04, 0.96, 0, 1, "FP"),
        (3.6, 0.03, 0.97, 1, 0, "FN"),
    ],
)
def test_plot_slide(
    test_output_dirs: OutputFolderForTests,
    scale: int,
    gt_prob: float,
    pred_prob: float,
    case: str,
    gt_label: int,
    pred_label: int,
) -> None:
    set_random_seed(0)
    slide_image = np.random.rand(3, 1000, 2000)
    slide_node = SlideNode(
        slide_id="slide_0", gt_prob_score=gt_prob, pred_prob_score=pred_prob, true_label=gt_label, pred_label=pred_label
    )
    fig = plot_slide(case=case, slide_node=slide_node, slide_image=slide_image, scale=scale)
    assert isinstance(fig, matplotlib.figure.Figure)
    file = Path(test_output_dirs.root_dir) / "plot_slide.png"
    resize_and_save(5, 5, file)
    assert file.exists()
    expected = full_ml_test_data_path("histo_heatmaps") / f"slide_{scale}_{case}.png"
    # To update the stored results, uncomment this line:
    expected.write_bytes(file.read_bytes())
    assert_binary_files_match(file, expected)


@pytest.mark.skipif(is_windows(), reason="Rendering is different on Windows")
def test_plot_heatmap_overlay(test_output_dirs: OutputFolderForTests) -> None:
    set_random_seed(0)
    slide_image = np.random.rand(3, 1000, 2000)
    slide_node = SlideNode(
        slide_id=1, gt_prob_score=0.04, pred_prob_score=0.96, true_label=1, pred_label=0  # type: ignore
    )
    location_bbox = [100, 100]
    tile_size = 224
    level = 0
    fig = plot_heatmap_overlay(case="FN",
                               slide_node=slide_node,
                               slide_image=slide_image,
                               results=test_dict,  # type: ignore
                               location_bbox=location_bbox,
                               tile_size=tile_size,
                               level=level)
    assert isinstance(fig, matplotlib.figure.Figure)
    file = Path(test_output_dirs.root_dir) / "plot_heatmap_overlay.png"
    resize_and_save(5, 5, file)
    assert file.exists()
    expected = full_ml_test_data_path("histo_heatmaps") / "heatmap_overlay.png"
    # To update the stored results, uncomment this line:
    # expected.write_bytes(file.read_bytes())
    assert_binary_files_match(file, expected)


@pytest.mark.parametrize("n_classes", [1, 3])
@pytest.mark.skipif(is_windows(), reason="Rendering is different on Windows")
def test_plot_normalized_confusion_matrix(test_output_dirs: OutputFolderForTests, n_classes: int) -> None:
    set_random_seed(0)
    if n_classes > 1:
        cm = np.random.randint(1, 1000, size=(n_classes, n_classes))
        class_names = [str(i) for i in range(n_classes)]
    else:
        cm = np.random.randint(1, 1000, size=(n_classes + 1, n_classes + 1))
        class_names = [str(i) for i in range(n_classes + 1)]
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    assert (cm_n <= 1).all()

    fig = plot_normalized_confusion_matrix(cm=cm_n, class_names=class_names)
    assert isinstance(fig, matplotlib.figure.Figure)
    file = Path(test_output_dirs.root_dir) / f"plot_confusion_matrix_{n_classes}.png"
    resize_and_save(5, 5, file)
    assert file.exists()
    expected = full_ml_test_data_path("histo_heatmaps") / f"confusion_matrix_{n_classes}.png"
    # To update the stored results, uncomment this line:
    # expected.write_bytes(file.read_bytes())
    assert_binary_files_match(file, expected)


@pytest.mark.fast
@pytest.mark.parametrize("level", [0, 1, 2])
def test_location_selected_tiles(level: int) -> None:
    set_random_seed(0)
    slide = 1
    location_bbox = [100, 100]
    slide_image = np.random.rand(3, 1000, 2000)

    coords_list = []
    slide_ids = [item[0] for item in test_dict[ResultsKey.SLIDE_ID]]  # type: ignore
    slide_idx = slide_ids.index(slide)
    for tile_idx in range(len(test_dict[ResultsKey.IMAGE_PATH][slide_idx])):  # type: ignore
        tile_coords = np.transpose(np.array([test_dict[ResultsKey.TILE_LEFT][slide_idx][tile_idx].cpu().numpy(),
                                             test_dict[ResultsKey.TILE_TOP][slide_idx][tile_idx].cpu().numpy()]))
        coords_list.append(tile_coords)

    coords = np.array(coords_list)
    tile_coords_transformed = location_selected_tiles(tile_coords=coords,
                                                      location_bbox=location_bbox,
                                                      level=level)
    tile_xs, tile_ys = tile_coords_transformed.T
    level_dict = {0: 1, 1: 4, 2: 16}
    factor = level_dict[level]
    assert min(tile_xs) >= 0
    assert max(tile_xs) <= slide_image.shape[2] // factor
    assert min(tile_ys) >= 0
    assert max(tile_ys) <= slide_image.shape[1] // factor
