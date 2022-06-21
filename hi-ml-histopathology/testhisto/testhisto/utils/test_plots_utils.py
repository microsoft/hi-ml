#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Set
from unittest.mock import MagicMock, patch
import pytest
from histopathology.utils.naming import ModelKey, PlotOption
from histopathology.utils.plots_utils import DeepMILPlotsHandler
from histopathology.utils.tiles_selection_utils import SlideNode, TilesSelector


def test_plots_handler_wrong_plot_options() -> None:
    plot_options = {PlotOption.HISTOGRAM, "foo"}
    with pytest.raises(ValueError) as ex:
        _ = DeepMILPlotsHandler(plot_options)  # type: ignore
    assert "The selected plot option is not a valid option" in str(ex)


def test_plots_handler_wrong_class_names() -> None:
    plot_options = {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX}
    with pytest.raises(ValueError) as ex:
        _ = DeepMILPlotsHandler(plot_options, class_names=[])
    assert "No class_names were provided while activating confusion matrix plotting." in str(ex)


def assert_plot_func_called_if_among_plot_options(
    mock_plot_func: MagicMock, plot_option: PlotOption, plot_options: Set[PlotOption]
) -> int:
    calls_count = 0
    if plot_option in plot_options:
        mock_plot_func.assert_called()
        calls_count += 1
    else:
        mock_plot_func.assert_not_called()
    return calls_count


@pytest.mark.parametrize(
    "plot_options",
    [
        {},
        {PlotOption.HISTOGRAM},
        {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX},
        {PlotOption.HISTOGRAM, PlotOption.TOP_BOTTOM_TILES, PlotOption.SLIDE_THUMBNAIL_HEATMAP},
        {
            PlotOption.HISTOGRAM,
            PlotOption.CONFUSION_MATRIX,
            PlotOption.TOP_BOTTOM_TILES,
            PlotOption.SLIDE_THUMBNAIL_HEATMAP,
        },
    ],
)
@patch("histopathology.utils.plots_utils.save_confusion_matrix")
@patch("histopathology.utils.plots_utils.save_scores_histogram")
@patch("histopathology.utils.plots_utils.save_top_and_bottom_tiles")
@patch("histopathology.utils.plots_utils.save_slide_thumbnail_and_heatmap")
def test_plots_handler_plots_only_desired_plot_options(
    mock_slide: MagicMock,
    mock_tile: MagicMock,
    mock_histogram: MagicMock,
    mock_conf: MagicMock,
    plot_options: Set[PlotOption],
) -> None:
    plots_handler = DeepMILPlotsHandler(plot_options, class_names=["foo"])
    plots_handler.slides_dataset = MagicMock()

    slide_node = SlideNode(slide_id="1", prob_score=0.5, true_label=1, pred_label=0)
    tiles_selector = TilesSelector(n_classes=2, num_slides=4, num_tiles=2)
    tiles_selector.top_slides_heaps = {0: [slide_node] * 4, 1: [slide_node] * 4}

    plots_handler.save_all_plot_options(
        outputs_dir=MagicMock(), tiles_selector=tiles_selector, results=MagicMock(), stage=ModelKey.VAL
    )
    calls_count = 0
    calls_count += assert_plot_func_called_if_among_plot_options(
        mock_slide, PlotOption.SLIDE_THUMBNAIL_HEATMAP, plot_options
    )
    calls_count += assert_plot_func_called_if_among_plot_options(
        mock_tile, PlotOption.TOP_BOTTOM_TILES, plot_options
    )
    calls_count += assert_plot_func_called_if_among_plot_options(mock_histogram, PlotOption.HISTOGRAM, plot_options)
    calls_count += assert_plot_func_called_if_among_plot_options(
        mock_conf, PlotOption.CONFUSION_MATRIX, plot_options
    )
    assert calls_count == len(plot_options)
