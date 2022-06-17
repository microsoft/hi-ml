#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Set
from unittest.mock import MagicMock, patch
import pytest
from histopathology.utils.naming import ModelKey, PlotOptionsKey

from histopathology.utils.plots_utils import DeepMILPlotsHandler


@pytest.mark.parametrize("plot_options", [{PlotOptionsKey.HISTOGRAM}, {PlotOptionsKey.HISTOGRAM, "foo"}])
def test_plots_handler_wrong_plot_options(plot_options: Set[PlotOptionsKey]) -> None:
    try:
        _ = DeepMILPlotsHandler(plot_options)
    except Exception as err:
        assert isinstance(err, ValueError)


def assert_plot_func_called_if_among_plot_options(
    mock_plot_func: MagicMock, plot_option: PlotOptionsKey, plot_options: Set[PlotOptionsKey]
) -> int:
    calls_count = 0
    if plot_option in plot_options:
        mock_plot_func.assert_called_once()
        calls_count += 1
    else:
        mock_plot_func.assert_not_called()
    return calls_count


@pytest.mark.parametrize(
    "plot_options",
    [
        {},
        {PlotOptionsKey.HISTOGRAM},
        {PlotOptionsKey.HISTOGRAM, PlotOptionsKey.CONFUSION_MATRIX},
        {PlotOptionsKey.HISTOGRAM, PlotOptionsKey.TOP_BOTTOM_TILES, PlotOptionsKey.SLIDE_THUMBNAIL_HEATMAP},
        {
            PlotOptionsKey.HISTOGRAM,
            PlotOptionsKey.CONFUSION_MATRIX,
            PlotOptionsKey.TOP_BOTTOM_TILES,
            PlotOptionsKey.SLIDE_THUMBNAIL_HEATMAP,
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
    plot_options: Set[PlotOptionsKey],
) -> None:
    plots_handler = DeepMILPlotsHandler(plot_options, class_names=["foo"])
    plots_handler.save_all_plot_options(
        outputs_dir=MagicMock(), tiles_selector=MagicMock(), results=MagicMock(), stage=ModelKey.VAL
    )
    calls_count = 0
    calls_count += assert_plot_func_called_if_among_plot_options(
        mock_slide, PlotOptionsKey.SLIDE_THUMBNAIL_HEATMAP, plot_options
    )
    calls_count += assert_plot_func_called_if_among_plot_options(
        mock_tile, PlotOptionsKey.TOP_BOTTOM_TILES, plot_options
    )
    calls_count += assert_plot_func_called_if_among_plot_options(mock_histogram, PlotOptionsKey.HISTOGRAM, plot_options)
    calls_count += assert_plot_func_called_if_among_plot_options(
        mock_conf, PlotOptionsKey.CONFUSION_MATRIX, plot_options
    )
    assert calls_count == len(plot_options)
