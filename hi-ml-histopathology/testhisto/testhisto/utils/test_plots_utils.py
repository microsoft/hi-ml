#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Collection
from unittest.mock import MagicMock, patch
import pytest
from histopathology.utils.naming import PlotOption, ResultsKey
from histopathology.utils.plots_utils import DeepMILPlotsHandler, save_confusion_matrix
from histopathology.utils.tiles_selection_utils import SlideNode, TilesSelector
from testhisto.mocks.container import MockDeepSMILETilesPanda


def test_plots_handler_wrong_class_names() -> None:
    plot_options = {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX}
    with pytest.raises(ValueError) as ex:
        _ = DeepMILPlotsHandler(plot_options, class_names=[])
    assert "No class_names were provided while activating confusion matrix plotting." in str(ex)


def test_plots_handler_slide_thumbnails_without_slide_dataset() -> None:
    with pytest.raises(ValueError) as ex:
        container = MockDeepSMILETilesPanda(tmp_path=Path("foo"))
        container.setup()
        container.data_module = MagicMock()
        container.data_module.train_dataset.N_CLASSES = 6
        outputs_handler = container.get_outputs_handler()
        outputs_handler.test_plots_handler.plot_options = {PlotOption.SLIDE_THUMBNAIL_HEATMAP}
        outputs_handler.set_slides_dataset_for_plots_handlers(container.get_slides_dataset())
    assert "You can not plot slide thumbnails and heatmaps without setting a slides_dataset." in str(ex)


def assert_plot_func_called_if_among_plot_options(
    mock_plot_func: MagicMock, plot_option: PlotOption, plot_options: Collection[PlotOption]
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
def test_plots_handler_plots_only_desired_plot_options(plot_options: Collection[PlotOption]) -> None:
    plots_handler = DeepMILPlotsHandler(plot_options, class_names=["foo"])
    plots_handler.slides_dataset = MagicMock()

    n_tiles = 4
    slide_node = SlideNode(slide_id="1", prob_score=0.5, true_label=1, pred_label=0)
    tiles_selector = TilesSelector(n_classes=2, num_slides=4, num_tiles=2)
    tiles_selector.top_slides_heaps = {0: [slide_node] * n_tiles, 1: [slide_node] * n_tiles}
    tiles_selector.bottom_slides_heaps = {0: [slide_node] * n_tiles, 1: [slide_node] * n_tiles}

    with patch("histopathology.utils.plots_utils.save_slide_thumbnail_and_heatmap") as mock_slide:
        with patch("histopathology.utils.plots_utils.save_top_and_bottom_tiles") as mock_tile:
            with patch("histopathology.utils.plots_utils.save_scores_histogram") as mock_histogram:
                with patch("histopathology.utils.plots_utils.save_confusion_matrix") as mock_conf:
                    plots_handler.save_plots(
                        outputs_dir=MagicMock(), tiles_selector=tiles_selector, results=MagicMock()
                    )

    calls_count = 0
    calls_count += assert_plot_func_called_if_among_plot_options(
        mock_slide, PlotOption.SLIDE_THUMBNAIL_HEATMAP, plot_options
    )
    calls_count += assert_plot_func_called_if_among_plot_options(mock_tile, PlotOption.TOP_BOTTOM_TILES, plot_options)
    calls_count += assert_plot_func_called_if_among_plot_options(mock_histogram, PlotOption.HISTOGRAM, plot_options)
    calls_count += assert_plot_func_called_if_among_plot_options(mock_conf, PlotOption.CONFUSION_MATRIX, plot_options)

    assert calls_count == len(plot_options)


def test_save_conf_matrix_integration(tmp_path: Path) -> None:
    results = {
        ResultsKey.TRUE_LABEL: [0, 1, 0, 1, 0, 1],
        ResultsKey.PRED_LABEL: [0, 1, 0, 0, 0, 1]
    }
    class_names = ["foo", "bar"]
    save_confusion_matrix(results, class_names, tmp_path)
    file = Path(tmp_path) / "normalized_confusion_matrix.png"
    assert file.exists()
