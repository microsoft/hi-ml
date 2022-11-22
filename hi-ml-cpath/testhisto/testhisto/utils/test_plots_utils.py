#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
from pathlib import Path
from typing import Any, Collection, Dict, List
from unittest.mock import MagicMock, patch
import pytest
from health_cpath.preprocessing.loading import LoadingParams, ROIType
from health_cpath.utils.naming import PlotOption, ResultsKey
from health_cpath.utils.plots_utils import DeepMILPlotsHandler, save_confusion_matrix, save_pr_curve
from health_cpath.utils.tiles_selection_utils import SlideNode, TilesSelector
from testhisto.mocks.container import MockDeepSMILETilesPanda


def test_plots_handler_wrong_class_names() -> None:
    plot_options = {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX}
    with pytest.raises(ValueError, match=r"No class_names were provided while activating confusion matrix plotting."):
        _ = DeepMILPlotsHandler(plot_options, class_names=[], loading_params=LoadingParams())


@pytest.mark.parametrize("roi_type", [r for r in ROIType])
def test_plots_handler_always_uses_roid_loading(roi_type: ROIType) -> None:
    plot_options = {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX}
    loading_params = LoadingParams(roi_type=roi_type)
    plots_handler = DeepMILPlotsHandler(plot_options, class_names=["foo", "bar"], loading_params=loading_params)
    assert plots_handler.loading_params.roi_type in [ROIType.MASK, ROIType.FOREGROUND]


@pytest.mark.parametrize(
    "slide_plot_options",
    [
        [PlotOption.SLIDE_THUMBNAIL],
        [PlotOption.ATTENTION_HEATMAP],
        [PlotOption.SLIDE_THUMBNAIL, PlotOption.ATTENTION_HEATMAP]
    ],
)
def test_plots_handler_slide_plot_options_without_slide_dataset(slide_plot_options: List[PlotOption]) -> None:
    exception_prompt = f"Plot option {slide_plot_options[0]} requires a slides dataset"
    with pytest.raises(ValueError, match=rf"{exception_prompt}"):
        container = MockDeepSMILETilesPanda(tmp_path=Path("foo"))
        container.setup()
        container.data_module = MagicMock()
        container.data_module.train_dataset.n_classes = 6
        outputs_handler = container.get_outputs_handler()
        outputs_handler.test_plots_handler.plot_options = slide_plot_options
        outputs_handler.set_slides_dataset_for_plots_handlers(container.get_slides_dataset())


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
        {PlotOption.HISTOGRAM, PlotOption.PR_CURVE},
        {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX},
        {PlotOption.HISTOGRAM, PlotOption.TOP_BOTTOM_TILES, PlotOption.ATTENTION_HEATMAP},
        {
            PlotOption.HISTOGRAM,
            PlotOption.PR_CURVE,
            PlotOption.CONFUSION_MATRIX,
            PlotOption.TOP_BOTTOM_TILES,
            PlotOption.SLIDE_THUMBNAIL,
            PlotOption.ATTENTION_HEATMAP,
        },
    ],
)
def test_plots_handler_plots_only_desired_plot_options(plot_options: Collection[PlotOption]) -> None:
    plots_handler = DeepMILPlotsHandler(plot_options, class_names=["foo1", "foo2"], loading_params=LoadingParams())
    plots_handler.slides_dataset = MagicMock()

    n_tiles = 4
    slide_node = SlideNode(slide_id="1", gt_prob_score=0.2, pred_prob_score=0.8, true_label=1, pred_label=0)
    tiles_selector = TilesSelector(n_classes=2, num_slides=4, num_tiles=2)
    tiles_selector.top_slides_heaps = {0: [slide_node] * n_tiles, 1: [slide_node] * n_tiles}
    tiles_selector.bottom_slides_heaps = {0: [slide_node] * n_tiles, 1: [slide_node] * n_tiles}

    patchers: Dict[PlotOption, Any] = {
        PlotOption.SLIDE_THUMBNAIL: patch("health_cpath.utils.plots_utils.save_slide_thumbnail"),
        PlotOption.ATTENTION_HEATMAP: patch("health_cpath.utils.plots_utils.save_attention_heatmap"),
        PlotOption.TOP_BOTTOM_TILES: patch("health_cpath.utils.plots_utils.save_top_and_bottom_tiles"),
        PlotOption.CONFUSION_MATRIX: patch("health_cpath.utils.plots_utils.save_confusion_matrix"),
        PlotOption.HISTOGRAM: patch("health_cpath.utils.plots_utils.save_scores_histogram"),
        PlotOption.PR_CURVE: patch("health_cpath.utils.plots_utils.save_pr_curve"),
    }

    mock_funcs = {option: patcher.start() for option, patcher in patchers.items()}  # type: ignore
    with patch.object(plots_handler, "get_slide_dict"):
        plots_handler.save_plots(outputs_dir=MagicMock(), tiles_selector=tiles_selector, results=MagicMock())
    patch.stopall()

    calls_count = 0
    for option, mock_func in mock_funcs.items():
        calls_count += assert_plot_func_called_if_among_plot_options(mock_func, option, plot_options)

    assert calls_count == len(plot_options)


def test_save_conf_matrix_integration(tmp_path: Path) -> None:
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)
    results = {
        ResultsKey.TRUE_LABEL: [0, 1, 0, 1, 0, 1],
        ResultsKey.PRED_LABEL: [0, 1, 0, 0, 0, 1]
    }
    class_names = ["foo", "bar"]

    save_confusion_matrix(results, class_names, tmp_path, stage='foo')
    file = Path(tmp_path) / "normalized_confusion_matrix_foo.png"
    assert file.exists()

    # check that an error is raised if true labels include indices greater than the expected number of classes
    invalid_results_1 = {
        ResultsKey.TRUE_LABEL: [0, 1, 0, 1, 0, 2],
        ResultsKey.PRED_LABEL: [0, 1, 0, 0, 0, 1]
    }
    with pytest.raises(ValueError) as e:
        save_confusion_matrix(invalid_results_1, class_names, tmp_path)
    assert "More entries were found in true labels than are available in class names" in str(e)

    # check that an error is raised if prediced labels include indices greater than the expected number of classes
    invalid_results_2 = {
        ResultsKey.TRUE_LABEL: [0, 1, 0, 1, 0, 1],
        ResultsKey.PRED_LABEL: [0, 1, 0, 0, 0, 2]
    }
    with pytest.raises(ValueError) as e:
        save_confusion_matrix(invalid_results_2, class_names, tmp_path)
    assert "More entries were found in predicted labels than are available in class names" in str(e)

    # check that confusion matrix still has correct shape even if results don't cover all expected labels
    class_names_extended = ["foo", "bar", "baz"]
    num_classes = len(class_names_extended)
    expected_conf_matrix_shape = (num_classes, num_classes)
    with patch("health_cpath.utils.plots_utils.plot_normalized_confusion_matrix") as mock_plot_conf_matrix:
        with patch("health_cpath.utils.plots_utils.save_figure"):
            save_confusion_matrix(results, class_names_extended, tmp_path)
            mock_plot_conf_matrix.assert_called_once()
            actual_conf_matrix = mock_plot_conf_matrix.call_args[1].get('cm')
            assert actual_conf_matrix.shape == expected_conf_matrix_shape


def test_pr_curve_integration(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    results = {
        ResultsKey.TRUE_LABEL: [0, 1, 0, 1, 0, 1],
        ResultsKey.PROB: [0.1, 0.8, 0.6, 0.3, 0.5, 0.4]
    }

    # check plot is produced and it has right filename
    save_pr_curve(results, tmp_path, stage='foo')  # type: ignore
    file = Path(tmp_path) / "pr_curve_foo.png"
    assert file.exists()
    os.remove(file)

    # check warning is logged and plot is not produced if NOT a binary case
    results[ResultsKey.TRUE_LABEL] = [0, 1, 0, 2, 0, 1]

    save_pr_curve(results, tmp_path, stage='foo')  # type: ignore
    warning_message = "The PR curve plot implementation works only for binary cases, this plot will be skipped."
    assert warning_message in caplog.records[-1].getMessage()

    assert not file.exists()
