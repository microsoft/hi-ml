#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional
from unittest.mock import MagicMock, patch
import pytest
import torch
import numpy as np

from health_ml.utils.common_utils import is_windows
from health_ml.utils.fixed_paths import OutputFolderForTests
from health_cpath.datasets.panda_dataset import PandaDataset
from health_cpath.preprocessing.loading import LoadingParams, ROIType
from health_cpath.utils.naming import PlotOption, ResultsKey
from health_cpath.utils.plots_utils import (DeepMILPlotsHandler, save_confusion_matrix, save_pr_curve,
                                            save_roc_curve, get_list_from_results_dict, get_stratified_outputs)
from health_cpath.utils.tiles_selection_utils import SlideNode, TilesSelector
from testhisto.mocks.container import MockDeepSMILETilesPanda
from testhisto.utils.utils_testhisto import assert_binary_files_match, full_ml_test_data_path


def test_plots_handler_wrong_class_names() -> None:
    plot_options = {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX}
    with pytest.raises(ValueError, match=r"No class_names were provided while activating confusion matrix plotting."):
        _ = DeepMILPlotsHandler(plot_options, class_names=[], loading_params=LoadingParams())


@pytest.mark.parametrize("roi_type", [r for r in ROIType])
def test_plots_handler_always_uses_roid_loading(roi_type: ROIType) -> None:
    plot_options = {PlotOption.HISTOGRAM, PlotOption.CONFUSION_MATRIX}
    loading_params = LoadingParams(roi_type=roi_type)
    plots_handler = DeepMILPlotsHandler(plot_options, class_names=["foo", "bar"], loading_params=loading_params)
    assert plots_handler.loading_params.roi_type in [ROIType.MASK, ROIType.FOREGROUND, ROIType.MASKSUBROI]


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
            PlotOption.ATTENTION_HISTOGRAM,
            PlotOption.ROC_CURVE,
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
        PlotOption.ATTENTION_HISTOGRAM: patch("health_cpath.utils.plots_utils.save_attention_histogram"),
        PlotOption.TOP_BOTTOM_TILES: patch("health_cpath.utils.plots_utils.save_top_and_bottom_tiles"),
        PlotOption.CONFUSION_MATRIX: patch("health_cpath.utils.plots_utils.save_confusion_matrix"),
        PlotOption.HISTOGRAM: patch("health_cpath.utils.plots_utils.save_scores_histogram"),
        PlotOption.PR_CURVE: patch("health_cpath.utils.plots_utils.save_pr_curve"),
        PlotOption.ROC_CURVE: patch("health_cpath.utils.plots_utils.save_roc_curve"),
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


@pytest.mark.parametrize("stratify_metadata", [["A", "B", "A", "A", "B", "B"], None])
def test_pr_curve_integration(tmp_path: Path, caplog: pytest.LogCaptureFixture,
                              stratify_metadata: Optional[List[Any]]) -> None:
    results = {
        ResultsKey.TRUE_LABEL: [0, 1, 0, 1, 0, 1],
        ResultsKey.PROB: [0.1, 0.8, 0.6, 0.3, 0.5, 0.4]
    }

    # check plot is produced and it has right filename
    save_pr_curve(results, tmp_path, stage='foo', stratify_metadata=stratify_metadata)      # type: ignore
    file = Path(tmp_path) / "pr_curve_foo.png"
    assert file.exists()
    os.remove(file)

    # check warning is logged and plot is not produced if NOT a binary case
    results[ResultsKey.TRUE_LABEL] = [0, 1, 0, 2, 0, 1]

    save_pr_curve(results, tmp_path, stage='foo', stratify_metadata=stratify_metadata)      # type: ignore
    warning_message = "The PR curve plot implementation works only for binary cases, this plot will be skipped."
    assert warning_message in caplog.records[-1].getMessage()

    assert not file.exists()


@pytest.mark.parametrize("stratify_metadata", [["A", "B", "A", "A", "B", "B"], None])
def test_roc_curve_integration(tmp_path: Path, caplog: pytest.LogCaptureFixture,
                               stratify_metadata: Optional[List[Any]]) -> None:
    results = {
        ResultsKey.TRUE_LABEL: [0, 1, 0, 1, 0, 1],
        ResultsKey.PROB: [0.1, 0.8, 0.6, 0.3, 0.5, 0.4]
    }
    stratify_metadata = ["A", "B", "A", "A", "B", "B"]

    # check plot is produced and it has right filename
    save_roc_curve(results, tmp_path, stage='foo', stratify_metadata=stratify_metadata)      # type: ignore
    file = Path(tmp_path) / "roc_curve_foo.png"
    assert file.exists()
    os.remove(file)

    # check warning is logged and plot is not produced if NOT a binary case
    results[ResultsKey.TRUE_LABEL] = [0, 1, 0, 2, 0, 1]

    save_roc_curve(results, tmp_path, stage='foo', stratify_metadata=stratify_metadata)      # type: ignore
    warning_message = "The ROC curve plot implementation works only for binary cases, this plot will be skipped."
    assert warning_message in caplog.records[-1].getMessage()

    assert not file.exists()


def test_get_list_from_results_dict() -> None:
    results = {ResultsKey.TRUE_LABEL: [torch.tensor(0), torch.tensor(1), torch.tensor(0)],
               ResultsKey.PRED_LABEL: [torch.tensor(1), torch.tensor(0), torch.tensor(0)],
               ResultsKey.PROB: [torch.tensor(0.9), torch.tensor(0.6), torch.tensor(0.8)]}
    true_labels = get_list_from_results_dict(results=results, results_key=ResultsKey.TRUE_LABEL)
    scores = get_list_from_results_dict(results=results, results_key=ResultsKey.PROB)
    pred_labels = get_list_from_results_dict(results=results, results_key=ResultsKey.PRED_LABEL)
    assert all(isinstance(x, int) for x in true_labels)
    assert all(isinstance(x, float) for x in scores)
    assert all(isinstance(x, int) for x in pred_labels)
    assert len(true_labels) == len(pred_labels) == len(scores)


def test_get_stratified_outputs() -> None:
    results = {ResultsKey.TRUE_LABEL: [torch.tensor(0), torch.tensor(1), torch.tensor(0), torch.tensor(1)],
               ResultsKey.PROB: [torch.tensor(0.9), torch.tensor(0.6), torch.tensor(0.8), torch.tensor(0.9)]}
    stratify_metadata = ["A", "B", "A", "A"]
    true_labels = get_list_from_results_dict(results=results, results_key=ResultsKey.TRUE_LABEL)
    scores = get_list_from_results_dict(results=results, results_key=ResultsKey.PROB)
    stratified_outputs = get_stratified_outputs(true_labels=true_labels, scores=scores,
                                                stratify_metadata=stratify_metadata)
    assert isinstance(stratified_outputs, dict)
    assert len(stratified_outputs.keys()) == len(np.unique(stratify_metadata))
    for key in stratified_outputs.keys():
        assert len(stratified_outputs[key][0]) == len(stratified_outputs[key][1])


@pytest.mark.parametrize("stratify_plots_by", ['data_provider', None])
def test_plots_handler_get_metadata(mock_panda_slides_root_dir: Path, stratify_plots_by: Optional[str]) -> None:
    results = {ResultsKey.TRUE_LABEL: [torch.tensor(0), torch.tensor(1), torch.tensor(0)],
               ResultsKey.PRED_LABEL: [torch.tensor(1), torch.tensor(0), torch.tensor(0)],
               ResultsKey.PROB: [torch.tensor(0.9), torch.tensor(0.6), torch.tensor(0.8)],
               ResultsKey.SLIDE_ID: [['_0', '_0'], ['_1', '_1'], ['_2', '_2']]}

    plot_options = {PlotOption.PR_CURVE, PlotOption.ROC_CURVE}
    plots_handler = DeepMILPlotsHandler(plot_options=plot_options, class_names=[], loading_params=LoadingParams(),
                                        stratify_plots_by=stratify_plots_by)
    slides_dataset = PandaDataset(root=mock_panda_slides_root_dir)
    plots_handler.slides_dataset = slides_dataset
    metadata = plots_handler.get_metadata(results=results)    # type: ignore
    if stratify_plots_by is None:
        assert metadata is None
    else:
        assert len(metadata) == len(results[ResultsKey.PRED_LABEL])         # type: ignore
    # check if the metadata corresponds to the correct slides
    if metadata is not None:
        result_ids = [x[0] for x in results[ResultsKey.SLIDE_ID]]           # type: ignore
        for i in range(len(result_ids)):
            df = slides_dataset.dataset_df
            slide_row = df.iloc[np.where(df.index == result_ids[i])]
            metadata_val = slide_row[stratify_plots_by].values[0]
            assert metadata_val == metadata[i]


@pytest.mark.skipif(is_windows(), reason="Rendering is different on Windows")
def test_save_roc_curve_stratification(test_output_dirs: OutputFolderForTests) -> None:
    results = {
        ResultsKey.TRUE_LABEL: [0, 1, 0, 1, 0, 1],
        ResultsKey.PROB: [0.1, 0.8, 0.6, 0.3, 0.5, 0.4]
    }
    stratify_metadata = ["A", "B", "A", "A", "B", "B"]
    target_dir = Path(test_output_dirs.root_dir)
    save_roc_curve(results, target_dir, stage='stratify', stratify_metadata=stratify_metadata)         # type: ignore
    file = target_dir / "roc_curve_stratify.png"
    assert file.exists()

    expected = full_ml_test_data_path("histo_heatmaps") / "plot_roc_curve_stratify.png"
    # To update the stored results, uncomment this line:
    # expected.write_bytes(file.read_bytes())
    assert_binary_files_match(file, expected)
