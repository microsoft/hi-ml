#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import more_itertools as mi
import numpy as np
import pandas as pd
import torch
from ruamel.yaml import YAML
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from torchmetrics.metric import Metric

from histopathology.datasets.base_dataset import SlidesDataset
from histopathology.utils.metrics_utils import (plot_attention_tiles, plot_heatmap_overlay,
                                                plot_normalized_confusion_matrix, plot_scores_hist, plot_slide,
                                                select_k_tiles)
from histopathology.utils.naming import MetricsKey, ResultsKey, SlideKey
from histopathology.utils.viz_utils import load_image_dict


def validate_class_names(class_names: Optional[Sequence[str]], n_classes: int) -> Tuple[str]:
    """Return valid names for the specified number of classes.

    :param class_names: List of class names. If `None`, will return `('0', '1', ...)`.
    :param n_classes: Number of classes. If `1` (binary), expects `len(class_names) == 2`.
    :return: Validated class names tuple with length `2` for binary classes (`n_classes == 1`), otherwise `n_classes`.
    """
    effective_n_classes = n_classes if n_classes > 1 else 2
    if class_names is None:
        class_names = [str(i) for i in range(effective_n_classes)]
    if len(class_names) != effective_n_classes:
        raise ValueError(f"Mismatch in number of class names ({class_names}) and number"
                         f"of classes ({effective_n_classes})")
    return tuple(class_names)


def save_figure(fig: plt.figure, figpath: Path) -> None:
    fig.savefig(figpath, bbox_inches='tight')
    plt.close(fig)


def normalize_dict_for_df(dict_old: Dict[str, Any]) -> Dict:
    # slide-level dictionaries are processed by making value dimensions uniform and converting to numpy arrays.
    # these steps are required to convert the dictionary to pandas dataframe.
    dict_new = dict()
    bag_size = len(dict_old[ResultsKey.SLIDE_ID])
    for key, value in dict_old.items():
        if key not in [ResultsKey.CLASS_PROBS, ResultsKey.PROB]:
            if isinstance(value, torch.Tensor):
                value = value.squeeze(0).cpu().numpy()
                if value.ndim == 0:
                    value = np.full(bag_size, fill_value=value)
            dict_new[key] = value
        elif key == ResultsKey.CLASS_PROBS:
            if isinstance(value, torch.Tensor):
                value = value.squeeze(0).cpu().numpy()
                for i in range(len(value)):
                    dict_new[key + str(i)] = np.repeat(value[i], bag_size)
    return dict_new


def move_list_to_device(list_encoded_features: List) -> List:
    # a list of features on cpu obtained from original list on gpu
    features_list = []
    for feature in list_encoded_features:
        feature = feature.squeeze(0).cpu()
        features_list.append(feature)
    return features_list


def collate_results(outputs: List[Dict[ResultsKey, Any]]) -> Dict[ResultsKey, List[Any]]:
    results: Dict[str, List[Any]] = {}
    for key in outputs[0].keys():
        results[key] = []
        for batch_outputs in outputs:
            results[key] += batch_outputs[key]
    return results


def save_outputs_and_features(results: Dict[ResultsKey, List[Any]], outputs_dir: Path) -> None:
    print("Saving outputs ...")
    # collate at slide level
    list_slide_dicts = []
    # any column can be used here, the assumption is that the first dimension is the N of slides
    for slide_idx in range(len(results[ResultsKey.SLIDE_ID])):
        slide_dict = {key: results[key][slide_idx] for key in results
                      if key not in [ResultsKey.IMAGE, ResultsKey.LOSS]}
        list_slide_dicts.append(slide_dict)

    assert outputs_dir.is_dir(), f"No such dir: {outputs_dir}"
    print(f"Metrics results will be output to {outputs_dir}")
    csv_filename = outputs_dir / 'test_output.csv'

    # Collect the list of dictionaries in a list of pandas dataframe and save
    df_list = []
    for slide_dict in list_slide_dicts:
        slide_dict = normalize_dict_for_df(slide_dict)
        df_list.append(pd.DataFrame.from_dict(slide_dict))
    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(csv_filename, mode='w+', header=True)


def save_features(results: Dict[ResultsKey, List[Any]], outputs_dir: Path) -> None:
    # Collect all features in a list and save
    features_list = move_list_to_device(results[ResultsKey.IMAGE])
    torch.save(features_list, outputs_dir / 'test_encoded_features.pickle')


def save_top_and_bottom_tiles(results: Dict[ResultsKey, List[Any]], n_classes: int, figures_dir: Path) \
        -> Dict[str, List[str]]:
    print("Selecting tiles ...")

    def select_k_tiles_from_results(label: int, select: Tuple[str, str]) \
            -> List[Tuple[Any, Any, List, List]]:
        return select_k_tiles(results, n_slides=10, label=label, n_tiles=10, select=select)

    # Class 0
    tn_top_tiles = select_k_tiles_from_results(label=0, select=('highest_pred', 'highest_att'))
    tn_bottom_tiles = select_k_tiles_from_results(label=0, select=('highest_pred', 'lowest_att'))
    fp_top_tiles = select_k_tiles_from_results(label=0, select=('lowest_pred', 'highest_att'))
    fp_bottom_tiles = select_k_tiles_from_results(label=0, select=('lowest_pred', 'lowest_att'))
    report_cases = {'TN': [tn_top_tiles, tn_bottom_tiles],
                    'FP': [fp_top_tiles, fp_bottom_tiles]}

    # Class 1 to n_classes-1
    n_classes_to_select = n_classes if n_classes > 1 else 2
    for i in range(1, n_classes_to_select):
        fn_top_tiles = select_k_tiles_from_results(label=i, select=('lowest_pred', 'highest_att'))
        fn_bottom_tiles = select_k_tiles_from_results(label=i, select=('lowest_pred', 'lowest_att'))
        tp_top_tiles = select_k_tiles_from_results(label=i, select=('highest_pred', 'highest_att'))
        tp_bottom_tiles = select_k_tiles_from_results(label=i, select=('highest_pred', 'lowest_att'))
        report_cases.update({'TP_' + str(i): [tp_top_tiles, tp_bottom_tiles],
                             'FN_' + str(i): [fn_top_tiles, fn_bottom_tiles]})

    selected_slide_ids: Dict[str, List[str]] = {}
    for key in report_cases.keys():
        print(f"Plotting {key} (tiles, thumbnails, attention heatmaps)...")
        key_dir = figures_dir / key
        key_dir.mkdir(parents=True, exist_ok=True)

        n_slides = len(report_cases[key][0])
        selected_slide_ids[key] = []
        for i in range(n_slides):
            slide_id, score, paths, top_attn = report_cases[key][0][i]
            fig = plot_attention_tiles(slide_id, score, paths, top_attn, key + '_top', ncols=4)
            save_figure(fig=fig, figpath=key_dir / f'{slide_id}_top.png')

            _, _, paths, bottom_attn = report_cases[key][1][i]
            fig = plot_attention_tiles(slide_id, score, paths, bottom_attn, key + '_bottom', ncols=4)
            save_figure(fig=fig, figpath=key_dir / f'{slide_id}_bottom.png')

            selected_slide_ids[key].append(slide_id)

    return selected_slide_ids


def save_slide_thumbnails_and_heatmaps(results: Dict[ResultsKey, List[Any]], selected_slide_ids: Dict[str, List[str]],
                                       tile_size: int, level: int, slide_dataset: SlidesDataset,
                                       figures_dir: Path) -> None:
    for key in selected_slide_ids:
        print(f"Plotting {key} (tiles, thumbnails, attention heatmaps)...")
        key_dir = figures_dir / key
        key_dir.mkdir(parents=True, exist_ok=True)
        for slide_id in selected_slide_ids[key]:
            save_slide_thumbnail_and_heatmap(results, slide_id=slide_id, tile_size=tile_size, level=level,
                                             slide_dataset=slide_dataset, key_dir=key_dir)


def save_slide_thumbnail_and_heatmap(results: Dict[ResultsKey, List[Any]], slide_id: str, tile_size: int, level: int,
                                     slide_dataset: SlidesDataset, key_dir: Path) -> None:
    slide_dict = mi.first_true(slide_dataset, pred=lambda entry: entry[SlideKey.SLIDE_ID] == slide_id)
    slide_dict = load_image_dict(slide_dict, level=level, margin=0)  # type: ignore
    slide_image = slide_dict[SlideKey.IMAGE]
    location_bbox = slide_dict[SlideKey.LOCATION]

    fig = plot_slide(slide_image=slide_image, scale=1.0)
    save_figure(fig=fig, figpath=key_dir / f'{slide_id}_thumbnail.png')

    fig = plot_heatmap_overlay(slide=slide_id, slide_image=slide_image, results=results,
                               location_bbox=location_bbox, tile_size=tile_size, level=level)
    save_figure(fig=fig, figpath=key_dir / f'{slide_id}_heatmap.png')


def save_scores_histogram(results: Dict[ResultsKey, List[Any]], figures_dir: Path) -> None:
    print("Plotting histogram ...")
    fig = plot_scores_hist(results)
    save_figure(fig=fig, figpath=figures_dir / 'hist_scores.png')


def save_confusion_matrix(conf_matrix_metric: ConfusionMatrix, class_names: List[str], figures_dir: Path) -> None:
    print("Computing and saving confusion matrix...")
    cf_matrix = conf_matrix_metric.compute().cpu().numpy()
    #  We can't log tensors in the normal way - just print it to console
    print('test/confusion matrix:')
    print(cf_matrix)
    #  Save the normalized confusion matrix as a figure in outputs
    cf_matrix_n = cf_matrix / cf_matrix.sum(axis=1, keepdims=True)
    fig = plot_normalized_confusion_matrix(cm=cf_matrix_n, class_names=class_names)
    save_figure(fig=fig, figpath=figures_dir / 'normalized_confusion_matrix.png')


class DeepMILOutputsHandler:
    _BEST_EPOCH_KEY = 'best_epoch'
    _BEST_VALUE_KEY = 'best_value'
    _PRIMARY_METRIC_KEY = 'primary_metric'

    def __init__(self, outputs_root: Path, n_classes: int, tile_size: int, level: int,
                 slide_dataset: Optional[SlidesDataset], class_names: Optional[Sequence[str]],
                 primary_val_metric: MetricsKey, maximise: bool) -> None:
        self.outputs_root = outputs_root

        self.n_classes = n_classes
        self.tile_size = tile_size
        self.level = level
        self.slide_dataset = slide_dataset
        self.class_names = validate_class_names(class_names, self.n_classes)

        self.primary_val_metric = primary_val_metric
        self.maximise = maximise

        self._init_best_metric()

    @property
    def best_metric_file_path(self) -> Path:
        return self.outputs_root / "best_val_metric.yml"

    def _init_best_metric(self) -> None:
        if self.best_metric_file_path.exists():
            contents = YAML().load(self.best_metric_file_path)
            self._best_metric_epoch = contents[self._BEST_EPOCH_KEY]
            self._best_metric_value = contents[self._BEST_VALUE_KEY]
            if contents[self._PRIMARY_METRIC_KEY] != self.primary_val_metric:
                raise ValueError(f"Expected primary metric '{self.primary_val_metric}', but found "
                                 f"'{contents[self._PRIMARY_METRIC_KEY]}' in {self.best_metric_file_path}")
        else:
            self._best_metric_epoch = 0
            self._best_metric_value = float('-inf') if self.maximise else float('inf')

    def _save_best_metric(self) -> None:
        contents = {self._BEST_EPOCH_KEY: self._best_metric_epoch,
                    self._BEST_VALUE_KEY: self._best_metric_value,
                    self._PRIMARY_METRIC_KEY: self.primary_val_metric.value}
        YAML().dump(contents, self.best_metric_file_path)

    def should_save_validation_outputs(self, metrics_dict: Mapping[MetricsKey, Metric], epoch: int) -> bool:
        metric_value = float(metrics_dict[self.primary_val_metric].compute())

        if self.maximise:
            is_best = metric_value > self._best_metric_value
        else:
            is_best = metric_value < self._best_metric_value

        if is_best:
            self._best_metric_value = metric_value
            self._best_metric_epoch = epoch
            self._save_best_metric()

        return is_best

    @property
    def validation_outputs_dir(self) -> Path:
        return self.outputs_root / "val"

    @property
    def previous_validation_outputs_dir(self) -> Path:
        return self.validation_outputs_dir.with_name("val_old")

    @property
    def test_outputs_dir(self) -> Path:
        return self.outputs_root / "test"

    def _save_outputs(self, outputs: List[Dict[ResultsKey, Any]],
                      metrics_dict: Mapping[MetricsKey, Metric],
                      outputs_dir: Path) -> None:
        # outputs object consists of a list of dictionaries (of metadata and results, including encoded features)
        # It can be indexed as outputs[batch_idx][batch_key][bag_idx][tile_idx]
        # example of batch_key ResultsKey.SLIDE_ID_COL
        # for batch keys that contains multiple values for slides e.g. ResultsKey.BAG_ATTN_COL
        # outputs[batch_idx][batch_key][bag_idx][tile_idx]
        # contains the tile value
        # TODO: Synchronise this with checkpoint saving (e.g. on_save_checkpoint())
        results = collate_results(outputs)
        figures_dir = outputs_dir / "fig"

        outputs_dir.mkdir(exist_ok=True, parents=True)
        figures_dir.mkdir(exist_ok=True, parents=True)

        save_outputs_and_features(results, outputs_dir)

        print("Selecting tiles ...")
        selected_slide_ids = save_top_and_bottom_tiles(results, n_classes=self.n_classes, figures_dir=figures_dir)

        if self.slide_dataset is not None:
            save_slide_thumbnails_and_heatmaps(results, selected_slide_ids, tile_size=self.tile_size, level=self.level,
                                               slide_dataset=self.slide_dataset, figures_dir=figures_dir)

        save_scores_histogram(results, figures_dir=figures_dir)

        conf_matrix: ConfusionMatrix = metrics_dict[MetricsKey.CONF_MATRIX]  # type: ignore
        save_confusion_matrix(conf_matrix, class_names=self.class_names, figures_dir=figures_dir)

    def save_validation_outputs(self, outputs: List[Dict[ResultsKey, Any]],
                                metrics_dict: Mapping[MetricsKey, Metric], epoch: int) -> None:
        if self.should_save_validation_outputs(metrics_dict, epoch):
            # First move existing outputs to a temporary directory, to avoid mixing
            # outputs of different epochs in case writing fails halfway through
            if self.validation_outputs_dir.exists():
                self.validation_outputs_dir.replace(self.previous_validation_outputs_dir)

            self._save_outputs(outputs, metrics_dict, self.validation_outputs_dir)

            # Writing completed successfully; delete temporary back-up
            if self.previous_validation_outputs_dir.exists():
                shutil.rmtree(self.previous_validation_outputs_dir)

    def save_test_outputs(self, outputs: List[Dict[ResultsKey, Any]],
                          metrics_dict: Mapping[MetricsKey, Metric]) -> None:
        self._save_outputs(outputs, metrics_dict, self.test_outputs_dir)
