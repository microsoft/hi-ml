#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import more_itertools as mi
import numpy as np
import pandas as pd
import torch
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from torchmetrics.metric import Metric

from histopathology.datasets.base_dataset import SlidesDataset
from histopathology.models.deepmil import validate_class_names
from histopathology.utils.metrics_utils import (plot_attention_tiles, plot_heatmap_overlay,
                                                plot_normalized_confusion_matrix, plot_scores_hist, plot_slide,
                                                select_k_tiles)
from histopathology.utils.naming import MetricsKey, ResultsKey, SlideKey
from histopathology.utils.viz_utils import load_image_dict


def save_figure(fig: plt.figure, figpath: Path) -> None:
    fig.savefig(figpath, bbox_inches='tight')


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
    fig.close()

    fig = plot_heatmap_overlay(slide=slide_id, slide_image=slide_image, results=results,
                               location_bbox=location_bbox, tile_size=tile_size, level=level)
    save_figure(fig=fig, figpath=key_dir / f'{slide_id}_heatmap.png')
    fig.close()


def save_scores_histogram(results: Dict[ResultsKey, List[Any]], figures_dir: Path) -> None:
    print("Plotting histogram ...")
    fig = plot_scores_hist(results)
    save_figure(fig=fig, figpath=figures_dir / 'hist_scores.png')
    fig.close()


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
    fig.close()


class DeepMILOutputsHandler:
    def __init__(self, outputs_dir: Path, n_classes: int, tile_size: int, level: int,
                 slide_dataset: Optional[SlidesDataset], class_names: Optional[Sequence[str]]) -> None:
        super().__init__()
        self.outputs_dir = outputs_dir
        self.figures_dir = outputs_dir / "fig"

        self.n_classes = n_classes
        self.tile_size = tile_size
        self.level = level
        self.slide_dataset = slide_dataset
        self.class_names = validate_class_names(class_names, self.n_classes)

    def save_outputs(self, outputs: List[Dict[ResultsKey, Any]],
                     metrics_dict: Mapping[MetricsKey, Metric]) -> None:  # type: ignore
        print("... test_epoch_end() called")
        # outputs object consists of a list of dictionaries (of metadata and results, including encoded features)
        # It can be indexed as outputs[batch_idx][batch_key][bag_idx][tile_idx]
        # example of batch_key ResultsKey.SLIDE_ID_COL
        # for batch keys that contains multiple values for slides e.g. ResultsKey.BAG_ATTN_COL
        # outputs[batch_idx][batch_key][bag_idx][tile_idx]
        # contains the tile value
        # TODO: Ensure this works with multi-GPU (e.g. using @rank_zero_only and pl_module.all_gather())
        # TODO: Synchronise this with checkpoint saving (e.g. on_save_checkpoint())
        results = collate_results(outputs)

        save_outputs_and_features(results, self.outputs_dir)

        print("Selecting tiles ...")
        selected_slide_ids = save_top_and_bottom_tiles(results, n_classes=self.n_classes,
                                                       figures_dir=self.figures_dir)

        if self.slide_dataset is not None:
            save_slide_thumbnails_and_heatmaps(results, selected_slide_ids, tile_size=self.tile_size, level=self.level,
                                               slide_dataset=self.slide_dataset, figures_dir=self.figures_dir)

        save_scores_histogram(results, figures_dir=self.figures_dir)

        conf_matrix: ConfusionMatrix = metrics_dict[MetricsKey.CONF_MATRIX]  # type: ignore
        save_confusion_matrix(conf_matrix, class_names=self.class_names, figures_dir=self.figures_dir)
