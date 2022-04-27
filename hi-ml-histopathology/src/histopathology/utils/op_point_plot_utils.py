#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from histopathology.utils.array_utils import interp_index_1d, sliced_search_2d
from histopathology.utils.conf_matrix import ConfusionMatrix


def create_grid_confusion_matrix(num_total: int, num_positives: int, grid_size: int) -> ConfusionMatrix:
    num_steps = grid_size * 1j  # complex step tells np.mgrid that this is the number of steps, not step size
    tp2d, pp2d = np.mgrid[:num_positives:num_steps, :num_total:num_steps]  # type: ignore
    fp2d = pp2d - tp2d
    cm2d = ConfusionMatrix(num_total=num_total,
                           num_positives=num_positives,
                           true_positives=tp2d,
                           false_positives=fp2d,
                           thresholds=fp2d,
                           _validate_args=False)
    return cm2d


def get_level_coordinates(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, levels: Sequence[float],
                          frac_x: Optional[float] = None, frac_y: Optional[float] = None,
                          ascending: Optional[bool] = None) -> List[Tuple[float, float]]:
    indices_ij = [sliced_search_2d(zz, level, frac_x, frac_y, ascending) for level in levels]

    locations_xy = [(interp_index_1d(j, xx[i, :]), interp_index_1d(i, yy[:, j]))
                    for i, j in indices_ij]
    return locations_xy


def annotate_levels(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, levels: Sequence[float], ax: Axes,
                    label_fmt: Any = None, frac_x: Optional[float] = None, frac_y: Optional[float] = None,
                    contour_kwargs: Optional[Dict[str, Any]] = None,
                    clabel_kwargs: Optional[Dict[str, Any]] = None) -> None:
    _contour_kwargs = dict(colors='0.7', linewidths=.5)
    if contour_kwargs:
        _contour_kwargs.update(contour_kwargs)
    contours = ax.contour(xx, yy, zz, levels=levels, **_contour_kwargs)

    label_locations = get_level_coordinates(xx, yy, zz, levels=levels, frac_x=frac_x, frac_y=frac_y, ascending=None)

    _clabel_kwargs = dict(fontsize='small')
    if clabel_kwargs:
        _clabel_kwargs.update(clabel_kwargs)
    ax.clabel(contours, manual=label_locations, fmt=label_fmt, **_clabel_kwargs)


def add_sens_spec_annotations(cm: ConfusionMatrix, levels: Sequence[float], ax: Axes) -> None:
    cm2d = create_grid_confusion_matrix(cm.num_total, cm.num_positives, grid_size=100)

    annotate_levels(cm2d.pred_positives, cm2d.true_positives, cm2d.sensitivity, levels=levels,
                    frac_x=0.4, label_fmt=lambda x: f"$SEN={100*x:.0f}\\%$", ax=ax)

    annotate_levels(cm2d.pred_positives, cm2d.true_positives, cm2d.specificity, levels=levels,
                    frac_y=0.6, label_fmt=lambda x: f"$SPC={100*x:.0f}\\%$", ax=ax)

    spc_boundary_contours = ax.contourf(cm2d.pred_positives, cm2d.true_positives, cm2d.specificity,
                                        hatches=['xxxx', '', 'xxxx'], colors='none', levels=[0, 1], extend='both')
    plt.setp(spc_boundary_contours.collections, edgecolor='0.7', linewidth=0)
