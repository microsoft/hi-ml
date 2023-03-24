#  -------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  -------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt

from health_multimodal.common.visualization import _plot_bounding_boxes


def test_plot_bounding_boxes() -> None:
    _, ax = plt.subplots()
    bboxes = [
        (1, 1, 3, 3),
        (5, 2, 2, 4),
    ]
    initial_patches_count = len(ax.patches)
    _plot_bounding_boxes(ax, bboxes)
    final_patches_count = len(ax.patches)
    assert final_patches_count - initial_patches_count == len(bboxes)
