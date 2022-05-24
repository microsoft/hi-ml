import torch
import heapq
import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from math import ceil
from histopathology.utils.naming import ResultsKey, SlideKey

from histopathology.utils.output_utils import save_figure


class TileNode:
    def __init__(self, data: Tensor, attn: float, id: Optional[int], x: Optional[float], y: Optional[float]) -> None:
        self.data = data
        self.attn = attn
        self.id = id
        self.x = x
        self.y = y


class SlideNode:
    def __init__(self, prob_score: float, slide_id: str) -> None:
        self.slide_id = slide_id
        self.prob_score = prob_score
        self.top_tiles: List[TileNode] = []
        self.bottom_tiles: List[TileNode] = []

    def __lt__(self, other):
        return self.prob_score < other.prob_score

    def update_tiles(self, tiles: Tensor, att_scores: Tensor, k_tiles: int) -> None:
        k_tiles = min(k_tiles, len(att_scores))
        sorted_indices = torch.argsort(att_scores.squeeze(), descending=True)
        self.top_tiles = [TileNode(data=tiles[i], attn=att_scores[i]) for i in sorted_indices[:k_tiles]]
        self.bottom_tiles = [TileNode(data=tiles[i], attn=att_scores[i]) for i in sorted_indices[-k_tiles:]]

    def shallow_copy(self):
        return SlideNode(self.prob_score, self.slide_id)

    def plot_attention_tiles(self, top: bool, case: str, key_dir: Path, ncols: int = 5, size: Tuple = (10, 10)) -> None:

        tiles = self.top_tiles if top else self.bottom_tiles
        suffix = "top" if top else "bottom"
        nrows = int(ceil(len(tiles) / ncols))

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=size)
        fig.suptitle(f"{case}: {self.slide_id} P=%.2f" % abs(self.prob_score))

        for i, tile_node in enumerate(tiles):
            axs.ravel()[i].imshow(tile_node.cpu().item(), clim=(0, 255), cmap="gray")
            axs.ravel()[i].set_title("%.6f" % tile_node.attn.cpu().item())

        for i in range(len(axs.ravel())):
            axs.ravel()[i].set_axis_off()

        save_figure(fig=fig, figpath=key_dir / f"{self.slide_id}_{suffix}.png")
        return fig


class KTopBottomTilesHandler:
    def __init__(
        self, n_classes: int, figures_dir: Path, k_tiles: int = 10, k_slides: int = 10, ncols: int = 4
    ) -> None:
        self.n_classes = n_classes
        self.figures_dir = figures_dir
        self.k_tiles = k_tiles
        self.k_slides = k_slides
        self.ncols = ncols
        self.n_classes_to_select = n_classes if n_classes > 1 else 2
        self.top_slides_heaps: Dict[int, List[SlideNode]] = {class_id: [] for class_id in range(self.n_classes)}
        self.bottom_slides_heaps: Dict[int, List[SlideNode]] = {class_id: [] for class_id in range(self.n_classes)}

    def _update_slides_heap(
        self, heap: List[SlideNode], tiles: Tensor, att_scores: Tensor, slide_node: SlideNode
    ) -> None:
        heapq.heappush(heap, slide_node)
        if len(heap) == self.n_slides + 1:
            old_slide_node = heap[0]
            heapq.heappop(heap)
            if old_slide_node.slide_id != slide_node.slide_id:
                slide_node.update_tiles(tiles, att_scores, self.k_tiles)

    def update_top_bottom_slides_heaps(self, batch: Dict, results: dict) -> None:
        slide_ids = np.unique(batch[SlideKey.SLIDE_ID])  # to account for repetitions in tiles pipeline
        # TODO double check that order is preserved affter unique
        bag_labels = results[ResultsKey.TRUE_LABEL]
        probs_perclass = results[ResultsKey.CLASS_PROBS]
        for gt_label in bag_labels:
            probs_gt_label = probs_perclass[:, gt_label]
            for i, slide_id in enumerate(slide_ids):
                self._update_slides_heap(
                    heap=self.top_slides_heaps[gt_label],
                    tiles=batch[SlideKey.IMAGE][i],
                    att_scores=results[ResultsKey.BAG_ATTN][i],
                    slide_node=SlideNode(prob_score=probs_gt_label[i], slide_id=slide_id),
                )
                self._update_slides_heap(
                    heap=self.bottom_slides_heaps[gt_label],
                    tiles=batch[SlideKey.IMAGE][i],
                    att_scores=results[ResultsKey.BAG_ATTN][i],
                    slide_node=SlideNode(prob_score=-probs_gt_label[i], slide_id=slide_id),
                )

    def make_figure_dirs(self, case: str) -> Path:
        key_dir = self.figures_dir / case
        key_dir.mkdir(parents=True, exist_ok=True)
        return key_dir

    def save_top_and_bottom_tiles(self):
        for class_id in range(self.n_classes_to_select):

            for slide_node in self.top_slides_heaps[class_id]:
                case = "TN" if class_id == 0 else f"TP_{class_id}"
                key_dir = self.make_figure_dirs(case=case, class_id=class_id)
                slide_node.plot_attention_tiles(top=True, case=case, key_dir=key_dir, ncols=self.ncols)
                slide_node.plot_attention_tiles(top=False, case=case, key_dir=key_dir, ncols=self.ncols)

            for slide_node in self.bottom_slides_heaps[class_id]:
                case = "FP" if class_id == 0 else f"FN_{class_id}"
                key_dir = self.make_figure_dirs(case=case, class_id=class_id)
                slide_node.plot_attention_tiles(top=True, case=case, key_dir=key_dir, ncols=self.ncols)
                slide_node.plot_attention_tiles(top=False, case=case, key_dir=key_dir, ncols=self.ncols)
