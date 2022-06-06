#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import torch
import heapq
import numpy as np
import matplotlib.pyplot as plt

from math import ceil
from torch import Tensor
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union, Any
from collections import ChainMap

from histopathology.utils.naming import ResultsKey, SlideKey, TileKey
from histopathology.utils.viz_utils import save_figure


class TileNode:
    """Data structure class for tile nodes used by `SlideNode` to store top and bottom tiles by attention scores."""

    def __init__(
        self,
        data: Tensor,
        attn: float,
        id: Optional[int] = None,
        left: Optional[float] = None,
        top: Optional[float] = None,
    ) -> None:
        """
        :param data: A tensor of tile data of shape (channels, height, width), e.g (3, 224, 224).
        :param attn: The attention score assigned to the tile.
        :param id: An optional tile id, defaults to None
        :param left: An optional tile coordinate at the LEFT (e.g. x_coor), defaults to None
        :param top: An optional tile coordinate at the TOP (e.g. y_coor), defaults to None
        """
        self.data = data.cpu()
        self.attn = attn
        self.id = id
        self.left = left
        self.top = top


class SlideNode:
    """Data structure class for slide nodes used by `TopBottomTilesHandler` to store top and bottom slides by
    probability score. """

    def __init__(self, slide_id: str, prob_score: float,) -> None:
        """
        :param prob_score: The probability score assigned to the slide node. This scalar defines the order of the
            slide_node among the nodes in the max/min heap.
        :param slide_id: The slide id in the data cohort.
        """
        self.slide_id = slide_id
        self.prob_score = prob_score
        self.top_tiles: List[TileNode] = []
        self.bottom_tiles: List[TileNode] = []

    def __lt__(self, other: "SlideNode") -> bool:
        return self.prob_score < other.prob_score

    def update_selected_tiles(self, tiles: Tensor, attn_scores: Tensor, num_top_tiles: int) -> None:
        """Update top and bottom k tiles values from a set of tiles and their assigned attention scores.

        :param tiles: A stack of tiles with shape (n_tiles, channels, height, width).
        :param attn_scores: A tensor of attention scores assigned by the deepmil model to tiles of shape (n_tiles, 1).
        :param num_top_tiles: The number of tiles to select as top and bottom tiles.
        """
        num_top_tiles = min(num_top_tiles, len(attn_scores))

        _, top_k_indices = torch.topk(attn_scores.squeeze(), k=num_top_tiles, largest=True, sorted=True)
        self.top_tiles = [TileNode(data=tiles[i], attn=attn_scores[i].item()) for i in top_k_indices]

        _, bottom_k_indices = torch.topk(attn_scores.squeeze(), k=num_top_tiles, largest=False, sorted=True)
        self.bottom_tiles = [TileNode(data=tiles[i], attn=attn_scores[i].item()) for i in bottom_k_indices]

    def _shallow_copy(self) -> "SlideNode":
        """Returns a shallow copy of the current slide node contaning only the slide_id and its probability score."""
        return SlideNode(self.slide_id, self.prob_score)

    def plot_attention_tiles(
        self, tile_nodes: List[TileNode], case: str, num_columns: int = 4, size: Tuple[int, int] = (10, 10)
    ) -> plt.Figure:
        """Plot and save top or bottom tiles figures with their attention scores.

        :param tile_nodes: A list of top or bottom tiles nodes.
        :param case: The report case (e.g., TP, FN, ...)
        :param num_columns: Number of columns to create the subfigures grid, defaults to 4
        :param size: The figure size , defaults to (10, 10)
        """
        num_rows = int(ceil(len(tile_nodes) / num_columns))
        assert (
            num_rows > 0
        ), "The number of selected top and bottom tiles is too low. Try debugging with a higher num_top_tiles and/or a "
        "higher number of batches."

        fig, axs = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=size)
        fig.suptitle(f"{case}: {self.slide_id} P={abs(self.prob_score):.2f}")

        for i, tile_node in enumerate(tile_nodes):
            axs.ravel()[i].imshow(np.transpose(tile_node.data.numpy(), (1, 2, 0)), clim=(0, 255), cmap="gray")
            axs.ravel()[i].set_title("%.6f" % tile_node.attn)

        for i in range(len(axs.ravel())):
            axs.ravel()[i].set_axis_off()
        return fig


SlideOrTileKey = Union[SlideKey, TileKey]
SlideDict = Dict[int, List[SlideNode]]
TileDict = Dict[str, List[TileNode]]


class TopBottomTilesHandler:
    """Class that manages selecting top and bottom tiles on the fly during validation or test of DeepMIL models."""

    def __init__(self, n_classes: int, num_top_slides: int = 10, num_top_tiles: int = 12, num_columns: int = 4) -> None:
        """
        :param n_classes: Number of MIL classes (set `n_classes=1` for binary).
        :param num_top_slides: Number of top and bottom slides to select to define top and bottom tiles based of
            prediction scores. Defaults to 10.
        :param num_top_tiles: Number of tiles to select as top and bottom tiles based on attn scores. Defaults to 12.
        :param num_columns: Number of columns to use to plot top and bottom tiles.
        """
        self.n_classes = n_classes if n_classes > 1 else 2
        self.num_top_slides = num_top_slides
        self.num_top_tiles = num_top_tiles
        self.num_columns = num_columns
        self.top_slides_heaps: SlideDict = {class_id: [] for class_id in range(self.n_classes)}
        self.bottom_slides_heaps: SlideDict = {class_id: [] for class_id in range(self.n_classes)}
        self.report_cases_slide_ids = self.init_report_cases()

    def init_report_cases(self) -> Dict[str, List[str]]:
        """ Initializes the report cases dictionary to store slide_ids per case.
        Possible cases are set such as class 0 is considered to be the negative class.

        :return: A dictionary that maps TN/FP TP_{i}/FN_{i}, i={1,..., self.n_classes+1} cases to an empty list to be
            filled with corresponding slide ids.
        """
        report_cases: Dict[str, List[str]] = {"TN": [], "FP": []}
        report_cases.update({f"TP_{class_id}": [] for class_id in range(1, self.n_classes)})
        report_cases.update({f"FN_{class_id}": [] for class_id in range(1, self.n_classes)})
        return report_cases

    def _reset_slides_heaps(self, new_top_slides_heaps: SlideDict, new_bottom_slides_heaps: SlideDict) -> None:
        self.top_slides_heaps = new_top_slides_heaps
        self.bottom_slides_heaps = new_bottom_slides_heaps

    def _update_label_slides(
        self, class_slides_heap: List[SlideNode], tiles: Tensor, attn_scores: Tensor, slide_node: SlideNode,
    ) -> None:
        """Update the selected slides of a given class label on the fly by updating the content of class_slides_heap.
        First, we push a shallow slide_node into the slides_heaps[gt_label]. The order in slides_heaps[gt_label] is
        defined by the slide_node.prob_score that is positive in top_slides_heaps nodes and negative in
        bottom_slides_heaps nodes.
        Second, we check if we exceeded self.num_top_slides to be selected.
            If so, we update the slides_node top and bottom tiles only if it has been kept in the heap.
            Else, we haven't reached the heaps max_capacity so we save the top and bottom tiles.

        :param class_slides_heap: The class_slides_heap dict to be updated. Either self.top_slides_heaps or
            self.bottom_slides_heaps.
        :param tiles: Tiles of a given whole slide retrieved from the current validation or test batch.
            (n_tiles, channels, height, width)
        :param attn_scores: The tiles attention scores to determine top and bottom tiles. (n_tiles, )
        :param slide_node: A shallow version of slide_node that contains only slide_id and its assigned prob_score.
        """
        heapq.heappush(class_slides_heap, slide_node)
        if len(class_slides_heap) == self.num_top_slides + 1:
            old_slide_node = heapq.heappop(class_slides_heap)
            if old_slide_node.slide_id != slide_node.slide_id:
                slide_node.update_selected_tiles(tiles, attn_scores, self.num_top_tiles)
        else:
            slide_node.update_selected_tiles(tiles, attn_scores, self.num_top_tiles)

    def update_slides_selection(self, batch: Dict[SlideOrTileKey, Any], results: Dict[ResultsKey, Any]) -> None:
        """Updates top and bottom slides heaps on the fly during validation or test.

        :param batch: The current validation/test batch that contains the slides info, corresponding tiles and
            additional metadata.
        :param results: Batch results that contain attention scores, probability scores and the true labels.
        """
        slide_ids = batch[SlideKey.SLIDE_ID]
        if isinstance(slide_ids[0], list):
            slide_ids = [slide_id[0] for slide_id in slide_ids]  # to account for repetitions in tiles pipeline
        batch_size = len(batch[SlideKey.IMAGE])
        for i in range(batch_size):
            label = results[ResultsKey.TRUE_LABEL][i].item()
            probs_gt_label = results[ResultsKey.CLASS_PROBS][:, label]
            self._update_label_slides(
                class_slides_heap=self.top_slides_heaps[label],
                tiles=batch[SlideKey.IMAGE][i],
                attn_scores=results[ResultsKey.BAG_ATTN][i].squeeze(),
                slide_node=SlideNode(slide_id=slide_ids[i], prob_score=probs_gt_label[i].item()),
            )
            self._update_label_slides(
                class_slides_heap=self.bottom_slides_heaps[label],
                tiles=batch[SlideKey.IMAGE][i],
                attn_scores=results[ResultsKey.BAG_ATTN][i].squeeze(),
                slide_node=SlideNode(slide_id=slide_ids[i], prob_score=-probs_gt_label[i].item()),
            )

    def _shallow_copy_slides_heaps(self, slides_heaps: SlideDict) -> SlideDict:
        """Returns a shallow copy of slides heaps to be synchronised across devices.

        :param slides_heaps: The slides_heaps dict to be shallow copied. Either self.top_slides_heaps or
            self.bottom_slides_heaps.
        """
        shallow_slides_heaps_copy: SlideDict = {}
        for class_id, slide_nodes in slides_heaps.items():
            shallow_slides_heaps_copy[class_id] = [slide_node._shallow_copy() for slide_node in slide_nodes]
        return shallow_slides_heaps_copy

    def _reduce_slides_heaps_list(self, world_size: int, slides_heaps_list: List[SlideDict]) -> SlideDict:
        """Reduces the list of slides_heaps gathered across devices into a single slides_heaps that contain top or
        bottom shallow slide nodes.

        :param world_size: The number of devices in the ddp context.
        :param slides_heaps_list: A list of slides heaps gathered across devices.
        :return: A reduced version of slides_heaps_list to a single slides_heaps containing only top or bottom slides.
        """
        slides_heaps: SlideDict = {class_id: [] for class_id in range(self.n_classes)}
        for class_id in range(self.n_classes):
            for rank in range(world_size):
                for slide_node in slides_heaps_list[rank][class_id]:
                    heapq.heappush(slides_heaps[class_id], slide_node)
                    if len(slides_heaps[class_id]) == self.num_top_slides + 1:
                        heapq.heappop(slides_heaps[class_id])
        return slides_heaps

    @staticmethod
    def _gather_dictionaries(world_size: int, dicts: Dict, return_list: bool = False) -> Union[List[Dict], Dict]:
        """Gathers python dictionaries accross devices.

        :param world_size: The number of devices in the ddp context.
        :param dicts: The dictionaries to be gathered across devices.
        :param return_list: Flag to return the gathered dictionaries as a list, defaults to False
        :return: Either a list of dictionaries or a reduced version of this list as a single dictionary.
        """
        dicts_list = [None] * world_size  # type: ignore
        torch.distributed.all_gather_object(dicts_list, dicts)
        if return_list:
            return dicts_list  # type: ignore
        return dict(ChainMap(*dicts_list))  # type: ignore

    def _aggregate_shallow_slides_heaps(self, world_size: int, shallow_slides_heaps: SlideDict) -> SlideDict:
        """Gathers shallow slides heaps across devices and reduces the gathered slides_heaps_list into the new select
        slides across devices.

        :param world_size: The number of devices in the ddp context.
        :param shallow_slides_heaps: Reference to shallow slides heaps to be gathered across devices.
        :return: A reduced slides_heaps conatining only the best slide nodes across all devices.
        """
        slides_heaps_list: List[SlideDict] = self._gather_dictionaries(  # type: ignore
            world_size, shallow_slides_heaps, return_list=True
        )
        return self._reduce_slides_heaps_list(world_size=world_size, slides_heaps_list=slides_heaps_list)

    def _collect_tiles_for_selected_slides_on_device(
        self, new_slides_heaps: SlideDict, slides_heaps: SlideDict
    ) -> Tuple[TileDict, TileDict]:
        """Select top and bottom tiles from top or bottom slides_heaps in current device.

        :param new_slides_heaps: Selected top or bottom slides_heaps that contains shallow slide nodes.
        :param slides_heaps: The slides_heaps dict from where to select tiles. Either self.top_slides_heaps or
            self.bottom_slides_heaps.
        :return: Two dictionaries containing slide_ids and their corresponding top and bottom tiles.
        """
        top_tiles = {}
        bottom_tiles = {}
        for class_id in range(self.n_classes):
            for new_slide_node in new_slides_heaps[class_id]:
                for slide_node in slides_heaps[class_id]:
                    if slide_node.slide_id == new_slide_node.slide_id:
                        top_tiles[new_slide_node.slide_id] = slide_node.top_tiles
                        bottom_tiles[new_slide_node.slide_id] = slide_node.bottom_tiles
                        break
        return top_tiles, bottom_tiles

    def _update_shallow_slides_heaps_with_top_bottom_tiles(
        self, slides_heaps: SlideDict, top_tiles: TileDict, bottom_tiles: TileDict,
    ) -> None:
        """Update shallow version of slides heaps with top and bottom tiles gathered across devices.

        :param slides_heaps: The slides_heaps to update. Either self.top_slides_heaps or self.bottom_slides_heaps.
        :param top_tiles: Top tiles dictionary with slide_ids as keys.
        :param bottom_tiles: Bottom tiles dictionary with slide_ids as keys.
        """
        for class_id in range(self.n_classes):
            for slide_node in slides_heaps[class_id]:
                slide_node.top_tiles = top_tiles[slide_node.slide_id]
                slide_node.bottom_tiles = bottom_tiles[slide_node.slide_id]

    def gather_selected_tiles_across_devices(self) -> None:
        """Gathers top and bottom tiles across devices in ddp context. For each of top and bottom slides heaps:
            1- make a shallow copy of top and bottom slides_heaps
            2- gather best shallow slides across gpus
            3- select top and bottom tiles available in each device
            4- gather these tiles across devices
            5- update the synchronized shallow slide nodes across devices with their top and bottom tiles
        """
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            if world_size > 1:

                shallow_top_slides_heaps = self._shallow_copy_slides_heaps(slides_heaps=self.top_slides_heaps)
                shallow_bottom_slides_heaps = self._shallow_copy_slides_heaps(slides_heaps=self.bottom_slides_heaps)

                agg_top_slides_heaps = self._aggregate_shallow_slides_heaps(world_size, shallow_top_slides_heaps)
                agg_bottom_slides_heaps = self._aggregate_shallow_slides_heaps(world_size, shallow_bottom_slides_heaps)

                top_slides_top_tiles, top_slides_bottom_tiles = self._collect_tiles_for_selected_slides_on_device(
                    new_slides_heaps=agg_top_slides_heaps, slides_heaps=self.top_slides_heaps
                )
                bot_slides_top_tiles, bot_slides_bottom_tiles = self._collect_tiles_for_selected_slides_on_device(
                    new_slides_heaps=agg_bottom_slides_heaps, slides_heaps=self.bottom_slides_heaps
                )

                self._reset_slides_heaps(
                    new_top_slides_heaps=agg_top_slides_heaps, new_bottom_slides_heaps=agg_bottom_slides_heaps
                )

                top_tiles: TileDict = self._gather_dictionaries(world_size, top_slides_top_tiles)  # type: ignore
                bottom_tiles: TileDict = self._gather_dictionaries(world_size, top_slides_bottom_tiles)  # type: ignore
                self._update_shallow_slides_heaps_with_top_bottom_tiles(self.top_slides_heaps, top_tiles, bottom_tiles)

                top_tiles: TileDict = self._gather_dictionaries(world_size, bot_slides_top_tiles)  # type: ignore
                bottom_tiles: TileDict = self._gather_dictionaries(world_size, bot_slides_bottom_tiles)  # type: ignore
                self._update_shallow_slides_heaps_with_top_bottom_tiles(
                    self.bottom_slides_heaps, top_tiles, bottom_tiles
                )

    def make_figure_dirs(self, case: str, figures_dir: Path) -> Path:
        """Create the figure directory"""
        case_dir = figures_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def plot_slide_node_attention_tiles(self, case: str, figures_dir: Path, slide_node: SlideNode) -> None:
        """Plots the top and bottom attention tiles of a given slide_node

        :param case: The report case (e.g., TP, FN, ...)
        :param figures_dir: The path to the directory where to save the attention tiles figure.
        :param slide_node: the slide_node for which we plot top and bottom tiles.
        """
        case_dir = self.make_figure_dirs(case=case, figures_dir=figures_dir)

        top_tiles_fig = slide_node.plot_attention_tiles(
            tile_nodes=slide_node.top_tiles, case=case, num_columns=self.num_columns
        )
        save_figure(fig=top_tiles_fig, figpath=case_dir / f"{slide_node.slide_id}_top.png")

        bottom_tiles_fig = slide_node.plot_attention_tiles(
            tile_nodes=slide_node.bottom_tiles, case=case, num_columns=self.num_columns
        )
        save_figure(fig=bottom_tiles_fig, figpath=case_dir / f"{slide_node.slide_id}_bottom.png")

        self.report_cases_slide_ids[case].append(slide_node.slide_id)

    def save_top_and_bottom_tiles(self, figures_dir: Path) -> None:
        """Save top and bottom tiles figures.

        :param figures_dir: The path to the directory where to save the attention tiles figure.
        """
        logging.info(f"Plotting {self.num_top_tiles} top and bottom tiles of {self.num_top_slides} top and slides...")
        for class_id in range(self.n_classes):
            for slide_node in self.top_slides_heaps[class_id]:
                case = "TN" if class_id == 0 else f"TP_{class_id}"
                self.plot_slide_node_attention_tiles(case, figures_dir, slide_node)

            for slide_node in self.bottom_slides_heaps[class_id]:
                case = "FP" if class_id == 0 else f"FN_{class_id}"
                self.plot_slide_node_attention_tiles(case, figures_dir, slide_node)
