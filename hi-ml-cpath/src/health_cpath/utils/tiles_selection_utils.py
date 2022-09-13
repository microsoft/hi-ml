#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch
import heapq

from torch import Tensor
from typing import Optional, List, Tuple, Dict, Union, Any
from collections import ChainMap

from health_cpath.utils.naming import ResultsKey, SlideKey, TileKey


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

    def __init__(
        self, slide_id: str, gt_prob_score: float, pred_prob_score: float, true_label: int, pred_label: int
    ) -> None:
        """
        :param gt_prob_score: The probability score assigned to ground truth label of slide node. This scalar defines
            the order of the slide_node among the nodes in the max/min heap.
        :param pred_prob_score: The probability score assigned to predicted label of slide node.
        :param slide_id: The slide id in the data cohort.
        :param true_label: The ground truth label of the slide node.
        :param pred_label: The label predicted by the model.
        """
        self.slide_id = slide_id
        self.gt_prob_score = gt_prob_score
        self.pred_prob_score = pred_prob_score
        self.true_label = true_label
        self.pred_label = pred_label
        self.top_tiles: List[TileNode] = []
        self.bottom_tiles: List[TileNode] = []

    def __lt__(self, other: "SlideNode") -> bool:
        return self.gt_prob_score < other.gt_prob_score

    def update_selected_tiles(self, tiles: Tensor, attn_scores: Tensor, num_top_tiles: int) -> None:
        """Update top and bottom k tiles values from a set of tiles and their assigned attention scores.

        :param tiles: A stack of tiles with shape (n_tiles, channels, height, width).
        :param attn_scores: A tensor of attention scores assigned by the deepmil model to tiles of shape (n_tiles, 1).
        :param num_top_tiles: The number of tiles to select as top and bottom tiles.
        """
        num_top_tiles = min(num_top_tiles, len(attn_scores))

        _, top_k_indices = torch.topk(attn_scores, k=num_top_tiles, largest=True, sorted=True)
        self.top_tiles = [TileNode(data=tiles[i], attn=attn_scores[i].item()) for i in top_k_indices]

        _, bottom_k_indices = torch.topk(attn_scores, k=num_top_tiles, largest=False, sorted=True)
        self.bottom_tiles = [TileNode(data=tiles[i], attn=attn_scores[i].item()) for i in bottom_k_indices]

    def _shallow_copy(self) -> "SlideNode":
        """Returns a shallow copy of the current slide node contaning only the slide_id and its probability score."""
        return SlideNode(self.slide_id, self.gt_prob_score, self.pred_prob_score, self.true_label, self.pred_label)


SlideOrTileKey = Union[SlideKey, TileKey]
SlideDict = Dict[int, List[SlideNode]]
TileDict = Dict[str, List[TileNode]]


class TilesSelector:
    """Class that manages selecting top and bottom tiles on the fly during validation or test of DeepMIL models."""

    def __init__(self, n_classes: int, num_slides: int = 10, num_tiles: int = 12) -> None:
        """
        :param n_classes: Number of MIL classes (set `n_classes=1` for binary).
        :param num_slides: Number of top and bottom slides to select to define top and bottom tiles based of
            prediction scores. Defaults to 10.
        :param num_tiles: Number of tiles to select as top and bottom tiles based on attn scores. Defaults to 12.
        """
        if num_slides > 0 and num_tiles == 0:
            raise ValueError(
                "You should use `num_top_tiles>0` to be able to select top and bottom tiles for `num_top_slides>0`. "
                "You can set `num_top_slides=0` to disable top and bottom tiles plotting."
            )

        self.n_classes = n_classes if n_classes > 1 else 2
        self.num_slides = num_slides
        self.num_tiles = num_tiles
        self.top_slides_heaps: SlideDict = self._initialise_slide_heaps()
        self.bottom_slides_heaps: SlideDict = self._initialise_slide_heaps()

    def _initialise_slide_heaps(self) -> SlideDict:
        return {class_id: [] for class_id in range(self.n_classes)}

    def _clear_cached_slides_heaps(self) -> None:
        self.top_slides_heaps = self._initialise_slide_heaps()
        self.bottom_slides_heaps = self._initialise_slide_heaps()

    def _reset_slides_heaps(self, new_top_slides_heaps: SlideDict, new_bottom_slides_heaps: SlideDict) -> None:
        self.top_slides_heaps = new_top_slides_heaps
        self.bottom_slides_heaps = new_bottom_slides_heaps

    def _update_label_slides(
        self, class_slides_heap: List[SlideNode], tiles: Tensor, attn_scores: Tensor, slide_node: SlideNode,
    ) -> None:
        """Update the selected slides of a given class label on the fly by updating the content of class_slides_heap.
        First, we push a shallow slide_node into the slides_heaps[gt_label]. The order in slides_heaps[gt_label] is
        defined by the slide_node.gt_prob_score that is positive in top_slides_heaps nodes and negative in
        bottom_slides_heaps nodes.
        Second, we check if we exceeded self.num_top_slides to be selected.
            If so, we update the slides_node top and bottom tiles only if it has been kept in the heap.
            Else, we haven't reached the heaps max_capacity so we save the top and bottom tiles.

        :param class_slides_heap: The class_slides_heap dict to be updated. Either self.top_slides_heaps or
            self.bottom_slides_heaps.
        :param tiles: Tiles of a given whole slide retrieved from the current validation or test batch.
            (n_tiles, channels, height, width)
        :param attn_scores: The tiles attention scores to determine top and bottom tiles. (n_tiles, )
        :param slide_node: A shallow version of slide_node that contains only slide_id and additional metadata.
        """
        heapq.heappush(class_slides_heap, slide_node)
        if len(class_slides_heap) == self.num_slides + 1:
            old_slide_node = heapq.heappop(class_slides_heap)
            if old_slide_node.slide_id != slide_node.slide_id:
                slide_node.update_selected_tiles(tiles, attn_scores, self.num_tiles)
        else:
            slide_node.update_selected_tiles(tiles, attn_scores, self.num_tiles)

    def update_slides_selection(self, batch: Dict[SlideOrTileKey, Any], results: Dict[ResultsKey, Any]) -> None:
        """Updates top and bottom slides heaps on the fly during validation or test.

        :param batch: The current validation/test batch that contains the slides info, corresponding tiles and
            additional metadata.
        :param results: Batch results that contain attention scores, probability scores and the true labels.
        """
        if self.num_slides > 0:
            slide_ids = batch[SlideKey.SLIDE_ID]
            if isinstance(slide_ids[0], list):
                slide_ids = [slide_id[0] for slide_id in slide_ids]  # to account for repetitions in tiles pipeline
            batch_size = len(batch[SlideKey.IMAGE])
            for i in range(batch_size):
                gt_label = results[ResultsKey.TRUE_LABEL][i].item()
                pred_label = results[ResultsKey.PRED_LABEL][i].item()
                gt_prob_score = results[ResultsKey.CLASS_PROBS][:, gt_label][i].item()
                pred_prob_score = results[ResultsKey.CLASS_PROBS][:, pred_label][i].item()
                tiles = batch[SlideKey.IMAGE][i]
                attn_scores = results[ResultsKey.BAG_ATTN][i].squeeze(0)
                if pred_label == gt_label:
                    self._update_label_slides(
                        class_slides_heap=self.top_slides_heaps[gt_label],
                        tiles=tiles,
                        attn_scores=attn_scores,
                        slide_node=SlideNode(
                            slide_id=slide_ids[i],
                            gt_prob_score=gt_prob_score,
                            pred_prob_score=pred_prob_score,
                            true_label=gt_label,
                            pred_label=pred_label,
                        ),
                    )
                elif pred_label != gt_label:
                    self._update_label_slides(
                        class_slides_heap=self.bottom_slides_heaps[gt_label],
                        tiles=tiles,
                        attn_scores=attn_scores,
                        slide_node=SlideNode(
                            slide_id=slide_ids[i],
                            # negative score for bottom slides to reverse order in max heap
                            gt_prob_score=-gt_prob_score,
                            pred_prob_score=pred_prob_score,
                            true_label=gt_label,
                            pred_label=pred_label,
                        ),
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
        slides_heaps: SlideDict = self._initialise_slide_heaps()
        for class_id in range(self.n_classes):
            for rank in range(world_size):
                for slide_node in slides_heaps_list[rank][class_id]:
                    heapq.heappush(slides_heaps[class_id], slide_node)
                    if len(slides_heaps[class_id]) == self.num_slides + 1:
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
        if torch.distributed.is_initialized() and self.num_slides > 0:
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
