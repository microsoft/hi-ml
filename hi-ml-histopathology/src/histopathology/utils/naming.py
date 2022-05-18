#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from enum import Enum


class SlideKey(str, Enum):
    SLIDE_ID = 'slide_id'
    IMAGE = 'image'
    IMAGE_PATH = 'image_path'
    MASK = 'mask'
    MASK_PATH = 'mask_path'
    LABEL = 'label'
    SPLIT = 'split'
    SCALE = 'scale'
    ORIGIN = 'origin'
    FOREGROUND_THRESHOLD = 'foreground_threshold'
    METADATA = 'metadata'
    LOCATION = 'location'


class TileKey(str, Enum):
    TILE_ID = 'tile_id'
    SLIDE_ID = 'slide_id'
    IMAGE = 'image'
    IMAGE_PATH = 'image_path'
    MASK = 'mask'
    MASK_PATH = 'mask_path'
    LABEL = 'label'
    SPLIT = 'split'
    TILE_X = 'tile_x'  # TODO: deprecate ambiguous x and y names
    TILE_Y = 'tile_y'
    TILE_LEFT = 'left'
    TILE_TOP = 'top'
    TILE_RIGHT = 'right'
    TILE_BOTTOM = 'bottom'
    OCCUPANCY = 'occupancy'
    FOREGROUND_THRESHOLD = 'foreground_threshold'
    SLIDE_METADATA = 'slide_metadata'
    NUM_DISCARDED = 'num_discarded'

    @staticmethod
    def from_slide_metadata_key(slide_metadata_key: str) -> str:
        return 'slide_' + slide_metadata_key


class ResultsKey(str, Enum):
    SLIDE_ID = 'slide_id'
    TILE_ID = 'tile_id'
    FEATURES = 'features'
    IMAGE_PATH = 'image_path'
    LOSS = 'loss'
    PROB = 'prob'
    CLASS_PROBS = 'prob_class'
    PRED_LABEL = 'pred_label'
    TRUE_LABEL = 'true_label'
    BAG_ATTN = 'bag_attn'
    TILE_X = "x"  # TODO: deprecate ambiguous x and y names
    TILE_Y = "y"
    TILE_LEFT = 'left'
    TILE_TOP = 'top'
    TILE_RIGHT = 'right'
    TILE_BOTTOM = 'bottom'


class MetricsKey(str, Enum):
    ACC = 'accuracy'
    ACC_MACRO = 'macro_accuracy'
    ACC_WEIGHTED = 'weighted_accuracy'
    CONF_MATRIX = 'confusion_matrix'
    AUROC = 'auroc'
    PRECISION = 'precision'
    RECALL = 'recall'
    F1 = 'f1score'


class ModelKey(str, Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
