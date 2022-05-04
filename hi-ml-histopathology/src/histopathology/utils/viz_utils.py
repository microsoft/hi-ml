#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from monai.data.dataset import Dataset
from monai.data.image_reader import WSIReader
from torch.utils.data import DataLoader

from health_ml.utils.type_annotations import TupleInt3
from histopathology.utils.naming import SlideKey


def load_image_dict(sample: dict, level: int, margin: int) -> Dict[SlideKey, Any]:
    """
    Load image from metadata dictionary
    :param sample: dict describing image metadata. Example:
        {'image_id': ['1ca999adbbc948e69783686e5b5414e4'],
        'image': ['/tmp/datasets/PANDA/train_images/1ca999adbbc948e69783686e5b5414e4.tiff'],
         'mask': ['/tmp/datasets/PANDA/train_label_masks/1ca999adbbc948e69783686e5b5414e4_mask.tiff'],
         'data_provider': ['karolinska'],
         'isup_grade': tensor([0]),
         'gleason_score': ['0+0']}
    :param level: level of resolution to be loaded
    :param margin: margin to be included
    :return: a dict containing the image data and metadata
    """
    from histopathology.datasets.panda_dataset import LoadPandaROId
    loader = LoadPandaROId(WSIReader('cuCIM'), level=level, margin=margin)
    img = loader(sample)
    return img


def plot_panda_data_sample(panda_dir: str, nsamples: int, ncols: int, level: int, margin: int,
                           title_key: str = 'data_provider') -> None:
    """
    :param panda_dir: path to the dataset, it's expected a file called "train.csv" exists at the path.
        Look at the PandaDataset for more detail
    :param nsamples: number of random samples to be visualized
    :param ncols: number of columns in the figure grid. Nrows is automatically inferred
    :param level: level of resolution to be loaded
    :param margin: margin to be included
    :param title_key: metadata key in image_dict used to label each subplot
    """
    from histopathology.datasets.panda_dataset import PandaDataset
    panda_dataset = Dataset(PandaDataset(root=panda_dir))[:nsamples]  # type: ignore
    loader = DataLoader(panda_dataset, batch_size=1)

    nrows = math.ceil(nsamples / ncols)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(9, 9))

    for dict_images, ax in zip(loader, axes.flat):
        slide_id = dict_images[SlideKey.SLIDE_ID]
        title = dict_images[SlideKey.METADATA][title_key]
        print(f">>> Slide {slide_id}")
        img = load_image_dict(dict_images, level=level, margin=margin)
        ax.imshow(img[SlideKey.IMAGE].transpose(1, 2, 0))
        ax.set_title(title)
    fig.tight_layout()


def add_text(image: Image, text: str, y: float = 0.9, color: TupleInt3 = (27, 77, 40), fontsize_step: int = 2):
    font_path = Path('/usr/share/fonts/dejavu/DejaVuSans.ttf')  # TODO: stop hard-coding this
    fontsize = 48  # TODO: stop hard-coding this
    draw = ImageDraw.Draw(image)
    image_size_x, image_size_y = image.size

    font = ImageFont.truetype(str(font_path), fontsize)
    text_size_x, text_size_y = draw.textsize(text, font=font)
    while text_size_x >= image_size_x:
        fontsize -= fontsize_step
        font = ImageFont.truetype(str(font_path), fontsize)
        text_size_x, text_size_y = draw.textsize(text, font=font)
    start_x = image_size_x // 2 - text_size_x // 2
    start_y = image_size_y * y - text_size_y // 2
    xy = start_x, start_y
    draw.text(xy, text, fill=color, font=font, align='center')
