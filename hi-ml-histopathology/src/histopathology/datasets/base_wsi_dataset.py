from typing import Any, Dict, List, Generator
from pathlib import Path
import torch
import numpy as np
from cucim import CuImage
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from random import shuffle

import sys
himl_root = Path(__file__).resolve().parent.parent.parent.parent
folders_to_add = [Path("src")]
for folder in folders_to_add:
    if folder.is_dir():
        sys.path.insert(0, str(folder))

from histopathology.datasets.base_dataset import SlidesDataset, TilesDataset
from histopathology.utils.naming import SlideKey


def process_chunk(params):
    # Loads a strip of image data and then loops through the patches
    # adding those above the threshold to the returned list
    # Each cache_strips x rows, it will load the next strip
    start_loc_list = params[0]
    inp_file = params[1]
    patch_size = params[2]
    level = params[3]
    factor = params[4]
    border = params[5]
    x_width = params[6]
    res = []
    slide = CuImage(inp_file)##
    cache_strips = 4
    low_res_patch_size = int((patch_size + (factor - 1)) // factor)
    y_limit = 0

    for start_loc in start_loc_list:
        if start_loc[1] >= y_limit:
            y_offset = start_loc[1]
            strip = np.array(slide.read_region((border, y_offset), (x_width, low_res_patch_size * cache_strips), level))

            y_limit = y_offset + patch_size * cache_strips
            y_offset = 0
            y_last = start_loc[1]
        else:
            if start_loc[1] != y_last:
                y_last = start_loc[1]
                y_offset += low_res_patch_size

        x_offset = int(start_loc[0] // factor)

        region = strip[y_offset: y_offset+low_res_patch_size, x_offset: x_offset+low_res_patch_size]

        if region.flatten().var() > 120:
            res.append((start_loc[0] - border, start_loc[1] - border))

    return res


def filter_patches(patches) -> List:
    patch_size = 164
    out = []

    # clean up a little - only add a patch if it has at least one neighbour
    for patch in patches:
        num = 0
        if (patch[0]-patch_size,patch[1]-patch_size) in patches:
            num += 1

        if (patch[0]+patch_size,patch[1]-patch_size) in patches:
            num += 1

        if num == 2:
            out.append(patch)
        else:

            if (patch[0]+patch_size,patch[1]+patch_size) in patches:
                num += 1

            if num==2:
                out.append(patch)
            else:
                if (patch[0]-patch_size,patch[1]+patch_size) in patches:
                    if num == 1:
                        out.append(patch)

    return out


def generate_patch_list(image_file, level, patch_size=256, inner_patch_size=164, chunk_size=2048,
                        random_allocation=False) -> Dict[str, Any]:

    # open low res image for thresholding
    wsi = CuImage(image_file)
    border = (patch_size - inner_patch_size) // 2

    # get dimensions at inference resolution
    sizes = wsi.metadata["cucim"]["resolutions"]
    w = sizes["level_dimensions"][0][0]
    h = sizes["level_dimensions"][0][1]
    wt = sizes["level_dimensions"][level][0]
    factor = w / wt

    start_loc_data = [(sx, sy)
                      for sy in range(border, h - border, inner_patch_size)
                      for sx in range(border, w - border, inner_patch_size)
                      ]

    results_dict = {}

    if random_allocation:
        shuffle(start_loc_data)

    start = 0
    end = len(start_loc_data)

    start_loc_list_iter = [(start_loc_data[i: i + chunk_size], image_file, inner_patch_size, level, factor, border, wt) \
                           for i in range(start, end, chunk_size)
                           ]

    with ProcessPoolExecutor(max_workers=8) as executor:
        result_futures = {executor.submit(process_chunk, x): x for x in start_loc_list_iter}
        for future in as_completed(result_futures):
            res1 = future.result()
            if res1:
                for patch in res1:
                    results_dict[patch] = 1

    results_dict = filter_patches(results_dict)
    return results_dict


class WSIDataset(SlidesDataset, torch.utils.data.IterableDataset):
    def __init__(self, patch_size, level, to_nchw=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.patch_list = []
        slides_paths = self.dataset_df[SlidesDataset.IMAGE_COLUMN].apply(lambda x: str(self.root_dir / x))
        self.images_list = slides_paths.tolist()
        self.current_image = ""
        self.patch_size = patch_size
        self.level = level
        self.to_nchw = to_nchw
        for image in self.images:
            patches = generate_patch_list(image)
            self.patch_list.append(patches)

    def __iter__(self) -> Generator[Dict, str, Any]:
        worker_info = torch.utils.data.get_worker_info()
        for i, patches in enumerate(self.patch_list):
            per_worker = int(math.ceil((len(patches)) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(patches))
            self.current_image = self.images_list[i]
            wsi = CuImage(self.current_image)

            for j in range(iter_start, iter_end):
                patch = patches[j]
                img = torch.tensor(np.array(wsi.read_region(location=patch,
                                   size=(self.patch_size, self.patch_size),
                                   level=self.level))
                                   ).type(torch.float32)
                if img.shape == (self.patch_size, self.patch_size, 3):
                    # TODO add label, slide id and others metadata columns to dict
                    if self.to_nchw:
                        yield {TilesDataset.IMAGE_COLUMN: torch.moveaxis(img, -1, 0),
                               TilesDataset.TILE_X_COLUMN: patch[0],
                               TilesDataset.TILE_Y_COLUMN: patch[1],
                               SlideKey.IMAGE_PATH: self.current_image}
                    else:
                        yield {TilesDataset.IMAGE_COLUMN: img,
                               TilesDataset.TILE_X_COLUMN: patch[0],
                               TilesDataset.TILE_Y_COLUMN: patch[1],
                               SlideKey.IMAGE_PATH: self.current_image
                               }
                else:
                    raise ValueError(f"Unexpected image shape {img.shape}")

            # if worker_info is None:
            #     print("Workers = None")
            #     # num_workers=0
            #     for i, patch in enumerate(self.patch_list):
            #         self.current_image = self.image_list[i]
            #         wsi = CuImage(self.current_image)
            #         for item in patch:
            #             img = torch.tensor(np.array(wsi.read_region(location=item,
            #                   size=(256, 256), level=0))).type(torch.float32)
            #             # TODO add label, slide id and others metadata columns to dict
            #             if self.to_nchw:
            #                 yield {TilesDataset.IMAGE_COLUMN: torch.moveaxis(img, -1, 0),
            #                         TilesDataset.TILE_X_COLUMN: patch[0],
            #                         TilesDataset.TILE_Y_COLUMN: patch[1],
            #                         SlideKey.IMAGE_PATH: self.current_image}
            #             else:
            #                 yield {TilesDataset.IMAGE_COLUMN: img,
            #                         TilesDataset.TILE_X_COLUMN: patch[0],
            #                         TilesDataset.TILE_Y_COLUMN: patch[1],
            #                         SlideKey.IMAGE_PATH: self.current_image
            #                         }
            #         else:
            #             raise ValueError(f"Unexpected image shape {img.shape}")

    def patches(self) -> List:
        return self.patch_list

    def preload_batch(self):
        pass

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = WSIDataset(root=Path("/tmp/datasets/PANDA"), patch_size=224, level=2,)
    dataloader = DataLoader(dataset=dataset, batch_size=2, pin_memory=True, num_workers=2, drop_last=True)
    tile_count = 0

    for step, batch in enumerate(dataloader):
        if step < 2:
            coords = np.stack((np.array(batch[TilesDataset.TILE_X_COLUMN][0][:]).astype(int),
                               np.array(batch[TilesDataset.TILE_Y_COLUMN][1][:]).astype(int))).T
            tile_count += len(coords)
