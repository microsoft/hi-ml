#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import os
import pandas as pd
import json
import logging
from health_azure.utils import is_local_rank_zero

from health_ml.utils.common_utils import logging_to_stdout


def dump_dataset_csv_from_wsi_ids(
    datalist_json: str, dataset_csv: str, dest_dataset_csv: str, subset: str, image_id: str = "slide_id"
) -> None:

    # get target wsi ids
    with open(datalist_json, "r") as f:
        slides = json.load(f)
    wsi_ids = [slide["image"].split(".")[0] for slide in slides[subset]]

    # select corresponding tiles from dataset
    df = pd.read_csv(dataset_csv)
    df.to_csv(dataset_csv, index=False)
    df.loc[df[image_id].isin(wsi_ids)].to_csv(dest_dataset_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    panda_root = "/tmp/datasets/PANDA_tiles/PANDA_tiles_20210926-135446/panda_tiles_level1_224"
    logging_to_stdout("INFO" if is_local_rank_zero() else "ERROR")
    parser.add_argument("--datalist-json", type=str, default="panda/sub_datalist.json")
    parser.add_argument("--dataset-csv", type=str, default=os.path.join(panda_root, "dataset.csv"))
    parser.add_argument("--dest-csv", type=str, default="panda/sub_train_tiles.csv")
    parser.add_argument("--subset", type=str, default="training")
    parser.add_argument(
        "--image-id", type=str, default="slide_id", help="use image_id for slides and slide_id for tiles"
    )
    args = parser.parse_args()
    logging.info(f"Selecting target WSI ids from {args.datalist_json} and writing them to {args.dest_csv}")
    dump_dataset_csv_from_wsi_ids(args.datalist_json, args.dataset_csv, args.dest_csv, args.subset, args.image_id)
