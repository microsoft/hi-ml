#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import argparse
import pandas as pd
import numpy as np

from health_azure.utils import is_local_rank_zero
from health_ml.utils.common_utils import logging_to_stdout

GLEASON_SCORES = ["0+0", "4+4", "3+3", "4+3", "negative", "4+5", "3+4", "5+4", "5+5", "5+3", "3+5"]
ISUP_GRADES = [0, 4, 1, 3, 0, 5, 2, 5, 5, 4, 4]
DATA_PROVIDERS = ["karolinska", "radboud"]


def create_mock_dataframe(n_samples: int = 4) -> None:
    data: dict = {"image_id": [], "data_provider": [], "isup_grade": [], "gleason_score": []}
    for i in range(n_samples):
        rand_id = np.random.randint(0, 11)
        data["image_id"].append(f"{i}")
        data["data_provider"].append(np.random.choice(DATA_PROVIDERS, 1)[0])
        data["isup_grade"].append(ISUP_GRADES[rand_id])
        data["gleason_score"].append(GLEASON_SCORES[rand_id])
    df = pd.DataFrame(data=data)
    df.to_csv("pathmnist/dataset.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging_to_stdout("INFO" if is_local_rank_zero() else "ERROR")
    parser.add_argument("--n-samples", type=int, default=4)
    args = parser.parse_args()
    logging.info(f"Creating mock dataset csv for {args.n_samples} WSIs")
    create_mock_dataframe(args.n_samples)
