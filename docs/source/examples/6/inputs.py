#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import numpy as np
from sklearn import datasets

from health_azure import submit_to_azure_if_needed


def main() -> None:
    run_info = submit_to_azure_if_needed(
        compute_cluster_name="lite-testing-ds2",
        default_datastore="himldatasets",
        output_datasets=["himl_sample6_input"])
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    dataset = run_info.output_datasets[0] or Path("dataset")
    dataset.mkdir(exist_ok=True)
    X_csv = dataset / "X.csv"
    np.savetxt(X_csv, X, delimiter=',')
    y_csv = dataset / "y.csv"
    np.savetxt(y_csv, y, delimiter=',')


if __name__ == "__main__":
    main()
