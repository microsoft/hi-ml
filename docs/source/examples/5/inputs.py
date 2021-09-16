#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import numpy as np
from sklearn import datasets


def main() -> None:
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    dataset = Path("dataset")
    dataset.mkdir(exist_ok=False)
    X_csv = dataset / "X.csv"
    np.savetxt(X_csv, X, delimiter=',')
    y_csv = dataset / "y.csv"
    np.savetxt(y_csv, y, delimiter=',')


if __name__ == "__main__":
    main()
