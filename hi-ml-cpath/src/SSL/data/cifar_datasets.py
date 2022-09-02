#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from torchvision.datasets import CIFAR10, CIFAR100

from health_cpath.datasets.dataset_return_index import DatasetWithReturnIndex


class HimlCifar10(DatasetWithReturnIndex, CIFAR10):
    """
    Wrapper class around torchvision CIFAR10 class to optionally return the
    index on top of the image and the label in __getitem__ as well as defining num_classes property.
    """

    @property
    def num_classes(self) -> int:
        return 10


class HimlCifar100(DatasetWithReturnIndex, CIFAR100):
    """
    Wrapper class around torchvision CIFAR100 class class to optionally return the
    index on top of the image and the label in __getitem__ as well as defining num_classes property.
    """

    @property
    def num_classes(self) -> int:
        return 100
