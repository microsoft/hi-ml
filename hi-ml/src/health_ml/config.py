#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from enum import Enum, unique


@unique
class PaddingMode(Enum):
    """
    Supported padding modes for numpy and torch image padding.
    """
    #: Zero padding scheme.
    Zero = 'constant'
    #: Pads with the edge values of array.
    Edge = 'edge'
    #: Pads with the linear ramp between end_value and the array edge value.
    LinearRamp = "linear_ramp"
    #: Pads with the maximum value of all or part of the vector along each axis.
    Maximum = "maximum"
    #: Pads with the mean value of all or part of the vector along each axis.
    Mean = "mean"
    #: Pads with the median value of all or part of the vector along each axis.
    Median = "median"
    #: Pads with the minimum value of all or part of the vector along each axis.
    Minimum = "minimum"
    #: Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
    Reflect = "reflect"
    #: Pads with the reflection of the vector mirrored along the edge of the array.
    Symmetric = "symmetric"
    #: Pads with the wrap of the vector along the axis.
    #: The first values are used to pad the end and the end values are used to pad the beginning.
    Wrap = "wrap"
    #: No padding is performed
    NoPadding = "no_padding"


@unique
class PhotometricNormalizationMethod(Enum):
    """
    Contains the valid methods that can be used to perform photometric normalization of a medical image.
    """
    Unchanged = "None"
    SimpleNorm = "Simple Norm"
    MriWindow = "MRI Window"
    CtWindow = "CT Window"
    TrimmedNorm = "Trimmed Norm"
