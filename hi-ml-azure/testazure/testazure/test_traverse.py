#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from dataclasses import dataclass

from health_azure.traverse import object_to_dict, object_to_yaml, yaml_to_dict


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    optimizer: str = "Adam"


@dataclass
class TransformConfig:
    blur_sigma: float = 0.1
    blur_p: float = 0.2
    nested: OptimizerConfig = OptimizerConfig()


@dataclass
class FullConfig:
    optimizer: OptimizerConfig = OptimizerConfig()
    transforms: TransformConfig = TransformConfig()


def test_traverse1() -> None:
    config = TransformConfig()
    d = object_to_dict(config)
    assert d == {"blur_sigma": 0.1, "blur_p": 0.2}


def test_traverse2() -> None:
    config = FullConfig()
    d = object_to_dict(config)
    assert d == {
        "optimizer": {"learning_rate": 1e-3, "optimizer": "Adam"},
        "transforms": {"blur_sigma": 0.1, "blur_p": 0.2}
    }


def test_to_yaml_rountrip() -> None:
    config = FullConfig()
    yaml = object_to_yaml(config)
    print(yaml)

    dict = yaml_to_dict(yaml)
    assert dict == object_to_dict(config)
