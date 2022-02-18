#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Iterator

from torch.nn.parameter import Parameter
from torch.optim import Adam, Optimizer, SGD
from torch.optim.rmsprop import RMSprop

from health_ml.deep_learning_config import OptimizerParams, OptimizerType


def create_optimizer(config: OptimizerParams, parameters: Iterator[Parameter]) -> Optimizer:
    # Select optimizer type
    if config.optimizer_type in [OptimizerType.Adam, OptimizerType.AMSGrad]:
        return Adam(parameters,
                    config.l_rate,
                    config.adam_betas,
                    config.opt_eps,
                    config.weight_decay,
                    amsgrad=config.optimizer_type == OptimizerType.AMSGrad)
    elif config.optimizer_type == OptimizerType.SGD:
        return SGD(parameters,
                   config.l_rate,
                   config.momentum,
                   weight_decay=config.weight_decay)
    elif config.optimizer_type == OptimizerType.RMSprop:
        return RMSprop(parameters,
                       config.l_rate,
                       config.rms_alpha,
                       config.opt_eps,
                       config.weight_decay,
                       config.momentum)
    else:
        raise NotImplementedError(f"Optimizer type {config.optimizer_type.value} is not implemented")
