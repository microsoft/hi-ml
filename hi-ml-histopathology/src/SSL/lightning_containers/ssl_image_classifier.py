#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import List

import param
from pytorch_lightning import Callback

from health_ml.lightning_container import LightningModuleWithOptimizer
from health_ml.utils.checkpoint_utils import get_best_checkpoint_path

from SSL.data.datamodules import HimlVisionDataModule
from SSL.lightning_containers.ssl_container import DataModuleTypes, SSLContainer
from SSL.utils import create_ssl_image_classifier


class SSLClassifierContainer(SSLContainer):
    """
    This module is used to train a linear classifier on top of a frozen (or not) encoder.

    If you are running on AML, you can specify the SSL training run id via the --pretraining_run_recovery_id flag. This
    will automatically download the checkpoints for you and take the latest one as the starting weights of your
    classifier.

    If you are running locally, you can specify the path to your SSL weights via the --local_ssl_weights_path parameter.

    See docs/self_supervised_models.md for more details.
    """
    freeze_encoder = param.Boolean(default=True, doc="Whether to freeze the pretrained encoder or not.")
    local_ssl_weights_path = param.ClassSelector(class_=Path, default=None, doc="Local path to SSL weights")

    def create_model(self) -> LightningModuleWithOptimizer:
        """
        This method must create the actual Lightning model that will be trained.
        """
        if self.local_ssl_weights_path is None:
            path_to_checkpoint = get_best_checkpoint_path(self.checkpoint_folder)
        else:
            path_to_checkpoint = self.local_ssl_weights_path
        data_module: DataModuleTypes = self.data_module
        assert isinstance(data_module, HimlVisionDataModule)
        model = create_ssl_image_classifier(num_classes=data_module.dataset_train.dataset.num_classes,
                                            pl_checkpoint_path=str(path_to_checkpoint),
                                            freeze_encoder=self.freeze_encoder,
                                            class_weights=data_module.class_weights)

        return model

    def get_data_module(self) -> DataModuleTypes:
        """
        Gets the data that is used for the training and validation steps.
        Here we use different data loader for training of linear head and training of SSL model.
        """
        if hasattr(self, "data_module"):
            return self.data_module
        self.data_module: DataModuleTypes = self._create_ssl_data_modules(is_ssl_encoder_module=False)
        if self.use_balanced_binary_loss_for_linear_head:
            self.data_module.class_weights = self.data_module.compute_class_weights()
        return self.data_module

    def get_callbacks(self) -> List[Callback]:
        # This class inherits from SSLContainer, where the get_callbacks method adds the online evaluator.
        # We don't need that for the classifier.
        return []
