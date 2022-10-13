#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import param
from pytorch_lightning import Callback, LightningModule
from yacs.config import CfgNode

from health_ml.lightning_container import LightningContainer

from SSL.data.cifar_datasets import HimlCifar10, HimlCifar100
from SSL.data.cxr_datasets import CheXpert, CovidDataset, NIHCXR, RSNAKaggleCXR
from SSL.data.datamodules import CombinedDataModule, HimlVisionDataModule
from SSL.data.transforms_utils import CIFARLinearHeadTransform, \
    CIFARTrainTransform, \
    get_ssl_transforms_from_config
from SSL.encoders import get_encoder_output_dim
from SSL.lightning_modules.byol.byol_module import BootstrapYourOwnLatent
from SSL.lightning_modules.simclr_module import SimClrHiml
from SSL.lightning_modules.ssl_online_evaluator import SslOnlineEvaluatorHiml
from SSL.utils import SSLDataModuleType, SSLTrainingType, load_yaml_augmentation_config


@dataclass
class DataModuleArgs:
    augmentation_params: Optional[CfgNode]
    dataset_name: str
    dataset_path: Optional[Path]
    batch_size: int


class EncoderName(Enum):
    resnet18 = "resnet18"
    resnet50 = "resnet50"
    resnet101 = "resnet101"
    densenet121 = "densenet121"


class SSLDatasetName:
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    RSNAKaggleCXR = "RSNAKaggleCXR"
    NIHCXR = "NIHCXR"
    CheXpert = "CheXpert"
    Covid = "CovidDataset"


DataModuleTypes = Union[HimlVisionDataModule, CombinedDataModule]


class SSLContainer(LightningContainer):
    """
    This container is the based module to train an SSL model (either using BYOL or SimCLR).
    To have an overview of the parameters available for configuring this container please check out the documentation
    at docs/self_supervised_models.md.


    Note that this container is also used as the base class for SSLImageClassifier (finetuning container) as they share
    setup and datamodule methods.
    """
    DatasetToClassMapping = {SSLDatasetName.CIFAR10: HimlCifar10,
                             SSLDatasetName.CIFAR100: HimlCifar100,
                             SSLDatasetName.RSNAKaggleCXR: RSNAKaggleCXR,
                             SSLDatasetName.NIHCXR: NIHCXR,
                             SSLDatasetName.CheXpert: CheXpert,
                             SSLDatasetName.Covid: CovidDataset}

    ssl_augmentation_config = param.ClassSelector(class_=Path, allow_None=True,
                                                  doc="The path to the yaml config defining the parameters of the "
                                                      "augmentations. Ignored for CIFAR10 example")
    ssl_training_dataset_name: str = param.String(default="", doc="The name of the dataset")
    ssl_training_batch_size = param.Integer(
        doc="Training batch size per GPU. The effective batch size will be the number of GPUs times this number. "
            "For example, if you specify ssl_training_batch_size=100 and use 4 nodes with 4 gpus each, "
            "the effective batch size will be 1600.")
    ssl_training_type = param.ClassSelector(class_=SSLTrainingType, doc="Which algorithm to use for SSL training")
    ssl_encoder = param.ClassSelector(class_=EncoderName, doc="Which encoder to use for SSL")
    use_balanced_binary_loss_for_linear_head = param.Boolean(default=False,
                                                             doc="Whether to use a balanced loss for the training of "
                                                                 "the linear head")
    num_workers = param.Integer(default=4, doc="Number of workers to use for dataloader processes.")
    is_debug_model = param.Boolean(default=False,
                                   doc="If True, the training will be restricted to 1 batch per epoch."
                                       "Used for debugging and tests.")
    linear_head_augmentation_config = param.ClassSelector(class_=Path,
                                                          doc="The path to the yaml config for the linear head "
                                                              "augmentations")
    linear_head_dataset_name: str = param.String(default="",
                                                 doc="Name of the dataset to use for the linear head training")
    linear_head_batch_size = param.Integer(default=16, doc="Batch size for linear head tuning")
    learning_rate_linear_head_during_ssl_training = param.Number(default=1e-4,
                                                                 doc="Learning rate for linear head training during "
                                                                     "SSL training.")
    drop_last = param.Boolean(default=True, doc="If True drops the last incomplete batch")

    def setup(self) -> None:
        if self.is_debug_model:
            self.pl_limit_train_batches = 1
            self.pl_limit_val_batches = 1
        self.pl_find_unused_parameters = True
        self.total_num_gpus = self.num_gpus_per_node() * self.num_nodes
        self._load_config()
        # TODO: below is messy. Add specific command line args for training dataset and linear head dataset?

        if len(self.local_datasets) == 0:
            linear_head_dataset_path = None
        # If using the same data for training and linear head, or is just finetuning the linear head, local_datasets
        # may contain only one dataset entry
        elif (
            (self.linear_head_dataset_name == self.ssl_training_dataset_name)  # noqa: W504
            or (not (self.ssl_training_dataset_name) and self.linear_head_dataset_name)
        ) and len(self.local_datasets) == 1:
            # self.extra_local_dataset_paths = [self.local_dataset]
            linear_head_dataset_path = self.local_datasets[0]

        # If using different data for training and linear head, the user must specify 2 datasets (the first is the
        # training dataset and the second is the linear head dataset)
        elif (self.linear_head_dataset_name != self.ssl_training_dataset_name) and len(self.local_datasets) == 2:
            linear_head_dataset_path = self.local_datasets[1]
        else:
            raise TypeError(f"If linear_head_dataset_name ({self.linear_head_dataset_name}) does not equal"
                            f"ssl_training_dataset_name ({self.ssl_training_dataset_name}), then local_datasets"
                            f"must be a list of two entries: the first representing the training dataset and "
                            f"the second representing the linear head dataset. Instead found: {self.local_datasets}")

        self.datamodule_args = {SSLDataModuleType.LINEAR_HEAD:
                                DataModuleArgs(augmentation_params=self.classifier_augmentation_params,
                                               dataset_name=self.linear_head_dataset_name,
                                               dataset_path=linear_head_dataset_path,
                                               batch_size=self.linear_head_batch_size)}
        if self.ssl_training_dataset_name is not None:
            # The first entry in local_datasets should be the training dataset path
            if len(self.local_datasets) > 0:
                training_dataset_path = self.local_datasets[0]
            else:
                training_dataset_path = None
            self.datamodule_args.update(
                {SSLDataModuleType.ENCODER: DataModuleArgs(augmentation_params=self.ssl_augmentation_params,
                                                           dataset_name=self.ssl_training_dataset_name,
                                                           dataset_path=training_dataset_path,
                                                           batch_size=self.ssl_training_batch_size)})
        self.data_module: DataModuleTypes = self.get_data_module()
        self.inference_on_val_set = False
        self.inference_on_test_set = False
        # if self.perform_cross_validation:
        #     raise NotImplementedError("Cross-validation logic is not implemented for this module.")

    def _load_config(self) -> None:
        # For Chest-XRay you need to specify the parameters of the augmentations via a config file.
        self.ssl_augmentation_params = load_yaml_augmentation_config(
            self.ssl_augmentation_config) if self.ssl_augmentation_config is not None \
            else None
        self.classifier_augmentation_params = load_yaml_augmentation_config(
            self.linear_head_augmentation_config) if self.linear_head_augmentation_config is not None else \
            self.ssl_augmentation_params

    def create_model(self) -> LightningModule:
        """
        This method must create the actual Lightning model that will be trained.
        """
        # For small images like CIFAR, if using a resnet encoder, switch the first conv layer to a 3x3 kernel instead
        # of a 7x7 conv layer.
        use_7x7_first_conv_in_resnet = False if self.ssl_training_dataset_name.startswith("CIFAR") else True

        # Rescale the learning rate linearly according to the number of available GPUs, as seen in:
        # https://arxiv.org/abs/1706.02677, to avoid a drop in performance.
        gpus_per_node = self.num_gpus_per_node()
        num_of_total_gpus = self.num_nodes * gpus_per_node
        if num_of_total_gpus > 1:
            l_rate: float = self.l_rate * num_of_total_gpus  # type: ignore
            logging.info(f"We found {num_of_total_gpus} GPUs, SSL encoder learning rate has been adjusted from\
                 {self.l_rate} to {l_rate}")  # type: ignore
            self.l_rate = l_rate

        if self.ssl_training_type == SSLTrainingType.SimCLR:
            model: LightningModule = SimClrHiml(encoder_name=self.ssl_encoder.value,
                                                dataset_name=self.ssl_training_dataset_name,
                                                use_7x7_first_conv_in_resnet=use_7x7_first_conv_in_resnet,
                                                num_samples=self.data_module.num_train_samples,
                                                batch_size=self.data_module.batch_size,
                                                gpus=gpus_per_node,
                                                num_nodes=self.num_nodes,
                                                learning_rate=self.l_rate,
                                                max_epochs=self.max_epochs)
            logging.info(f"LR scheduling is using train_iters_per_epoch = {model.train_iters_per_epoch}")
        elif self.ssl_training_type == SSLTrainingType.BYOL:
            model = BootstrapYourOwnLatent(encoder_name=self.ssl_encoder.value,
                                           num_samples=self.data_module.num_train_samples,
                                           batch_size=self.data_module.batch_size,
                                           learning_rate=self.l_rate,
                                           use_7x7_first_conv_in_resnet=use_7x7_first_conv_in_resnet,
                                           warmup_epochs=10,
                                           max_epochs=self.max_epochs)
        else:
            raise ValueError(
                f"Unknown value for ssl_training_type, should be {SSLTrainingType.SimCLR.value} or "
                f"{SSLTrainingType.BYOL.value}. "
                f"Found {self.ssl_training_type.value}")
        model.hparams.update({'ssl_type': self.ssl_training_type.value,
                              "num_classes": self.data_module.num_classes})
        self.encoder_output_dim = get_encoder_output_dim(model, self.data_module)
        return model

    def get_data_module(self) -> DataModuleTypes:
        """
        Gets the data that is used for the training and validation steps.
        Here we use different data loader for training of linear head and training of SSL model.
        """
        if hasattr(self, "data_module"):
            return self.data_module
        encoder_data_module = self._create_ssl_data_modules(is_ssl_encoder_module=True)
        linear_data_module = self._create_ssl_data_modules(is_ssl_encoder_module=False)
        return CombinedDataModule(encoder_data_module, linear_data_module,
                                  self.use_balanced_binary_loss_for_linear_head)

    def _create_ssl_data_modules(self, is_ssl_encoder_module: bool) -> HimlVisionDataModule:
        """
        Returns torch lightning data module for encoder or linear head

        :param is_ssl_encoder_module: whether to return the data module for SSL training or for linear head. If true,
        :return transforms with two views per sample (batch like (img_v1, img_v2, label)). If False, return only one
        view per sample but also return the index of the sample in the dataset (to make sure we don't use twice the same
        batch in one training epoch (batch like (index, img_v1, label), as classifier dataloader expected to be shorter
        than SSL training, hence CombinedDataloader might loop over data several times per epoch).
        """
        datamodule_args = self.datamodule_args[SSLDataModuleType.ENCODER] if is_ssl_encoder_module else \
            self.datamodule_args[SSLDataModuleType.LINEAR_HEAD]

        train_transforms, val_transforms = self._get_transforms(datamodule_args.augmentation_params,
                                                                datamodule_args.dataset_name,
                                                                is_ssl_encoder_module)
        batch_multiplier = self.total_num_gpus if self.total_num_gpus > 0 else 1
        effective_batch_size = datamodule_args.batch_size * batch_multiplier
        logging.info(f"Batch size per GPU: {datamodule_args.batch_size}")
        logging.info(f"Effective batch size on {batch_multiplier} GPUs: {effective_batch_size}")
        dm = HimlVisionDataModule(dataset_cls=self.DatasetToClassMapping[datamodule_args.dataset_name],
                                  return_index=not is_ssl_encoder_module,  # index is only needed for linear head
                                  train_transforms=train_transforms,
                                  val_split=0.1,
                                  val_transforms=val_transforms,
                                  data_dir=str(datamodule_args.dataset_path),
                                  batch_size=datamodule_args.batch_size,
                                  num_workers=self.num_workers,
                                  seed=self.random_seed,
                                  drop_last=self.drop_last)
        dm.prepare_data()
        dm.setup()
        return dm

    def _get_transforms(self, augmentation_config: Optional[CfgNode],
                        dataset_name: str,
                        is_ssl_encoder_module: bool) -> Tuple[Any, Any]:
        """
        Returns the transformation pipeline for training and validation.
        :param augmentation_config: optional yaml config defining strength of augmenentations. Ignored for CIFAR
        examples.
        :param dataset_name: name of the dataset, value has to be in SSLDatasetName, determines which transformation
        pipeline to return.
        :param is_ssl_encoder_module: if True the transformation pipeline will yield two versions of the image it is
        applied on and it applies the training transformations also at validation time. Note that if your
        transformation does not contain any randomness, the pipeline will return two identical copies. If False, it
        will return only one transformation.
        :return: training transformation pipeline and validation transformation pipeline.
        """
        if dataset_name in [SSLDatasetName.RSNAKaggleCXR,
                            SSLDatasetName.NIHCXR,
                            SSLDatasetName.CheXpert,
                            SSLDatasetName.Covid]:
            assert augmentation_config is not None
            train_transforms, val_transforms = get_ssl_transforms_from_config(
                augmentation_config,
                return_two_views_per_sample=is_ssl_encoder_module,
                use_training_augmentations_for_validation=is_ssl_encoder_module
            )
        elif dataset_name in [SSLDatasetName.CIFAR10, SSLDatasetName.CIFAR100]:
            train_transforms = \
                CIFARTrainTransform(32) if is_ssl_encoder_module else CIFARLinearHeadTransform(32)
            val_transforms = \
                CIFARTrainTransform(32) if is_ssl_encoder_module else CIFARLinearHeadTransform(32)
        elif augmentation_config:
            train_transforms, val_transforms = get_ssl_transforms_from_config(
                augmentation_config,
                return_two_views_per_sample=is_ssl_encoder_module,
                use_training_augmentations_for_validation=is_ssl_encoder_module,
                expand_channels=False,
            )
            logging.warning(f"Dataset {dataset_name} unknown. The config will be consumed by "
                            f"get_ssl_transforms() to create the augmentation pipeline, make sure "
                            f"the transformations in your configs are compatible. ")
        else:
            raise ValueError(f"Dataset {dataset_name} unknown and no config has been passed.")

        return train_transforms, val_transforms

    def get_callbacks(self) -> List[Callback]:
        self.online_eval = SslOnlineEvaluatorHiml(class_weights=self.data_module.class_weights,  # type: ignore
                                                  z_dim=self.encoder_output_dim,
                                                  num_classes=self.data_module.num_classes,  # type: ignore
                                                  dataset=self.linear_head_dataset_name,  # type: ignore
                                                  drop_p=0.2,
                                                  learning_rate=self.learning_rate_linear_head_during_ssl_training)
        return [self.online_eval]
