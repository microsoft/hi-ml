#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


from typing import Any, Dict, Callable, Union
from torch import optim
from monai.transforms import Compose, ScaleIntensityRanged, RandRotate90d, RandFlipd
from health_cpath.configs.run_ids import innereye_ssl_checkpoint_binary
from health_azure.utils import create_from_matching_params
from health_cpath.models.transforms import MetaTensorToTensord
from health_cpath.preprocessing.loading import LoadingParams
from health_cpath.utils.wsi_utils import TilingParams
from health_ml.networks.layers.attention_layers import (
    TransformerPooling,
    TransformerPoolingBenchmark
)
from health_ml.utils.checkpoint_utils import CheckpointParser
from health_ml.deep_learning_config import OptimizerParams
from health_cpath.datasets.panda_dataset import PandaDataset
from health_cpath.datamodules.panda_module_benchmark import PandaSlidesDataModuleBenchmark
from health_cpath.models.encoders import (
    HistoSSLEncoder,
    ImageNetSimCLREncoder,
    Resnet50_NoPreproc,
    SSLEncoder,
)
from health_cpath.configs.classification.DeepSMILEPanda import DeepSMILESlidesPanda
from health_cpath.models.deepmil import DeepMILModule
from health_cpath.utils.deepmil_utils import ClassifierParams, EncoderParams, PoolingParams
from health_cpath.utils.naming import MetricsKey, ModelKey, SlideKey


class PandaSlidesDeepMILModuleBenchmark(DeepMILModule):
    """
    Myronenko et al. 2021 uses a cosine LR scheduler which needs to be defined in the PL module
    Hence, inherited `PandaSlidesDeepMILModuleBenchmark` from `DeepMILModule`
    """

    def __init__(self, n_epochs: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.n_epochs = n_epochs

    def configure_optimizers(self) -> Dict[str, Any]:           # type: ignore
        optimizer = optim.AdamW(self.parameters(), lr=self.optimizer_params.l_rate,
                                weight_decay=self.optimizer_params.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.n_epochs, eta_min=0)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class DeepSMILESlidesPandaBenchmark(DeepSMILESlidesPanda):
    """
    Configuration for PANDA experiments from Myronenko et al. 2021:
    (https://link.springer.com/chapter/10.1007/978-3-030-87237-3_32)
    `tune_encoder` sets the fine-tuning mode of the encoder. For fine-tuning, batch_size = 2 runs on 8 GPUs
    with ~ 6:24 min/epoch (train) and ~ 00:50 min/epoch (validation).
    """

    def __init__(self, **kwargs: Any) -> None:
        default_kwargs = dict(
            pool_type=TransformerPoolingBenchmark.__name__,
            num_transformer_pool_layers=4,
            num_transformer_pool_heads=8,
            pool_hidden_dim=2048,
            encoding_chunk_size=60,
            max_bag_size=56,
            batch_size=8,  # effective batch size = batch_size * num_GPUs
            batch_size_inf=8,
            max_epochs=50,
            l_rate=3e-4,
            weight_decay=0,
            primary_val_metric=MetricsKey.ACC)
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

    def setup(self) -> None:
        # Params specific to transformer pooling
        if self.pool_type in [TransformerPoolingBenchmark.__name__, TransformerPooling.__name__]:
            self.l_rate = 3e-5
            self.weight_decay = 0.1
        # Params specific to fine-tuning
        if self.tune_encoder:
            self.batch_size = 2
            self.batch_size_inf = 2
        super().setup()

    def get_transforms_dict(self, image_key: str) -> Dict[ModelKey, Union[Callable, None]]:
        # Use same transforms as demonstrated in
        # https://github.com/Project-MONAI/tutorials/blob/master/pathology/multiple_instance_learning/panda_mil_train_evaluate_pytorch_gpu.py
        transform_train = Compose([
            RandFlipd(keys=image_key, spatial_axis=0, prob=0.5),
            RandFlipd(keys=image_key, spatial_axis=1, prob=0.5),
            RandRotate90d(keys=image_key, prob=0.5),
            ScaleIntensityRanged(keys=image_key, a_min=0.0, a_max=255.0),
            MetaTensorToTensord(keys=image_key),  # rotate transforms add some metadata to affine matrix
        ])
        transform_inf = Compose([
            ScaleIntensityRanged(keys=image_key, a_min=0.0, a_max=255.0),
        ])
        return {ModelKey.TRAIN: transform_train, ModelKey.VAL: transform_inf, ModelKey.TEST: transform_inf}

    def get_data_module(self) -> PandaSlidesDataModuleBenchmark:  # type: ignore
        # Myronenko et al. 2021 uses 80-20 cross-val split and no hold-out test set
        # Hence, inherited `PandaSlidesDataModuleBenchmark` from `SlidesDataModule`
        return PandaSlidesDataModuleBenchmark(
            root_path=self.local_datasets[0],
            batch_size=self.batch_size,
            batch_size_inf=self.batch_size_inf,
            max_bag_size=self.max_bag_size,
            max_bag_size_inf=self.max_bag_size_inf,
            tiling_params=create_from_matching_params(self, TilingParams),
            loading_params=create_from_matching_params(self, LoadingParams),
            seed=self.get_effective_random_seed(),
            transforms_dict=self.get_transforms_dict(PandaDataset.IMAGE_COLUMN),
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=self.get_dataloader_kwargs(),
            pl_replace_sampler_ddp=self.pl_replace_sampler_ddp,
        )

    def create_model(self) -> DeepMILModule:
        self.data_module = self.get_data_module()
        outputs_handler = self.get_outputs_handler()
        deepmil_module = PandaSlidesDeepMILModuleBenchmark(
            n_epochs=self.max_epochs,
            label_column=SlideKey.LABEL,
            n_classes=self.data_module.train_dataset.n_classes,
            class_names=self.class_names,
            class_weights=self.data_module.class_weights,
            outputs_folder=self.outputs_folder,
            encoder_params=create_from_matching_params(self, EncoderParams),
            pooling_params=create_from_matching_params(self, PoolingParams),
            classifier_params=create_from_matching_params(self, ClassifierParams),
            optimizer_params=create_from_matching_params(self, OptimizerParams),
            outputs_handler=outputs_handler,
            analyse_loss=self.analyse_loss,
        )
        outputs_handler.set_slides_dataset_for_plots_handlers(self.get_slides_dataset())
        return deepmil_module


class SlidesPandaImageNetMILBenchmark(DeepSMILESlidesPandaBenchmark):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=Resnet50_NoPreproc.__name__, **kwargs)


class SlidesPandaImageNetSimCLRMILBenchmark(DeepSMILESlidesPandaBenchmark):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class SlidesPandaSSLMILBenchmark(DeepSMILESlidesPandaBenchmark):
    def __init__(self, **kwargs: Any) -> None:
        # If no SSL checkpoint is provided, use the default one
        self.ssl_checkpoint = self.ssl_checkpoint or CheckpointParser(innereye_ssl_checkpoint_binary)
        super().__init__(encoder_type=SSLEncoder.__name__, **kwargs)


class SlidesPandaHistoSSLMILBenchmark(DeepSMILESlidesPandaBenchmark):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)
