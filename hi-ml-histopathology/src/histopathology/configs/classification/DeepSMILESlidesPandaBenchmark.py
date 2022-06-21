#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


from typing import Any, Dict, Callable, Union
from torch import optim
from monai.transforms import Compose, ScaleIntensityRanged, RandRotate90d, RandFlipd

from health_ml.networks.layers.attention_layers import (
    TransformerPooling,
    TransformerPoolingBenchmark
)

from histopathology.datasets.panda_dataset import PandaDataset
from histopathology.datamodules.panda_module_benchmark import PandaSlidesDataModuleBenchmark
from histopathology.models.encoders import (
    HistoSSLEncoder,
    ImageNetEncoder_Resnet50,
    ImageNetSimCLREncoder,
    SSLEncoder,
)
from histopathology.configs.classification.DeepSMILEPanda import DeepSMILESlidesPanda
from histopathology.models.deepmil import SlidesDeepMILModule
from histopathology.utils.naming import MetricsKey, ModelKey, SlideKey


class DeepSMILESlidesPandaBenchmark(DeepSMILESlidesPanda):
    """
    Configuration for PANDA experiments from Myronenko et al. 2021:
    (https://link.springer.com/chapter/10.1007/978-3-030-87237-3_32)
    `is_finetune` sets the fine-tuning mode. For fine-tuning,
    batch_size = 2 runs on 8 GPUs with
    ~ 6:24 min/epoch (train) and ~ 00:50 min/epoch (validation).
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
        if self.is_finetune:
            self.batch_size = 2
        super().setup()

    def get_transforms_dict(self, image_key: str) -> Dict[ModelKey, Union[Callable, None]]:
        # Use same transforms as demonstrated in
        # https://github.com/Project-MONAI/tutorials/blob/master/pathology/multiple_instance_learning/panda_mil_train_evaluate_pytorch_gpu.py
        transform_train = Compose([
            RandFlipd(keys=image_key, spatial_axis=0, prob=0.5),
            RandFlipd(keys=image_key, spatial_axis=1, prob=0.5),
            RandRotate90d(keys=image_key, prob=0.5),
            ScaleIntensityRanged(keys=image_key, a_min=0.0, a_max=255.0)
        ])
        transform_inf = Compose([
            ScaleIntensityRanged(keys=image_key, a_min=0.0, a_max=255.0)
        ])
        # in case the transformations for training contain augmentations, val and test transform will be different
        return {ModelKey.TRAIN: transform_train, ModelKey.VAL: transform_inf, ModelKey.TEST: transform_inf}

    def get_data_module(self) -> PandaSlidesDataModuleBenchmark:
        # Myronenko et al. 2021 uses 80-20 cross-val split and no hold-out test set
        # Hence, inherited `PandaSlidesDataModuleBenchmark` from `SlidesDataModule`
        return PandaSlidesDataModuleBenchmark(
            root_path=self.local_datasets[0],
            max_bag_size=self.max_bag_size,
            batch_size=self.batch_size,
            max_bag_size_inf=self.max_bag_size_inf,
            level=self.level,
            tile_size=self.tile_size,
            step=self.step,
            random_offset=self.random_offset,
            pad_full=self.pad_full,
            background_val=self.background_val,
            filter_mode=self.filter_mode,
            transforms_dict=self.get_transforms_dict(PandaDataset.IMAGE_COLUMN),
            crossval_count=self.crossval_count,
            crossval_index=self.crossval_index,
            dataloader_kwargs=self.get_dataloader_kwargs(),
        )

    def create_model(self) -> SlidesDeepMILModule:
        self.data_module = self.get_data_module()
        pooling_layer, num_features = self.get_pooling_layer()
        outputs_handler = self.get_outputs_handler()
        deepmil_module = PandaSlidesDeepMILModuleBenchmark(encoder=self.get_model_encoder(),
                                                           label_column=SlideKey.LABEL,
                                                           n_classes=self.data_module.train_dataset.N_CLASSES,
                                                           pooling_layer=pooling_layer,
                                                           num_features=num_features,
                                                           dropout_rate=self.dropout_rate,
                                                           class_weights=self.data_module.class_weights,
                                                           l_rate=self.l_rate,
                                                           weight_decay=self.weight_decay,
                                                           adam_betas=self.adam_betas,
                                                           is_finetune=self.is_finetune,
                                                           class_names=self.class_names,
                                                           outputs_handler=outputs_handler,
                                                           chunk_size=self.encoding_chunk_size,
                                                           n_epochs=self.max_epochs)
        outputs_handler.set_slides_dataset_for_plots_handlers(self.get_slides_dataset())
        outputs_handler.set_conf_matrix_for_plots_handlers(deepmil_module.get_metrics())
        return deepmil_module


class PandaSlidesDeepMILModuleBenchmark(SlidesDeepMILModule):
    """
    Myronenko et al. 2021 uses a cosine LR scheduler which needs to be defined in the PL module
    Hence, inherited `PandaSlidesDeepMILModuleBenchmark` from `SlidesDeepMILModule`
    """

    def __init__(self, n_epochs: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.n_epochs = n_epochs

    def configure_optimizers(self) -> Dict[str, Any]:           # type: ignore
        optimizer = optim.AdamW(self.parameters(), lr=self.l_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.n_epochs, eta_min=0)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class SlidesPandaImageNetMILBenchmark(DeepSMILESlidesPandaBenchmark):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetEncoder_Resnet50.__name__, **kwargs)


class SlidesPandaImageNetSimCLRMILBenchmark(DeepSMILESlidesPandaBenchmark):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class SlidesPandaSSLMILBenchmark(DeepSMILESlidesPandaBenchmark):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=SSLEncoder.__name__, **kwargs)


class SlidesPandaHistoSSLMILBenchmark(DeepSMILESlidesPandaBenchmark):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)
