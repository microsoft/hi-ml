from typing import Any, Optional, Tuple, List
from yacs.config import CfgNode
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torchvision.transforms import ColorJitter, RandomHorizontalFlip, RandomGrayscale, RandomResizedCrop, Lambda,\
    RandomApply

from health_ml.utils.data_augmentations import GaussianBlur
from SSL.lightning_containers.ssl_container import SSLContainer
from SSL.data.transforms_utils import DualViewTransformWrapper
from SSL.data.transform_pipeline import ImageTransformationPipeline
from SSL.lightning_modules.ssl_online_evaluator import SslOnlineEvaluatorHiml


class HistoSSLContainer(SSLContainer):
    """
    Config to train SSL model on one of the histo datasets (e.g. PANDA, CRCk). The main reason to create a
    histo specific SSL class is to overwrite the augmentations that will be applied. Augmentation can be configured by
    using a configuration yml file or by specifying the set of transformations in the _get_transforms method.
    """

    def __init__(self, model_checkpoint_save_interval: int,
                 model_checkpoints_save_last_k: int,
                 model_monitor_metric: str,
                 model_monitor_mode: str,
                 **kwargs: Any) -> None:
        super().__init__(pl_find_unused_parameters=True, **kwargs)
        self.model_checkpoint_save_interval = model_checkpoint_save_interval
        self.model_checkpoints_save_last_k = model_checkpoints_save_last_k
        self.model_monitor_metric = model_monitor_metric
        self.model_monitor_mode = model_monitor_mode
        self.use_mixed_precision = True

    def _get_transforms(self, augmentation_config: Optional[CfgNode],
                        dataset_name: str, is_ssl_encoder_module: bool) -> Tuple[Any, Any]:
        if augmentation_config:
            return super()._get_transforms(augmentation_config, dataset_name, is_ssl_encoder_module)
        else:
            # is_ssl_encoder_module will be True for ssl training, False for linear head training
            train_transforms = self.get_transforms(apply_augmentations=True)
            val_transforms = self.get_transforms(apply_augmentations=is_ssl_encoder_module)

            if is_ssl_encoder_module:
                train_transforms = DualViewTransformWrapper(train_transforms)
                val_transforms = DualViewTransformWrapper(val_transforms)
        return train_transforms, val_transforms

    @staticmethod
    def get_transforms(apply_augmentations: bool) -> ImageTransformationPipeline:
        transforms: List[Any] = []
        if apply_augmentations:
            # SimClr augmentations
            transforms = [RandomResizedCrop(size=224),
                          RandomHorizontalFlip(p=0.5),
                          RandomApply([ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], 0.8),
                          RandomGrayscale(p=0.2),
                          GaussianBlur(int(224 * 0.1) + 1)]
        else:
            # TODO Are there some transformations that we want to apply anyway?
            # not sure it will work without, DualViewTransformWrapper will call
            # an empty list
            transforms += [Lambda(lambda x: x)]
        pipeline = ImageTransformationPipeline(transforms)
        return pipeline

    def get_callbacks(self) -> List[Callback]:
        self.online_eval = SslOnlineEvaluatorHiml(class_weights=self.data_module.class_weights,  # type: ignore
                                                  z_dim=self.encoder_output_dim,
                                                  num_classes=self.data_module.num_classes,  # type: ignore
                                                  dataset=self.linear_head_dataset_name.value,  # type: ignore
                                                  drop_p=0.2,
                                                  learning_rate=self.learning_rate_linear_head_during_ssl_training)

        # Save model, this in independent of the model recovery
        checkpoint_callback = ModelCheckpoint(dirpath=self.checkpoint_folder,
                                              filename='model_checkpoint{epoch}',
                                              monitor=self.model_monitor_metric,
                                              mode=self.model_monitor_mode,
                                              every_n_epochs=self.model_checkpoint_save_interval,
                                              save_top_k=self.model_checkpoints_save_last_k,
                                              save_last=True)
        return [self.online_eval, checkpoint_callback]
