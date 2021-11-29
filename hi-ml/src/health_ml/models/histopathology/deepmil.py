from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, argmax, mode, nn, no_grad, optim, round
from torchmetrics import Accuracy, AUROC, Precision, Recall, F1

import pandas as pd
import numpy as np

def _format_cuda_memory_stats() -> str:
    return (f"GPU {torch.cuda.current_device()} memory: "
            f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB allocated, "
            f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB reserved")


class DeepMILModule(LightningModule):
    """Base class for deep multiple-instance learning"""

    def __init__(self,
                 label_column: str,
                 n_classes: int,
                 encoder: Any,
                 pooling_layer: Callable[[int, int, int], nn.Module],
                 pool_hidden_dim: int = 128,
                 pool_out_dim: int = 1,
                 class_weights: Optional[Tensor] = None,
                 l_rate: float = 5e-4,
                 weight_decay: float = 1e-4,
                 adam_betas: Tuple[float, float] = (0.9, 0.99),
                 verbose: bool = False,
                 ) -> None:
        """
        :param label_column: Label key for input batch dictionary.
        :param n_classes: Number of output classes for MIL prediction.
        :param encoder: The tile encoder to use for feature extraction. If no encoding is needed,
        you should use `IdentityEncoder`.
        :param pooling_layer: Type of pooling to use in multi-instance aggregation. Should be a
        `torch.nn.Module` constructor accepting input, hidden, and output pooling `int` dimensions.
        :param pool_hidden_dim: Hidden dimension of pooling layer (default=128).
        :param pool_out_dim: Output dimension of pooling layer (default=1).
        :param class_weights: Tensor containing class weights (default=None).
        :param l_rate: Optimiser learning rate.
        :param weight_decay: Weight decay parameter for L2 regularisation.
        :param adam_betas: Beta parameters for Adam optimiser.
        :param verbose: if True statements about memory usage are output at each step
        """
        super().__init__()

        # Dataset specific attributes
        self.label_column = label_column
        self.n_classes = n_classes
        self.pool_hidden_dim = pool_hidden_dim
        self.pool_out_dim = pool_out_dim
        self.pooling_layer = pooling_layer
        self.class_weights = class_weights
        self.encoder = encoder
        self.num_encoding = self.encoder.num_encoding

        # Optimiser hyperparameters
        self.l_rate = l_rate
        self.weight_decay = weight_decay
        self.adam_betas = adam_betas

        self.save_hyperparameters()
        self.verbose = verbose

        self.aggregation_fn, self.num_pooling = self.get_pooling()
        self.classifier_fn = self.get_classifier()
        self.loss_fn = self.get_loss()
        self.activation_fn = self.get_activation()

        # Metrics Objects
        self.train_metrics = self.get_metrics()
        self.val_metrics = self.get_metrics()
        self.test_metrics = self.get_metrics()

    def get_pooling(self) -> Tuple[Callable, int]:
        pooling_layer = self.pooling_layer(self.num_encoding,
                                           self.pool_hidden_dim,
                                           self.pool_out_dim)
        num_features = self.num_encoding*self.pool_out_dim
        return pooling_layer, num_features

    def get_classifier(self) -> Callable:
        return nn.Linear(in_features=self.num_pooling,
                         out_features=self.n_classes)

    def get_loss(self) -> Callable:
        if self.n_classes > 1:
            return nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            pos_weight = None
            if self.class_weights is not None:
                pos_weight = Tensor([self.class_weights[1]/(self.class_weights[0]+1e-5)])
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def get_activation(self) -> Callable:
        if self.n_classes > 1:
            return nn.Softmax()
        else:
            return nn.Sigmoid()

    @staticmethod
    def get_bag_label(labels: Tensor) -> Tensor:
        # Get bag (batch) labels as majority vote
        bag_label = mode(labels).values
        return bag_label.view(1)

    def get_metrics(self) -> nn.ModuleDict:
        if self.n_classes > 1:
            return nn.ModuleDict({'accuracy': Accuracy(num_classes=self.n_classes, average='micro'),
                                  'macro_accuracy': Accuracy(num_classes=self.n_classes, average='macro'),
                                  'weighted_accuracy': Accuracy(num_classes=self.n_classes, average='weighted')})
        else:
            return nn.ModuleDict({'accuracy': Accuracy(),
                                   'auroc': AUROC(num_classes=self.n_classes),
                                   'precision': Precision(),
                                   'recall': Recall(),
                                   'f1score': F1()})

    def log_metrics(self,
                    stage: str) -> None:
        valid_stages = ['train', 'test', 'val']
        if stage not in valid_stages:
            raise Exception(f"Invalid stage. Chose one of {valid_stages}")
        for metric_name, metric_object in self.get_metrics_dict(stage).items():
            self.log(f'{stage}/{metric_name}', metric_object, on_epoch=True, on_step=False, logger=True, sync_dist=True)

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        with no_grad():
            H = self.encoder(images)                        # N X L x 1 x 1
        A, M = self.aggregation_fn(H)                       # A: K x N | M: K x L
        M = M.view(-1, self.num_encoding * self.pool_out_dim)
        Y_prob = self.classifier_fn(M)
        return Y_prob, A

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.l_rate, weight_decay=self.weight_decay,
                          betas=self.adam_betas)

    def get_metrics_dict(self, stage: str) -> nn.ModuleDict:
        return getattr(self, f'{stage}_metrics')

    def _shared_step(self, batch: Dict, batch_idx: int, stage: str) -> Dict[str, Tensor]:
        # The batch dict contains lists of tensors of different sizes, for all bags in the batch.
        # This means we can't stack them along a new axis without padding to the same length.
        # We could alternatively concatenate them, but this would require other changes (e.g. in
        # the attention layers) to correctly split the tensors by bag/slide ID.
        bag_labels_list = []
        bag_logits_list = []
        bag_attn_list = []
        for bag_idx in range(len(batch['label'])):
            images = batch['image'][bag_idx]
            labels = batch[self.label_column][bag_idx]
            bag_labels_list.append(self.get_bag_label(labels))
            logit, attn = self(images)
            bag_logits_list.append(logit.view(-1))
            bag_attn_list.append(attn)
        bag_logits = torch.stack(bag_logits_list)
        bag_labels = torch.stack(bag_labels_list).view(-1)

        if self.n_classes > 1:
            loss = self.loss_fn(bag_logits, bag_labels)
        else:
            loss = self.loss_fn(bag_logits.squeeze(1), bag_labels.float())

        probs = self.activation_fn(bag_logits)
        if self.n_classes > 1:
            preds = argmax(probs, dim=1)
        else:
            preds = round(probs)

        for metric_object in self.get_metrics_dict(stage).values():
            metric_object.update(preds.view(-1, 1), bag_labels.view(-1, 1))

        results = {'loss': loss, 'pred_label': preds, 'true_label': bag_labels, 'prob': probs, 'bag_attn': bag_attn_list}
        return results

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:  # type: ignore
        train_result = self._shared_step(batch, batch_idx, 'train')
        self.log('train/loss', train_result['loss'], on_epoch=True, on_step=True, logger=True, sync_dist=True)
        if self.verbose:
            print(f"After loading images batch {batch_idx} -", _format_cuda_memory_stats())
        self.log_metrics('train')
        return train_result['loss']

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:  # type: ignore
        val_result = self._shared_step(batch, batch_idx, 'val')
        self.log('val/loss', val_result['loss'], on_epoch=True, on_step=True, logger=True, sync_dist=True)
        self.log_metrics('val')
        return val_result['loss']

    def test_step(self, batch: Dict, batch_idx: int) -> Dict[str, Any]:  # type: ignore
        test_result = self._shared_step(batch, batch_idx, 'test')
        self.log('test/loss', test_result['loss'], on_epoch=True, on_step=True, logger=True, sync_dist=True)
        self.log_metrics('test')
        batch.update(test_result)
        list_slide_dicts = []
        list_encoded_features = []
        for bag_idx in range(len(batch['label'])):
            slide_dict = dict()
            for key in batch.keys():
                if key not in ['image', 'loss']:
                    slide_dict[key] = batch[key][bag_idx]
                if key == 'image':
                    batch_features = batch[key][bag_idx]
            list_slide_dicts.append(slide_dict)
            list_encoded_features.append(batch_features)
        outputs_dict = {'list_slide_dicts': list_slide_dicts, 'list_encoded_features': list_encoded_features, 'batch_loss': test_result['loss']}
        return outputs_dict

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:                                                  # type: ignore
        # outputs object consists of a list of dictionaries (of metadata and results) and a list of encoded features over batches
        # It can be indexed as outputs[batch_idx]['list_slide_dicts'][bag_idx][key][tile_idx]
        # and outputs[batch_idx]['list_encoded_features'][bag_idx][tile_idx]

        csv_filename = fixed_paths.repository_root_directory() / Path('outputs/test_output.csv')
        encoded_features_filename = fixed_paths.repository_root_directory() / Path('outputs/test_encoded_features.pickle')
        df = pd.DataFrame(columns=[str(key) for key in outputs[0]['list_slide_dicts'][0].keys()])
        df_list = []

        for output in outputs:
            list_slide_dicts = output['list_slide_dicts']
            list_encoded_features = output['list_encoded_features']

            # Collect the list of dictionaries in a pandas dataframe
            for slide_dict in list_slide_dicts:
                slide_dict = self.normalize_dict_for_df(slide_dict, use_gpu=False)
                df_list.append(pd.DataFrame.from_dict(slide_dict))

            # Collect all features in a list
            features_list = self.move_list_to_device(list_encoded_features, use_gpu=False)

        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(csv_filename, mode='w', header=True)
        torch.save(features_list, encoded_features_filename)

    @staticmethod
    def normalize_dict_for_df(dict_old: Dict[str, Any], use_gpu: bool) -> Dict:
        # slide-level dictionaries are processed by making value dimensions uniform and converting to numpy arrays.
        # these steps are required to convert the dictionary to pandas dataframe.
        device = 'cuda' if use_gpu else 'cpu'
        dict_new = dict()
        for key, value in dict_old.items():
            if isinstance(value, Tensor):
                value = value.squeeze(0).to(device).numpy()
                if value.ndim == 0:
                    bag_size = len(dict_old['label'])
                    value = np.full(bag_size, fill_value=value)
            dict_new[key] = value
        return dict_new

    @staticmethod
    def move_list_to_device(list_encoded_features: List, use_gpu: bool) -> List:
        # a list of features on cpu obtained from original list on gpu
        features_list = []
        device = 'cuda' if use_gpu else 'cpu'
        for feature in list_encoded_features:
            feature = feature.squeeze(0).to(device)
            features_list.append(feature)
        return features_list
