#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import shutil
import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Type

from torch import Tensor, argmax, nn, rand, randint, randn, round, stack, allclose
from torch.utils.data._utils.collate import default_collate

from health_ml.networks.layers.attention_layers import AttentionLayer
from health_cpath.configs.classification.BaseMIL import BaseMILTiles

from health_cpath.configs.classification.DeepSMILECrck import DeepSMILECrck
from health_cpath.configs.classification.DeepSMILEPanda import BaseDeepSMILEPanda, DeepSMILETilesPanda
from health_cpath.datamodules.base_module import HistoDataModule, TilesDataModule
from health_cpath.datasets.base_dataset import TilesDataset
from health_cpath.datasets.default_paths import PANDA_5X_TILES_DATASET_ID, TCGA_CRCK_DATASET_DIR
from health_cpath.models.deepmil import BaseDeepMILModule, TilesDeepMILModule
from health_cpath.models.encoders import IdentityEncoder, ImageNetEncoder, TileEncoder
from health_cpath.utils.deepmil_utils import EncoderParams, PoolingParams
from health_cpath.utils.naming import MetricsKey, ResultsKey
from testhisto.mocks.base_data_generator import MockHistoDataType
from testhisto.mocks.slides_generator import MockPandaSlidesGenerator, TilesPositioningType
from testhisto.mocks.tiles_generator import MockPandaTilesGenerator
from testhisto.mocks.container import MockDeepSMILETilesPanda, MockDeepSMILESlidesPanda
from health_ml.utils.common_utils import is_gpu_available

no_gpu = not is_gpu_available()


def get_supervised_imagenet_encoder_params() -> EncoderParams:
    return EncoderParams(encoder_type=ImageNetEncoder.__name__)


def get_attention_pooling_layer_params(pool_out_dim: int = 1) -> PoolingParams:
    return PoolingParams(pool_type=AttentionLayer.__name__, pool_out_dim=pool_out_dim, pool_hidden_dim=5)


def _test_lightningmodule(
    n_classes: int,
    batch_size: int,
    max_bag_size: int,
    pool_out_dim: int,
    dropout_rate: Optional[float],
) -> None:

    assert n_classes > 0

    module = TilesDeepMILModule(
        label_column="label",
        n_classes=n_classes,
        dropout_rate=dropout_rate,
        encoder_params=get_supervised_imagenet_encoder_params(),
        pooling_params=get_attention_pooling_layer_params(pool_out_dim)
    )

    bag_images = rand([batch_size, max_bag_size, *module.encoder.input_dim])
    bag_labels_list = []
    bag_logits_list = []
    bag_attn_list = []
    for bag in bag_images:
        if n_classes > 1:
            labels = randint(n_classes, size=(max_bag_size,))
        else:
            labels = randint(n_classes + 1, size=(max_bag_size,))
        bag_labels_list.append(module.get_bag_label(labels))
        logit, attn = module(bag)
        assert logit.shape == (1, n_classes)
        assert attn.shape == (pool_out_dim, max_bag_size)
        bag_logits_list.append(logit.view(-1))
        bag_attn_list.append(attn)

    bag_logits = stack(bag_logits_list)
    bag_labels = stack(bag_labels_list).view(-1)

    assert bag_logits.shape[0] == (batch_size)
    assert bag_labels.shape[0] == (batch_size)

    if module.n_classes > 1:
        loss = module.loss_fn(bag_logits, bag_labels)
    else:
        loss = module.loss_fn(bag_logits.squeeze(1), bag_labels.float())

    assert loss > 0
    assert loss.shape == ()

    probs = module.activation_fn(bag_logits)
    assert ((probs >= 0) & (probs <= 1)).all()

    if n_classes > 1:
        predlabels = argmax(probs, dim=1)
    else:
        predlabels = round(probs)

    predlabels = predlabels.view(-1, 1)
    assert predlabels.shape[0] == batch_size
    assert probs.shape == (batch_size, n_classes)

    bag_labels = bag_labels.view(-1, 1)
    if n_classes == 1:
        probs = probs.squeeze(dim=1)
    for metric_name, metric_object in module.train_metrics.items():
        if metric_name == MetricsKey.CONF_MATRIX:
            continue
        score = metric_object(probs, bag_labels.view(batch_size,))
        if metric_name == MetricsKey.COHENKAPPA:
            # A NaN value could result due to a division-by-zero error
            assert torch.all(score[~score.isnan()] >= -1)
            assert torch.all(score[~score.isnan()] <= 1)
        else:
            assert torch.all(score >= 0)
            assert torch.all(score <= 1)


@pytest.fixture(scope="session")
def mock_panda_tiles_root_dir(
    tmp_path_factory: pytest.TempPathFactory, tmp_path_to_pathmnist_dataset: Path
) -> Generator:
    tmp_root_dir = tmp_path_factory.mktemp("mock_tiles")
    tiles_generator = MockPandaTilesGenerator(
        dest_data_path=tmp_root_dir,
        src_data_path=tmp_path_to_pathmnist_dataset,
        mock_type=MockHistoDataType.PATHMNIST,
        n_tiles=4,
        n_slides=10,
        n_channels=3,
        tile_size=28,
        img_size=224,
    )
    logging.info("Generating temporary mock tiles that will be deleted at the end of the session.")
    tiles_generator.generate_mock_histo_data()
    yield tmp_root_dir
    shutil.rmtree(tmp_root_dir)


@pytest.fixture(scope="session")
def mock_panda_slides_root_dir(
    tmp_path_factory: pytest.TempPathFactory, tmp_path_to_pathmnist_dataset: Path
) -> Generator:
    tmp_root_dir = tmp_path_factory.mktemp("mock_slides")
    wsi_generator = MockPandaSlidesGenerator(
        dest_data_path=tmp_root_dir,
        src_data_path=tmp_path_to_pathmnist_dataset,
        mock_type=MockHistoDataType.PATHMNIST,
        n_tiles=4,
        n_slides=10,
        n_channels=3,
        n_levels=3,
        tile_size=28,
        background_val=255,
        tiles_pos_type=TilesPositioningType.RANDOM
    )
    logging.info("Generating temporary mock slides that will be deleted at the end of the session.")
    wsi_generator.generate_mock_histo_data()
    yield tmp_root_dir
    shutil.rmtree(tmp_root_dir)


@pytest.mark.parametrize("n_classes", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 15])
@pytest.mark.parametrize("max_bag_size", [1, 7])
@pytest.mark.parametrize("pool_out_dim", [1, 6])
@pytest.mark.parametrize("dropout_rate", [None, 0.5])
def test_lightningmodule_attention(
    n_classes: int,
    batch_size: int,
    max_bag_size: int,
    pool_out_dim: int,
    dropout_rate: Optional[float],
) -> None:
    _test_lightningmodule(n_classes=n_classes,
                          batch_size=batch_size,
                          max_bag_size=max_bag_size,
                          pool_out_dim=pool_out_dim,
                          dropout_rate=dropout_rate)


def validate_metric_inputs(scores: torch.Tensor, labels: torch.Tensor) -> None:
    def is_integral(x: torch.Tensor) -> bool:
        return (x == x.long()).all()  # type: ignore

    assert labels.shape == (scores.shape[0], )
    assert torch.is_floating_point(scores), "Received scores with integer dtype"
    assert not is_integral(scores), "Received scores with integral values"
    assert is_integral(labels), "Received labels with floating-point values"


def add_callback(fn: Callable, callback: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        callback(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


@pytest.mark.parametrize("n_classes", [1, 3])
def test_metrics(n_classes: int) -> None:
    input_dim = (128,)

    def _mock_get_encoder(  # type: ignore
        self, ssl_ckpt_run_id: Optional[str], outputs_folder: Optional[Path]
    ) -> TileEncoder:
        return IdentityEncoder(input_dim=input_dim)
    LABEL_COLUMN = "label"
    with patch("health_cpath.models.deepmil.EncoderParams.get_encoder", new=_mock_get_encoder):
        module = TilesDeepMILModule(label_column=LABEL_COLUMN,
                                    n_classes=n_classes,
                                    pooling_params=get_attention_pooling_layer_params(pool_out_dim=1))

        # Patching to enable running the module without a Trainer object
        module.trainer = MagicMock(world_size=1)  # type: ignore
        module.log = MagicMock()  # type: ignore
        module.outputs_handler = MagicMock()

        batch_size = 20
        bag_size = 5
        if n_classes > 1:
            class_weights = torch.rand(n_classes)
        else:
            class_weights = torch.tensor([0.8, 0.2])
        bags: List[Dict] = []
        for slide_idx in range(batch_size):
            bag_label = torch.multinomial(class_weights, 1)
            sample: Dict[str, Iterable] = {
                TilesDataset.SLIDE_ID_COLUMN: [str(slide_idx)] * bag_size,
                TilesDataset.TILE_ID_COLUMN: [f"{slide_idx}-{tile_idx}" for tile_idx in range(bag_size)],
                TilesDataset.IMAGE_COLUMN: rand(bag_size, *input_dim),
                LABEL_COLUMN: bag_label.expand(bag_size),
            }
            sample[TilesDataset.PATH_COLUMN] = [tile_id + '.png'
                                                for tile_id in sample[TilesDataset.TILE_ID_COLUMN]]
            bags.append(sample)
        batch = default_collate(bags)

        # ================
        # Test that the module metrics match manually computed metrics with the correct inputs
        module_metrics_dict = module.test_metrics
        independent_metrics_dict = module.get_metrics()

        # Patch the metrics to check that the inputs are valid. In particular, test that the scores
        # do not have integral values, which would suggest that hard labels were passed instead.
        for metric_obj in module_metrics_dict.values():
            metric_obj.update = add_callback(metric_obj.update, validate_metric_inputs)

        results = module.test_step(batch, 0)
        predicted_probs = results[ResultsKey.PROB]
        true_labels = results[ResultsKey.TRUE_LABEL]

        for key, metric_obj in module_metrics_dict.items():
            value = metric_obj.compute()
            expected_value = independent_metrics_dict[key](predicted_probs, true_labels.view(batch_size,))
            assert torch.allclose(value, expected_value), f"Discrepancy in '{key}' metric"

        assert all(key in results.keys() for key in [ResultsKey.SLIDE_ID, ResultsKey.TILE_ID, ResultsKey.IMAGE_PATH])


def move_batch_to_expected_device(batch: Dict[str, List], use_gpu: bool) -> Dict:
    device = "cuda" if use_gpu else "cpu"
    return {
        key: [
            value.to(device) if isinstance(value, Tensor) else value for value in values
        ]
        for key, values in batch.items()
    }


def assert_train_step(module: BaseDeepMILModule, data_module: HistoDataModule, use_gpu: bool) -> None:
    train_data_loader = data_module.train_dataloader()
    for batch_idx, batch in enumerate(train_data_loader):
        batch = move_batch_to_expected_device(batch, use_gpu)
        loss = module.training_step(batch, batch_idx)
        loss.retain_grad()
        loss.backward()
        assert loss.grad is not None
        assert loss.shape == (1, 1)
        assert isinstance(loss, Tensor)
        break


def assert_validation_step(module: BaseDeepMILModule, data_module: HistoDataModule, use_gpu: bool) -> None:
    val_data_loader = data_module.val_dataloader()
    for batch_idx, batch in enumerate(val_data_loader):
        batch = move_batch_to_expected_device(batch, use_gpu)
        outputs_dict = module.validation_step(batch, batch_idx)
        loss = outputs_dict[ResultsKey.LOSS]  # noqa
        assert loss.shape == (1, 1)  # noqa
        assert isinstance(loss, Tensor)
        break


def assert_test_step(module: BaseDeepMILModule, data_module: HistoDataModule, use_gpu: bool) -> None:
    test_data_loader = data_module.test_dataloader()
    for batch_idx, batch in enumerate(test_data_loader):
        batch = move_batch_to_expected_device(batch, use_gpu)
        outputs_dict = module.test_step(batch, batch_idx)
        loss = outputs_dict[ResultsKey.LOSS]  # noqa
        assert loss.shape == (1, 1)  # noqa
        assert isinstance(loss, Tensor)
        break


CONTAINER_DATASET_DIR = {
    DeepSMILETilesPanda: PANDA_5X_TILES_DATASET_ID,
    DeepSMILECrck: TCGA_CRCK_DATASET_DIR,
}


@pytest.mark.parametrize("container_type", [DeepSMILETilesPanda,
                                            DeepSMILECrck])
@pytest.mark.parametrize("use_gpu", [True, False])
def test_container(container_type: Type[BaseMILTiles], use_gpu: bool) -> None:
    dataset_dir = CONTAINER_DATASET_DIR[container_type]
    if not os.path.isdir(dataset_dir):
        pytest.skip(
            f"Dataset for container {container_type.__name__} "
            f"is unavailable: {dataset_dir}"
        )
    if container_type is DeepSMILECrck:
        container = container_type(encoder_type=ImageNetEncoder.__name__)
    elif container_type is DeepSMILETilesPanda:
        container = DeepSMILETilesPanda(encoder_type=ImageNetEncoder.__name__)
    else:
        container = container_type()

    container.setup()

    data_module: TilesDataModule = container.get_data_module()  # type: ignore
    data_module.max_bag_size = 10

    module = container.create_model()
    module.outputs_handler = MagicMock()
    module.trainer = MagicMock(world_size=1)  # type: ignore
    module.log = MagicMock()  # type: ignore
    if use_gpu:
        module.cuda()

    assert_train_step(module, data_module, use_gpu)
    assert_validation_step(module, data_module, use_gpu)
    assert_test_step(module, data_module, use_gpu)


def _test_mock_panda_container(use_gpu: bool, mock_container: BaseDeepSMILEPanda, tmp_path: Path) -> None:
    container = mock_container(tmp_path=tmp_path)
    container.setup()
    data_module = container.get_data_module()
    module = container.create_model()

    module.trainer = MagicMock(world_size=1)  # type: ignore
    module.outputs_handler = MagicMock()
    module.log = MagicMock()  # type: ignore
    if use_gpu:
        module.cuda()

    assert_train_step(module, data_module, use_gpu)
    assert_validation_step(module, data_module, use_gpu)
    assert_test_step(module, data_module, use_gpu)


def test_mock_tiles_panda_container_cpu(mock_panda_tiles_root_dir: Path) -> None:
    _test_mock_panda_container(use_gpu=False, mock_container=MockDeepSMILETilesPanda,
                               tmp_path=mock_panda_tiles_root_dir)


@pytest.mark.skipif(no_gpu, reason="Test requires GPU")
@pytest.mark.gpu
@pytest.mark.parametrize("mock_container, tmp_path", [(MockDeepSMILETilesPanda, "mock_panda_tiles_root_dir"),
                                                      (MockDeepSMILESlidesPanda, "mock_panda_slides_root_dir")])
def test_mock_panda_container_gpu(mock_container: BaseDeepSMILEPanda,
                                  tmp_path: str,
                                  request: pytest.FixtureRequest) -> None:
    _test_mock_panda_container(use_gpu=True, mock_container=mock_container, tmp_path=request.getfixturevalue(tmp_path))


def test_class_weights_binary() -> None:
    class_weights = Tensor([0.5, 3.5])
    n_classes = 1

    module = TilesDeepMILModule(
        label_column="label",
        n_classes=n_classes,
        class_weights=class_weights,
        encoder_params=get_supervised_imagenet_encoder_params(),
        pooling_params=get_attention_pooling_layer_params(pool_out_dim=1)
    )

    logits = Tensor(randn(1, n_classes))
    bag_label = randint(n_classes + 1, size=(1,))

    pos_weight = Tensor([class_weights[1] / (class_weights[0] + 1e-5)])
    loss_weighted = module.loss_fn(logits.squeeze(1), bag_label.float())
    criterion_unweighted = nn.BCEWithLogitsLoss()
    loss_unweighted = criterion_unweighted(logits.squeeze(1), bag_label.float())
    if bag_label.item() == 1:
        assert allclose(loss_weighted, pos_weight * loss_unweighted)
    else:
        assert allclose(loss_weighted, loss_unweighted)


def test_class_weights_multiclass() -> None:
    class_weights = Tensor([0.33, 0.33, 0.33])
    n_classes = 3

    module = TilesDeepMILModule(
        label_column="label",
        n_classes=n_classes,
        class_weights=class_weights,
        encoder_params=get_supervised_imagenet_encoder_params(),
        pooling_params=get_attention_pooling_layer_params(pool_out_dim=1)
    )

    logits = Tensor(randn(1, n_classes))
    bag_label = randint(n_classes, size=(1,))

    loss_weighted = module.loss_fn(logits, bag_label)
    criterion_unweighted = nn.CrossEntropyLoss()
    loss_unweighted = criterion_unweighted(logits, bag_label)
    # The weighted and unweighted loss functions give the same loss values for batch_size = 1.
    # https://stackoverflow.com/questions/67639540/pytorch-cross-entropy-loss-weights-not-working
    # TODO: the test should reflect actual weighted loss operation for the class weights after
    # batch_size > 1 is implemented.
    assert allclose(loss_weighted, loss_unweighted)
