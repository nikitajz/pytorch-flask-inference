import pytest
import torchvision

from src.models.model_loader import get_available_models, load_model


@pytest.fixture
def get_config_file():
    from src.config import Config
    return Config()


def test_torchvision_models(get_config_file):
    # TODO: move to the config validation
    conf = get_config_file
    assert all([hasattr(torchvision.models, model_name) for model_name in conf.torchvision_models]), \
        "Specified in the conf.torchvision_models should be actual torchvision models"


def test_available_models(get_config_file):
    conf = get_config_file
    available_models = get_available_models(conf)
    assert conf.model_name not in available_models, "Default model shouldn't be in the list"


def test_load_model_torchvision(monkeypatch, get_config_file):
    model_name = 'vgg16'

    conf = get_config_file
    conf.torchvision_models = ['resnext101_32x8d', 'resnext50_32x4d', 'vgg16',
                               'vgg16_bn', 'vgg19', 'vgg19_bn']
    model = load_model(model_name, conf, pretrained=False)
    assert model.__class__.__name__.lower() in model_name, "Model name should correspond to model class name"


def test_load_model_custom(get_config_file):
    conf = get_config_file
    model_name = "custom_model_name"
    conf.custom_models = [model_name]
    with pytest.raises(NotImplementedError):
        load_model(model_name, conf)


def test_load_model_custom(get_config_file):
    conf = get_config_file
    model_name = "non-existing_model_name"
    with pytest.raises(ValueError):
        load_model(model_name, conf)
