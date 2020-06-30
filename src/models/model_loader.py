import torchvision

from src.config import Config

cfg = Config()


def load_model(model_name, conf):
    """
    This class abstracts out different models loading.
    :param model_name: str
    :return: nn.Module
    """
    if model_name in conf.torchvision_models:
        return getattr(torchvision.models, model_name)(pretrained=True)
    elif model_name in conf.custom_models:
        raise NotImplementedError("No custom models added so far")
    else:
        raise ValueError(f"Invalid model name to load. Available options are: {cfg.allowed_models}")
