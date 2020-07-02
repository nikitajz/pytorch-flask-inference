import torchvision


def load_model(model_name, conf):
    """
    This class abstracts out different models loading. One should take care of loading weights for pretrained custom
    models (e.g. from S3).
    Args:
        model_name (str): Model name
        conf (Config): config file

    Returns:
        nn.Module: Pretrained PyTorch model.
    """
    if model_name in conf.torchvision_models:
        return getattr(torchvision.models, model_name)(pretrained=True)
    elif model_name in conf.custom_models:
        raise NotImplementedError("No custom models added so far")
    else:
        raise ValueError(f"Invalid model name to load. Available options are: {conf.allowed_models}")


def get_available_models(conf):
    return [model_name for model_name in conf.allowed_models if model_name != conf.model_name]
