import torchvision


def load_model(model_name, conf, pretrained=True):
    """
    This class abstracts out different models loading. One should take care of loading weights for pretrained custom
    models (e.g. from S3).

    Args:
        model_name (str): Model name.
        conf (`config.Config`): Config file.
        pretrained (bool, optional): whether to load weights of pretrained model. (Default: True)

    Returns:
        torch.nn.Module: Pretrained PyTorch model.
    """
    if model_name in conf.torchvision_models:
        model = getattr(torchvision.models, model_name)(pretrained=pretrained)
    elif model_name in conf.custom_models:
        raise NotImplementedError("No custom models added so far")
    else:
        raise ValueError(f"Invalid model name to load. Available options are: {conf.allowed_models}")
    model = model.eval()
    return model.to(conf.device)


def get_available_models(conf):
    """Get list of available models except for the default one.

    Args:
        conf (`config.Config`): Config

    Returns:
        list[str]: Model names
    """
    return [model_name for model_name in conf.allowed_models if model_name != conf.model_name]
