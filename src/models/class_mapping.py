import json


def load_class_mapping(model_name, conf):
    """
    Load mapping class_id -> class_name for the specified model.
    Args:
        model_name (str): model name for corresponding class mapping
        conf (Config): config file

    Returns:
        dict: Class mapping {class_idx: class_name}
    """
    if model_name in conf.torchvision_models:
        imagenet_mapping = json.load(open('data/imagenet_class_mapping.json'))
        imagenet_mapping = {class_idx: class_name for class_idx, [_, class_name] in imagenet_mapping.items()}
        return imagenet_mapping
    elif model_name in conf.custom_models:
        raise NotImplementedError("No class mapping for the specified model defined")
    else:
        raise ValueError("Class mapping for the specified model is not available")
