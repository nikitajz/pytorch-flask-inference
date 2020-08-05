import json
import logging
import time

import requests as req
import torch
from flask import Flask, g, jsonify, redirect, request, render_template, Response, url_for
from torch.utils.data import DataLoader
from werkzeug import exceptions

from src.config import Config
from src.data_utils import ImageDataset
from src.features.transform import TransformImage
from src.models.class_mapping import load_class_mapping
from src.models.model_loader import get_available_models, load_model

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

app = Flask(__name__)
cfg = Config()
default_model = load_model(model_name=cfg.model_name, conf=cfg)
default_transform_image = TransformImage(cfg.model_name)
default_class_mapping = load_class_mapping(cfg.model_name, cfg)

IMAGE_FILE_TYPES = ["jpg", "jpeg", "png", "tiff"]


@app.before_request
def before_request():
    g.start = time.time()


@app.after_request
def after_request(response):
    te = time.time()
    if response.response and (200 <= response.status_code < 300):
        app.logger.info(f"Request completed in {te - g.start:.3f} sec")
    return response


@app.route("/status")
@app.route("/health-check")
def health_check():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Use POST method to submit an image file or GET method to submit an encoded url.

    Examples:
        POST method with an image file::

            $ curl -X POST -F file=@"<path_to_file.jpg>" http://<api_url>:<api_port>/predict

        GET method with encoded url::

            $ curl -G --data-urlencode "url=<http://example.com/img.jpg>" http://<api_url>:<api_port>/predict

        where <api_url> and <api_port> correspond to url and port where the service is deployed (e.g. 127.0.0.1:5000
        for local deployment)

    Returns:
        dict: predicted class::

            {
                'class_id': class_id1,
                'class_name': class_name1
            }

    """
    if request.method == "POST":
        try:
            file = request.files["file"]
            if file.filename == "" or file.filename is None:
                raise exceptions.BadRequestKeyError("No file provided")
            elif file.content_type.split("/")[0] != "image":
                app.logger.warning("File is not an image")
                return Response("File type should be image.", 400)
            img_bytes = file.read()
        except exceptions.BadRequestKeyError as exc:
            app.logger.warning(exc)
            return Response("No file provided or file is corrupted.", 400)
        except Exception as exc:
            app.logger.error(exc)
            return Response("Unknown server error", 500)

    elif request.method == "GET":
        img_url = request.args.get("url", type=str)
        app.logger.debug(f"URL: {img_url}")
        response = req.get(img_url)
        if response.status_code == 200:
            img_bytes = response.content
        else:
            err_msg = "Unable to download the image from the provided url"
            app.logger.warning(err_msg)
            return Response(err_msg, status=404)
    else:
        raise ValueError("Incorrect request type, use either GET or POST methods.")

    prediction_result = get_prediction(
        image_bytes=img_bytes,
        device=cfg.device,
        model=default_model,
        transform_image=default_transform_image,
    )
    return jsonify(prediction_result)


def get_prediction(image_bytes, device, model, transform_image):
    """Run model prediction for the sample image.
    For provided image and model name, perform necessary transformations, apply model (forward pass) and select best
    prediction.

    Args:
        image_bytes (bytes): Image to process
        device (str or torch.device): Device to run prediction on.
        transform_image (class): Transformation to apply to the original image.
            Should be a class instance with method ``__call__`` implemented.
        model (torch.nn.Module): Model instance.

    Returns:
        dict: predicted class::

            {
                'class_id': <class_id>,
                'class_name': <class_name>
            }

    """
    tensor = transform_image(image_bytes=image_bytes)
    tensor.unsqueeze_(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_class_idx = str(y_hat.item())
    predicted_class_id, predicted_class_name = default_class_mapping[predicted_class_idx]
    app.logger.info(f"Predicted class: {predicted_class_name} ({predicted_class_id})")
    return {"class_id": predicted_class_id, "class_name": predicted_class_name}


@app.route("/", methods=["POST", "GET"])
def upload_file():
    if request.method == "POST":
        if request.files:
            file = request.files["upload-file"]
            file_type = file.split(".")[-1]
            if file_type in IMAGE_FILE_TYPES:
                img_bytes = file.read()
                model_name = request.form["model-name"]
                if model_name != cfg.model_name:
                    reload_model(model_name)
                prediction_result = get_prediction(
                    img_bytes, cfg.device, default_model, default_transform_image
                )

                return render_template(
                    "prediction_result.html",
                    user_image=img_bytes,
                    model_name=model_name,
                    predicted_class_name=prediction_result["class_name"],
                    predicted_class_id=prediction_result["class_id"],
                )
            elif file_type in ["csv", "json"]:
                app.logger.error("Not yet implemented")
                return redirect(url_for("upload_file"))
        else:
            return redirect(url_for("upload_file"))

    return render_template(
        "image_upload.html",
        default_model=cfg.model_name,
        available_models=get_available_models(cfg),
    )


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Run model prediction.
    Given the supplied json or csv file with list of http(s) or s3 urls of images provide corresponding predictions.

    Example::

        {
            "urls": [
                "s3://open-images-dataset/validation/2fdfbf67d50e7726.jpg",
                "s3://open-images-dataset/validation/2fed663b4eb60fc8.jpg",
        }

    CSV should have a single column without headers.

    Returns:
        json: predicted class id and name according to class mapping dict
        Example::

            [
              [
                "n02690373",
                "airliner"
              ],
              [
                "n04429376",
                "throne"
              ]
            ]

    """
    if request.method == "POST":
        response_file = request.files["file"]
        if response_file is None:
            return Response("No response_file provided", 404)
        elif response_file.headers["Content-Type"] == "text/csv":
            urls = response_file.read().decode("utf-8").split()
        elif response_file.headers["Content-Type"] == "application/json":
            urls = json.loads(response_file.read())["urls"]
        else:
            return Response("Incorrect file type", 415)
        app.logger.debug(f"File length: {len(urls)} Samples:\n{urls[:2]}")
        data_loader = get_dataloader(urls)
        predictions = predict_all_samples(data_loader, default_model)
        return jsonify(predictions)


def reload_model(model_name):
    """Reload the model and related.
    Load a different model, corresponding class mapping and transform function into the global variables.

    Args:
        model_name (str): model name according to torchvision or custom model name, e.g. ``resnext101_32x8d``
    """
    app.logger.info(f"Reloading model: {model_name}")
    global default_model, default_class_mapping, default_transform_image
    default_model = load_model(model_name, cfg)
    default_class_mapping = load_class_mapping(model_name, cfg)
    default_transform_image = TransformImage(model_name)


def get_dataloader(url_list, model_name=cfg.model_name, batch_size=cfg.batch_size, num_workers=cfg.num_workers):
    """Create and return an instance of dataloader

    Args:
        url_list (list): List of http(s) or S3 urls to fetch images from.
        model_name (str, optional): Model name, used for corresponding TransformImage class.
            Default: derived from the config.
        batch_size (int, optional): Batch size, in case of GPU should be adjusted corresponding to model size and
            available memory on device. Default: derived from the config.
        num_workers (int, optional): Number of workers for dataloader. Default: derived from the config.

    Returns:
        `DataLoader`: torch dataloader instance
    """
    batch_size = min(len(url_list), batch_size)
    app.logger.debug(f"Batch size: {batch_size}")
    dataset = ImageDataset(url_list, transform=TransformImage(model_name=model_name))
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return dataloader


def predict_all_samples(dataloader, model, device=cfg.device):
    """Using the dataloader and model supplied, make prediction for each sample.

    Args:
        dataloader (`DataLoader`): DataLoader instance
        model (`torch.nn.Module`): Model instance
        device (str or `torch.device`): Device, the same as for the model. Default: derived from the config.

    Returns:
        list: Predictions sorted in the same order as samples.
    """
    all_predictions = list()
    for batch_sample in dataloader:
        with torch.no_grad():
            outputs = model.forward(batch_sample.to(device))
            _, predicted_class_indices = outputs.max(1)
            batch_predictions = [tuple(default_class_mapping[str(y_hat)]) for y_hat in predicted_class_indices.tolist()]
        all_predictions.extend(batch_predictions)
    return all_predictions


if __name__ == "__main__":
    app.logger.setLevel = logging.DEBUG
    app.logger.debug(f"Device: {cfg.device} Batch size: {cfg.batch_size}")
    app.run()
