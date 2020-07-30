import json
import logging

import requests as req
from flask import Flask, jsonify, request, render_template, url_for, redirect, Response
from torch.utils.data import DataLoader

from src.config import Config
from src.data_utils import ImageDataset
from src.features.transform import TransformImage
from src.models.class_mapping import load_class_mapping
from src.models.model_loader import load_model, get_available_models

app = Flask(__name__)
cfg = Config()
default_model = load_model(model_name=cfg.model_name, conf=cfg)
default_transform_image = TransformImage(cfg.model_name)
default_class_mapping = load_class_mapping(cfg.model_name, cfg)

IMAGE_FILE_TYPES = ['jpg', 'jpeg', 'png', 'tiff']


@app.route('/status')
@app.route('/health-check')
def health_check():
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Use POST method to submit an image file or GET method to submit an encoded url.
    To test use the following examples.
    POST method with an image file:
    `curl -X POST -F file=@"<path_to_file.jpg>" http://<flask_domain>:5000/predict`
    GET method with encoded url:
    `curl -G --data-urlencode "url=<http://example.com/img.jpg>" http://<flask_domain>:5000/predict`

    Returns:
        dict: {class_id: <class_id>, class_name: <class_name>}
    """
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        prediction_result = get_prediction(image_bytes=img_bytes, device=cfg.device, model=default_model,
                                           transform_image=default_transform_image)
        return jsonify(prediction_result)
    elif request.method == 'GET':
        img_url = request.args.get('url', type=str)
        app.logger.debug(f'URL: {img_url}')
        response = req.get(img_url)
        if response.status_code == 200:
            img_bytes = response.content
            prediction_result = get_prediction(image_bytes=img_bytes, device=cfg.device, model=default_model,
                                               transform_image=default_transform_image)
            return jsonify(prediction_result)
        else:
            err_msg = "Unable to get the image from provided url"
            app.logger.warning(err_msg)
            return Response(err_msg, status=404)
    else:
        raise ValueError("Incorrect request type, use either GET or POST method.")


def get_prediction(image_bytes, device, model, transform_image):
    """
    For provided image and model name, perform necessary transformations, apply model (forward pass) and select best
    prediction.
    Args:
        image_bytes (bytes): Image to process
        device (str or torch.device): device to run prediction on
        transform_image:
            transformation to apply to the original image. Should be a class instance with method __call__ implemented.
        model: model instance

    Returns:
        dict: {class_id: <class_id>, class_name: <class_name>}
    """
    tensor = transform_image(image_bytes=image_bytes).unsqueeze(0)
    tensor = tensor.to(device)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    # TODO: abstract out once custom model with different mapping is implemented
    predicted_class_idx = str(y_hat.item())
    predicted_class_id = default_class_mapping[predicted_class_idx][0]
    predicted_class_name = default_class_mapping[predicted_class_idx][1]
    app.logger.info(f"Predicted class: {predicted_class_name} ({predicted_class_id})")
    return {'class_id': predicted_class_id, 'class_name': predicted_class_name}


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == "POST":
        if request.files:
            file = request.files["upload-file"]
            file_type = file.split('.')[-1]
            if file_type in IMAGE_FILE_TYPES:
                img_bytes = file.read()
                model_name = request.form['model-name']
                if model_name != cfg.model_name:
                    app.logger.info(f'Reloading model: {model_name}')
                    global default_model, default_class_mapping, default_transform_image
                    default_model = load_model(model_name, cfg)
                    default_class_mapping = load_class_mapping(model_name, cfg)
                    default_transform_image = TransformImage(model_name)
                prediction_result = get_prediction(img_bytes, cfg.device, default_model, default_transform_image)

                return render_template("prediction_result.html",
                                       user_image=img_bytes,
                                       model_name=model_name,
                                       predicted_class_name=prediction_result['class_name'],
                                       predicted_class_id=prediction_result['class_id'])
            elif file_type in ['csv', 'json']:
                app.logger.error('Not yet implemented')
                return redirect(url_for('upload_file'))
        else:
            return redirect(url_for('upload_file'))

    return render_template("image_upload.html",
                           default_model=cfg.model_name,
                           available_models=get_available_models(cfg))


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if request.method == 'POST':
        response_file = request.files['file']
        if response_file is None:
            return Response("No response_file provided", 404)
        elif response_file.headers['Content-Type'] == 'text/csv':
            urls = response_file.read().decode("utf-8").split()
        elif response_file.headers['Content-Type'] == 'application/json':
            urls = json.loads(response_file.read())['urls']
        else:
            return Response("Incorrect file type", 415)
        app.logger.debug(f'File length: {len(urls)} Samples:\n{urls[:2]}')
        prediction_result = get_batch_prediction(urls)
        return jsonify(prediction_result)


def get_batch_prediction(url_list):
    batch_size = min(len(url_list), cfg.batch_size)
    app.logger.debug(f'Batch size: {batch_size}')
    dataset = ImageDataset(url_list,
                           transform=TransformImage(model_name=cfg.model_name)
                           )
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers)
    predictions = batch_prediction(dataloader, default_model)
    return predictions


def batch_prediction(dataloader, model):
    predicted = list()
    for batch_sample in dataloader:
        outputs = model.forward(batch_sample)
        _, predicted_class_indices = outputs.max(1)
        # TODO: abstract out once custom model with different mapping is implemented
        predictions = [tuple(default_class_mapping[str(y_hat)]) for y_hat in predicted_class_indices.tolist()]
        predicted.extend(predictions)
    return predicted


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)
    app.run()
