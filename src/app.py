import logging

import requests as req
from flask import Flask, jsonify, request, render_template, url_for, redirect, Response

from src.config import Config
from src.features.transform import transform_image
from src.models.class_mapping import load_class_mapping
from src.models.model_loader import load_model, get_available_models

logger = logging.getLogger(__name__)
app = Flask(__name__)
cfg = Config()
model = load_model(model_name=cfg.model_name, conf=cfg)
class_mapping = load_class_mapping(cfg.model_name, cfg)


@app.route('/status')
@app.route('/health-check')
def health_check():
    return jsonify({"status": "ok"})


@app.route('/predict', methods=['POST', 'GET'])
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
        prediction_result = get_prediction(image_bytes=img_bytes, model_name=cfg.model_name, device=cfg.device)
        return jsonify(prediction_result)
    elif request.method == 'GET':
        img_url = request.args.get('url', type=str)
        logger.debug(f'URL: {img_url}')
        response = req.get(img_url)
        if response.status_code == 200:
            img_bytes = response.content
            prediction_result = get_prediction(image_bytes=img_bytes, model_name=cfg.model_name, device=cfg.device)
            return jsonify(prediction_result)
        else:
            err_msg = "Unable to retrieve the message from requested url"
            logger.error(err_msg)
            return Response(err_msg, status=400)
    else:
        raise ValueError("Incorrect request type, use POST method.")


def get_prediction(image_bytes, model_name, device):
    """
    For provided image and model name, perform necessary transformations, apply model (forward pass) and select best
    prediction.
    Args:
        image_bytes (bytes): Image to process
        model_name (str): Model name to apply corresponding transformation.
        device (str or torch.device): device to run prediction on

    Returns:
        dict: {class_id: <class_id>, class_name: <class_name>}
    """
    tensor = transform_image(image_bytes=image_bytes, model_name=model_name)
    tensor = tensor.to(device)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    # TODO: abstract out once custom model with different mapping is implemented
    predicted_class_idx = str(y_hat.item())
    predicted_class_id = class_mapping[predicted_class_idx][0]
    predicted_class_name = class_mapping[predicted_class_idx][1]
    logger.info(f"Predicted class: {predicted_class_name} ({predicted_class_id})")
    return {'class_id': predicted_class_id, 'class_name': predicted_class_name}


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == "POST":
        if request.files:
            file = request.files["upload-file"]
            img_bytes = file.read()
            model_name = request.form['model-name']
            app.logger.info(f'Chosen model: {model_name}')
            pred_result = get_prediction(img_bytes, model_name, cfg.device)

            return render_template("prediction_result.html",
                                   user_image=img_bytes,
                                   model_name=model_name,
                                   predicted_class_name=pred_result['class_name'],
                                   predicted_class_id=pred_result['class_id'])
        else:
            return redirect(url_for('upload_file'))

    return render_template("image_upload.html",
                           default_model=cfg.model_name,
                           available_models=get_available_models(cfg))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG)
    app.run()
