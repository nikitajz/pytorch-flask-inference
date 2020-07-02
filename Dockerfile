FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

COPY src src
COPY requirements.txt .
COPY download_model_weights.py .

RUN pip install -r requirements.txt

#copy pretrained models weights into the container
RUN python download_model_weights.py
WORKDIR .