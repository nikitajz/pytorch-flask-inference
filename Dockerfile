FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime as intermediate

COPY requirements.txt src/requirements.txt
RUN pip install -r src/requirements.txt
COPY src src

COPY download_model_weights.py .

RUN mkdir /workspace/.cache
RUN python download_model_weights.py

FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime
COPY --from=intermediate /root/.cache /root/.cache
COPY requirements.txt src/requirements.txt
RUN pip install -r src/requirements.txt
COPY src src
WORKDIR .