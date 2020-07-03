## Serving PyTorch model using Flask and docker 
This app is a blueprint for serving PyTorch model using Flask. It mostly suitable for demo purposes without high load (it is not fit for concurrent or batch processing yet).
The app accepts an image either via web page upload or POST request to endpoint `<url>/predict`. For example:
```shell script
curl -X POST -F file=@"cat.jpg" http://0.0.0.0:5050/predict
```
It can be run both as usual flask app and deployed as Docker container with flask app inside. 
Take into account that the app downloads pre-trained models' weights, with default configuration ~1.5Gb.

In order to run docker container, first install [docker](https://docs.docker.com/engine/install/) and [docker-compose](https://docs.docker.com/compose/install/).
Then run the following commands (in the directory where you want to clone git repo):
```shell script
git clone https://github.com/nikitajz/ml_prod
cd ml_prod
docker compose up --build
```

Among the output, you should see the message similar to below:
>webapp_1  | [2020-07-03 08:49:42 +0000] [1] [INFO] Starting gunicorn 20.0.4   
>webapp_1  | [2020-07-03 08:49:42 +0000] [1] [INFO] Listening at: http://0.0.0.0:5050 (1)

Now the app is available locally at http://0.0.0.0:5050.
