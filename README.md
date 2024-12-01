# pistachio tensorflow 

The initial aim of this project was to get a pistachio classification model running using tensorflow.
During the process of this, decided that I wanted to use mlflow for experiment tracking, and to run that using a docker container, and set up a docker compose solution to manage containers.
Current things to do are:
  - write a notebook for test set evaluation
  - integrate hyperopt for tuning
  - try some other model architectures (e.g. wide and deep)


## image

source tf image 
```docker pull tensorflow/tensorflow:2.16.1-jupyter```
add pandas annd scipy

```bash
docker build -t tf_jupy:0.0.1 ./image
```

## running - docker compose 
container arguments are set in compose.yaml
to start things, run
```docker compose up -d```
this will start the jupy and mlflow containers
jupyterlab is at localhost:8888
mlflow ui is at localhost:5000

to log into jupyterlab requires a token, this can be found by running
```bash
docker exec tensorflow_jupy jupyter server list
```

## tensorboard
in a terminal (from within jupyter), run
```bash
tensorboard --logdir pistachio_model_logs --bind_all
```

## running - individual containers 
it's handy having jupyterlab and mlflow for tracking

first create a docker network
```bash
 docker network create pt_mlflow_net
 ```

then start jupyterlab in this network

```docker run -it --rm --name tensorflow_jupy  -v $PWD/notebooks:/tf/notebooks -v $PWD/mlflow:/mlflow -p 8888:8888 -p 6006:6006 --network pt_mlflow_net tf_jupy:0.0.1```


pull mlflow image
```bash
 docker pull ghcr.io/mlflow/mlflow:v2.0.1
 ```

then start mlflow 
```bash 
 docker run --rm -d --network pt_mlflow_net --name pistachio_mlflow -p 5000:5000  -v $PWD/mlflow:/mlflow ghcr.io/mlflow/mlflow:v2.0.1 mlflow server --backend-store-uri /mlflow --default-artifact-root /mlflow/artifacts --host 0.0.0.0 --port 5000
```

mounting /mlflow in both containers is like a dirty nfs


## links 

  - [tensorflow docker](https://www.tensorflow.org/install/docker)
  - [quickstart tutorial](https://www.tensorflow.org/tutorials/quickstart/advanced)

  - [custom training loop with early stopping](https://www.tensorflow.org/guide/migrate/early_stopping)
  - [custom training loop logging metrics](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)
  
  - [more on custom training loop](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
