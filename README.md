# pistachio tensorflow 

get a pistachio classification model running in tensorflow 

## image

source tf image 
```docker pull tensorflow/tensorflow:2.16.1-jupyter```
add pandas annd scipy

```bash
docker build -t tf_jupy:0.0.1 ./image
```


## running 

```docker run -it --rm --name tensorflow_jupy  -v $PWD/notebooks:/tf/notebooks -p 8888:8888 -p 6006:6006 --network pt_mlflow_net tf_jupy:0.0.1```

don't know that this will do gpu as yet.

## tensorboard
in a terminal (from within jupyter), run
```bash
tensorboard --logdir pistachio_model_logs --bind_all
```

## mlflow for experiment tracking

create a docker network
```bash
 docker network create pt_mlflow_net
 ```

pull mlflow image
```bash
 docker pull ghcr.io/mlflow/mlflow:v2.0.1
 ```

start mlflow 
```bash 
 docker run --rm -d --network pt_mlflow_net --name pistachio_mlflow -p 5000:5000  -v $PWD/mlflow:/mlflow ghcr.io/mlflow/mlflow:v2.0.1 mlflow server --backend-store-uri /mlflow --default-artifact-root /mlflow/artifacts --host 0.0.0.0 --port 5000
```



## links 

  - [tensorflow docker](https://www.tensorflow.org/install/docker)
  - [quickstart tutorial](https://www.tensorflow.org/tutorials/quickstart/advanced)

  - [custom training loop with early stopping](https://www.tensorflow.org/guide/migrate/early_stopping)
  - [custom training loop logging metrics](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough)
  
  - [more on custom training loop](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
