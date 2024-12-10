# mlflow 2.0.1 with postgres
FROM ghcr.io/mlflow/mlflow:v2.0.1

RUN apt-get -y update &&\
    apt-get -y install build-essential &&\
    pip install --upgrade pip &&\
    pip install psycopg2-binary

