# pistachio tensorflow 

get a pistachio classification model running in tensorflow 

## image

```docker pull tensorflow/tensorflow:2.16.1-jupyter```

## running 

```docker run -it --rm --name tensorflow_jupy  -v $PWD/notebooks:/tf/notebooks -p 8888:8888 tensorflow/tensorflow:2.16.1-jupyter```
don't know that this will do gpu as yet.

## links 

  - [tensorflow docker](https://www.tensorflow.org/install/docker)
  - [quickstart tutorial](https://www.tensorflow.org/tutorials/quickstart/advanced)
  