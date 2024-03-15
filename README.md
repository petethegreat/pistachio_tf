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

```docker run -it --rm --name tensorflow_jupy  -v $PWD/notebooks:/tf/notebooks -p 8888:8888 tf_jupy:0.0.1```

don't know that this will do gpu as yet.

## links 

  - [tensorflow docker](https://www.tensorflow.org/install/docker)
  - [quickstart tutorial](https://www.tensorflow.org/tutorials/quickstart/advanced)
  
