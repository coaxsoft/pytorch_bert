# Building BERT with PyTorch from scratch

![img](https://uploads-ssl.webflow.com/60100d26d33c7cce48258afd/6244769a9ec65d641e367414_BERT%20with%20PyTorch.png)

This is the repository containing the code for a tutorial

[Building BERT with PyTorch from scratch](https://coaxsoft.com/blog/building-bert-with-pytorch-from-scratch)

## Installation

After you clone the repository and setup virtual environment,
install dependencies

```shell
pip install -r requirements.txt
```

### Installation on Mac M1

You may experience difficulties installing `tensorboard`.
Tensorboard requires `grpcio` that should be installed with extra environment
variables. Read more in [StackOverflow](https://stackoverflow.com/questions/66640705/how-can-i-install-grpcio-on-an-apple-m1-silicon-laptop).

So, your installation line for Mac M1 should look like

```shell
export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

pip install -r requirements.txt
```
