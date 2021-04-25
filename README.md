## 克隆项目
```shell
# 服务端代码
git clone https://github.com/FENP/DNN_Partition-server.git
# 辅助工具
git clone https://github.com/FENP/pytorchtool.git
```

## 下载模型权重

```shell
cd ./pytorchtool
mkdir model_weight
cd ./model_weight
```

```shell
# AlexNet
mkdir alexnet
cd ./alexnet
wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
```

```shell
# InceptionV3
mkdir inception_v3
cd ./inception_v3
wget https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
```

## 编译thrift文件

参考客户端的编译过程，服务端可直接使用输出的python包：`dnnpartition`

## 获取模型各层参数

使用pytorchtool对模型各层进行分析，得到服务端的各层执行参数。

## 运行服务端

`python server.py`