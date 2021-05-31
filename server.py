import sys
sys.path.append('..')

import pytorchtool

import os
import io
import time
import pickle
import torch
import logging
import threading
from PIL import Image
from torchvision import models, transforms

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

from dnnpartition import collaborativeIntelligence

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def trans(img):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 转换为tensor
    img_tensor = inference_transform(Image.open(io.BytesIO(img)))
    img_tensor.unsqueeze_(0)        # chw --> bchw
    
    return img_tensor

class model:
    def __init__(self, model_name, use_gpu=False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.x = torch.rand(3, 224, 224).unsqueeze_(0)

        # 初始化状态
        self._initState = {}
        self._moduleDict = {}

        if self.model_name in 'inception':
            self.model_name = 'inception'
            self.path = "../pytorchtool/model_weight/inception_v3/inception_v3_google-1a9a5a14.pth"

            model = models.Inception3(aux_logits=False, transform_input=False, 
                                    init_weights=False)
            model.eval()
            self.model = model
            self.depth = 2
        elif self.model_name in 'alexnet':
            self.model_name = 'alexnet'
            self.path = "../pytorchtool/model_weight/alexnet/alexnet-owt-4df8aa71.pth"
            
            model = models.alexnet(False)
            model.eval()
            self.model = model
            self.depth = -1 
        else:
            raise RuntimeError("Wrong model name")

        if self.use_gpu:
            self.model = self.model.to(0)
            self.x = self.x.cuda()

        # 获取模型模块、设置模块初始化状态
        for layerName, module in pytorchtool.walk_modules(self.model, depth = self.depth):
            self._moduleDict[layerName] = module
            self._initState[layerName] = False

    def loadWeight(self):
        state_dict_read = torch.load(self.path)
        self.model.load_state_dict(state_dict_read, strict=False)

    def inference(self):
        with torch.no_grad():
            outputs = self.model(self.x)

    def loadlayerWeight(self, layerName):
        if(self._initState[layerName] == False):
            logging.debug("初始化层: %s", layerName)
            # 初始化该层
            self._moduleDict[layerName].load_state_dict(torch.load("../pytorchtool/model_weight/" + 
                self.model.__class__.__name__ + "/" + layerName + ".pth"), strict=False)
            self._initState[layerName] = True

class CollaborativeIntelligenceHandler(object):
    def layerInit(self, layerState):
        for (name, state) in layerState.items():
            if(state == 2):
                # self._m.loadlayerWeight(name)
                self._sModel2.loadLayer(name)
        print("层初始化结束")

    def initModel(self, name, use_gpu=False):
        self.use_gpu = use_gpu
        # m = model(name, use_gpu=use_gpu)
        # m.loadWeight()
        # self._m = m
        # self._sModel = pytorchtool.Surgery(m.model, 2, depth = m.depth)
        if name in 'alexnet':
            self._sModel2 = pytorchtool.Surgery2(name, "../pytorchtool/parameters/alexnet/dag")
        elif name in 'inception':
            self._sModel2 = pytorchtool.Surgery2(name, "../pytorchtool/parameters/inception/dag")
        elif name in 'resnet':
            self._sModel2 = pytorchtool.Surgery2(name, "../pytorchtool/parameters/resnet/dag")
        else:
            raise RuntimeError("Wrong model name")

    def partition(self, layerState):
        print("服务端获取层状态")
        # self._sModel.setLayerState(layerState)
        self._t = threading.Thread(target=CollaborativeIntelligenceHandler.layerInit, 
            name='layerInit', args=(self, layerState,))
        self._t.start()

        # print(layerState)
    def inference(self, middleResult):
        print("服务端获取中间层输入")
        middleResult = {k: pickle.loads(v) for k, v in middleResult.items()}
        if 'input' in middleResult:
            middleResult['input'] = trans(middleResult['input'])
        middleResult = {k: v.cuda() if self.use_gpu 
            else v.cpu() for k, v in middleResult.items()}
        # self._sModel.setMiddleResult(middleResult)
        self._t.join()
        out = pickle.dumps(self._sModel2.inferencePart(middleResult))
        return out 
        # return pickle.dumps(self._sModel(torch.rand(224, 224).unsqueeze_(0)))

def main():
    # 启动服务前进行一次GPU推理完成CUDA初始化以降低第一次推理的时间
    m = model('in', use_gpu=True)
    m.loadWeight()
    output = m.inference()
    print(output)

    handler = CollaborativeIntelligenceHandler()
    handler.initModel('res', use_gpu=True)

    processor = collaborativeIntelligence.Processor(handler)
    transport = TSocket.TServerSocket('192.168.1.121', 9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    rpcServer = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    print('启动服务....')
    rpcServer.serve()

if __name__ == '__main__':
    logging.basicConfig(filename='./log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', 
        level = logging.DEBUG,filemode='a',datefmt='%Y-%m-%d%I:%M:%S %p')
    main()