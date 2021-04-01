import sys
sys.path.append('..')

import pytorchtool

import pickle
import torch

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

class model:
    def __init__(self, model_name, use_gpu=False):
        self.model_name = model_name
        self.use_gpu = use_gpu

        if self.model_name in 'inception':
            self.model_name = 'inception'
            self.path = "../pytorchtool/model_weight/inception_v3/inception_v3_google-1a9a5a14.pth"

            model = models.Inception3(aux_logits=False, transform_input=False, 
                                    init_weights=False)
            model.eval()
            self.model = model
        elif self.model_name in 'alexnet':
            self.model_name = 'alexnet'
            self.path = "../pytorchtool/model_weight/alexnet/alexnet-owt-4df8aa71.pth"
            
            model = models.alexnet(False)
            model.eval()
            self.model = model 
        else:
            print("Wrong model name")

    def loadWeight(self):
        state_dict_read = torch.load(self.path)

        self.model.load_state_dict(state_dict_read, strict=False)

class CollaborativeIntelligenceHandler(object):
    def initModel(self, name):
        m = model(name)
        m.loadWeight()
        self._sModel = pytorchtool.Surgery(m.model)
    def partition(self, layerState):
        self._sModel.setLayerState(layerState)
        # print(layerState)
    def inference(self, middleResult):
        self._sModel.setMiddleResult(
            {k: pickle.loads(v) for k, v in middleResult.items()})
        return self._sModel(torch.rand(224, 224).unsqueeze_(0))

def main():
    handler = CollaborativeIntelligenceHandler()
    handler.initModel('alex')

    processor = collaborativeIntelligence.Processor(handler)
    transport = TSocket.TServerSocket(host, port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    rpcServer = TServer.TSimpleServer(processor, transport, tfactory, pfactory)

    print('启动服务....')
    rpcServer.serve()

if __name__ == '__main__':
    main()