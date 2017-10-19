"""
Parameters obtained from the source code located at
https://github.com/pudae/tensorflow-densenet
"""

from abc             import ABCMeta, abstractmethod

from utils.consts           import  *
from utils.fmapdesc         import  OfmapDesc

from layers.layer           import  Layer
from layers.convlayer       import  ConvLayer
from layers.avgpoollayer    import  AvgPoolLayer
from layers.maxpoollayer    import  MaxPoolLayer
from layers.batchnormlayer  import  BatchNormLayer
from layers.datalayer       import  DataLayer
from layers.relulayer       import  ReluLayer
from layers.fulllayer       import  FullLayer
from layers.softmaxlayer    import  SoftMaxLayer

#from blocks.denseblock      import  DenseBlock
#from blocks.tranblock       import  TranBlock
from denseblock      import  DenseBlock
from tranblock       import  TranBlock

from nets.network                import  Network

##########################################################
##########################################################
class DenseNet(Network):
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self, growth_rate, layers_in_dense_block, ofmap_desc, num_classes, useRelu):
        super(DenseNet, self).__init__()

        self.m_Growth_rate = growth_rate
        self.m_LayersInDenseBlock = layers_in_dense_block
        self.m_Ofmap_desc = ofmap_desc
        self.m_NumClasses = num_classes
        self.__UseRelu    = useRelu

    #-----------------------------------------------------------------
    def gGrowthRate(self):
        return self.m_Growth_rate

    #-----------------------------------------------------------------
    def construct(self):
        batch = 1
        ofmap_desc = self.m_Ofmap_desc
        layer = DataLayer(Layer.Param("data0", batch, self), ofmap_desc)

        ## Initial convolution + batch, relu, pool
        # Convolution IMAP=3, OMAPS=64, kernel 7x7, stride 2, image size 224 -> 112
        num_ofmaps = 2*self.m_Growth_rate
        layer = ConvLayer(Layer.Param("conv1", batch, self), layer, num_ofmaps, stride=2, kernel=7)
        # Batch Normalization, IMAPS=64, OMAPS=64, image size 112->112
        layer = BatchNormLayer(Layer.Param("bn1", batch, self), layer)
        # ReLU, IMAPS=64, OMAPS=64, image size 112->112
        layer = ReluLayer(Layer.Param("relu1", batch, self), layer)
        ## Pooling IMAPS=64, OMAPS=64, 3x3, stride 2, pad 1, 112 -> 56
        layer = MaxPoolLayer(Layer.Param("MaxPool1", batch, self), layer, stride=2, kernel=3)


        # Dense Blocks
        numDenseBlocks = len(self.m_LayersInDenseBlock)
        for blockIdx in range(numDenseBlocks-1):
            numLayersInBlock = self.m_LayersInDenseBlock[blockIdx]
            denseBlock = DenseBlock(batch, self, blockIdx, numLayersInBlock, self.m_Growth_rate, layer)
            layer = denseBlock.gLastLayer()
            tranBlock = TranBlock(batch, self, blockIdx, layer, 0.5)
            layer = tranBlock.gLastLayer()

        lastBlockIdx = numDenseBlocks - 1
        numLayersInBlock = self.m_LayersInDenseBlock[lastBlockIdx]
        denseBlock = DenseBlock(batch, self, lastBlockIdx, numLayersInBlock, self.m_Growth_rate, layer)
        layer = denseBlock.gLastLayer()

        ## Final blocks
        pfx = "TB" + str(lastBlockIdx)
        layer = BatchNormLayer(Layer.Param(pfx + "-bn1", batch, self), layer)
        layer = ReluLayer(Layer.Param(pfx + "-relu1", batch, self), layer)
        layer = AvgPoolLayer(Layer.Param(pfx + "-avg1", batch, self), layer, stride=7, kernel=7)

        s = "FC" + str(self.m_NumClasses)
        layer = FullLayer(Layer.Param(s, batch, self), layer, self.m_NumClasses)
        layer = SoftMaxLayer(Layer.Param("SoftMax", batch, self), layer)

        self.verify()


##########################################################
class DenseNet169(DenseNet):
    #-----------------------------------------------------------------
    def __init__(self, useRelu):
        growth_rate = 32
        layersInDenseBlock = [6, 12, 32, 32]
        ofmap_desc = OfmapDesc(3, 224)
        num_classes = 1000
        super(DenseNet169, self).__init__(growth_rate, layersInDenseBlock, ofmap_desc, num_classes, useRelu)

    #-----------------------------------------------------------------
    def gName(self):
        return "DenseNet-169"


