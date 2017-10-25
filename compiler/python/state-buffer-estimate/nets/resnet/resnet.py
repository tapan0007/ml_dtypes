"""
ResNet
"""

from abc             import ABCMeta, abstractmethod

from utils.consts           import  *
from utils.fmapdesc         import  OfmapDesc

from layers.layer           import  Layer
from layers.convlayer       import  ConvLayer
from layers.maxpoollayer    import  MaxPoolLayer
from layers.avgpoollayer    import  AvgPoolLayer
from layers.batchnormlayer  import  BatchNormLayer
from layers.datalayer       import  DataLayer
from layers.relulayer       import  ReluLayer
from layers.fulllayer       import  FullLayer
from layers.addlayer        import  AddLayer
from layers.softmaxlayer    import  SoftMaxLayer

from nets.network                import  Network

##########################################################
##########################################################
class ResNet(Network):
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self, ofmap_desc, useRelu):
        super(ResNet, self).__init__()

        self.m_Ofmap_desc = ofmap_desc
        self.__UseRelu = useRelu

    ############################################################################
    def Section(self, sectNum, batches, ntwk, prev_layer, numRepeat, convNumOfmaps, firstStride):

        (firstLayerBatch,otherLayersBatch) = batches

        branchSfx = "_branch"
        branchSfx = "_br"
        forkLayer = layer = prev_layer
        ResN = "res" + str(sectNum)       ## res2
        ResNchar = ResN + "a"           ## res2a
        branch = ResNchar + branchSfx

        layer = ConvLayer(Layer.Param(branch + "2a", firstLayerBatch, self), layer, convNumOfmaps, stride=firstStride, kernel=1)
        layer = ConvLayer(Layer.Param(branch + "2b", otherLayersBatch, self), layer, convNumOfmaps, stride=1, kernel=3)
        layer = ConvLayer(Layer.Param(branch + "2c", otherLayersBatch, self), layer, 4*convNumOfmaps, stride=1, kernel=1)

        ## The next layer is in parallel with the previous 3 layers. It's stride is equal to the first layers stride.
        reshapeShort = ConvLayer(Layer.Param(branch + "1", otherLayersBatch, self), forkLayer, 4*convNumOfmaps, stride=firstStride, kernel=1)
        layer = AddLayer(Layer.Param(ResNchar, otherLayersBatch, self), layer, reshapeShort)
        if self.__UseRelu:
            layer = ReluLayer(Layer.Param(ResNchar + "_relu", otherLayersBatch, self), layer)

        x = ord('b')
        for i in range(numRepeat-1):
            forkLayer = layer

            ResNchar = ResN + chr(x)            ## res3a, res4c, res5f, ...
            branch = ResNchar + branchSfx

            layer = ConvLayer(Layer.Param(branch + "2a", otherLayersBatch, self), layer, convNumOfmaps, stride=1, kernel=1)
            if self.__UseRelu:
                layer = ReluLayer(Layer.Param(branch + "2a_relu", otherLayersBatch, self), layer)
            layer = ConvLayer(Layer.Param(branch + "2b", otherLayersBatch, self), layer, convNumOfmaps, stride=1, kernel=3)
            if self.__UseRelu:
                layer = ReluLayer(Layer.Param(branch + "2b_relu", otherLayersBatch, self), layer)
            layer = ConvLayer(Layer.Param(branch + "2c", otherLayersBatch, self), layer, 4*convNumOfmaps, stride=1, kernel=1)
            layer = AddLayer(Layer.Param(ResNchar, otherLayersBatch, self), layer, forkLayer)
            if self.__UseRelu:
                layer = ReluLayer(Layer.Param(ResNchar + "_relu", otherLayersBatch, self), layer)
            x += 1

        return layer

    #-----------------------------------------------------------------
    def construct(self):
        ofmap_desc = self.m_Ofmap_desc
        ## (3,224)
        layer = DataLayer(Layer.Param("data0", 1, self), ofmap_desc)

        layer = ConvLayer(Layer.Param("conv1", 1, self), layer, 64, stride=2, kernel=7)                ## 7x7 conv, (3,224)->(64,112), stride 2,
        if self.__UseRelu:
            layer = ReluLayer(Layer.Param("conv1_relu", 1, self), layer)
        layer = MaxPoolLayer(Layer.Param("pool1", 2, self), layer, stride=2, kernel=3)              ## Pool (64,112)->(64,56)

        ########################################################################
        layer = self.Section(2, (2,2), self, layer, 3, 64, 1)
        layer = self.Section(3, (2,4), self, layer, 4, 128, 2)
        layer = self.Section(4, (4,8), self, layer, 6, 256, 2)
        layer = self.Section(5, (8,16), self, layer, 3, 512, 2)


        ########################################################################
        layer = AvgPoolLayer(Layer.Param("pool5", 16, self), layer, stride=1, kernel=7)
        layer = FullLayer(Layer.Param("fc1000", 16, self), layer, 1000)
        layer = SoftMaxLayer(Layer.Param("softmax", 16, self), layer)


        self.verify()


##########################################################
class ResNet50(ResNet):
    #-----------------------------------------------------------------
    def __init__(self, useRelu):
        ofmap_desc = OfmapDesc(3, 224)
        super(ResNet50, self).__init__(ofmap_desc, useRelu)

    #-----------------------------------------------------------------
    def gName(self):
        return "ResNet-50"


