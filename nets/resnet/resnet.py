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

from nets.network                import  Network

##########################################################
##########################################################
class ResNet(Network):
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self, ofmap_desc):
        super(ResNet, self).__init__()

        self.m_Ofmap_desc = ofmap_desc

    ############################################################################
    def Section(self, sectNum, ntwk, prev_layer, numRepeat, convNumOfmaps, firstStride):
        forkLayer = layer = prev_layer
        pfx="res" + str(sectNum) + "a_branch"
        layer = ConvLayer(pfx + "2a", self, layer, convNumOfmaps, stride=firstStride, kernel=1)
        layer = ConvLayer(pfx + "2b", self, layer, convNumOfmaps, stride=1, kernel=3)
        layer = ConvLayer(pfx + "2c", self, layer, 4*convNumOfmaps, stride=1, kernel=1)
        ## The next layer is in parallel with the previous 3 layers. It's stride
        ## is equal to the first layers stride.
        reshapeShort = ConvLayer(pfx + "1", self, forkLayer, 4*convNumOfmaps, stride=firstStride, kernel=1)
        layer = AddLayer("res" + str(sectNum) + "a", self, layer, reshapeShort)

        x = ord('b')
        for i in range(numRepeat-1):
            forkLayer = layer
            pfx = "res" + str(sectNum) + chr(x) + "_branch"
            layer = ConvLayer(pfx + "2a", self, layer, convNumOfmaps, stride=1, kernel=1)
            layer = ConvLayer(pfx + "2b", self, layer, convNumOfmaps, stride=1, kernel=3)
            layer = ConvLayer(pfx + "2c", self, layer, 4*convNumOfmaps, stride=1, kernel=1)
            layer = AddLayer("res" + str(sectNum) + chr(x), self, layer, forkLayer)
            x += 1

        return layer

    #-----------------------------------------------------------------
    def construct(self):
        ofmap_desc = self.m_Ofmap_desc
        ## (3,224)
        layer = DataLayer("Data", self, ofmap_desc)

        layer = ConvLayer("conv1", self, layer, 64, stride=2, kernel=7)                ## 7x7 conv, (3,224)->(64,112), stride 2,
        layer = MaxPoolLayer("pool1", self, layer, stride=2, kernel=3)              ## Pool (64,112)->(64,56)

        ########################################################################
        layer = self.Section(2, self, layer, 3, 64, 1)
        layer = self.Section(2, self, layer, 4, 128, 2)
        layer = self.Section(2, self, layer, 6, 256, 2)
        layer = self.Section(2, self, layer, 3, 512, 2)


        ########################################################################
        layer = MaxPoolLayer("pool5", self, layer, stride=2, kernel=3)
        layer = FullLayer("fc1000", self, layer, 1000)


        self.verify()


##########################################################
class ResNet50(ResNet):
    #-----------------------------------------------------------------
    def __init__(self):
        ofmap_desc = OfmapDesc(3, 224)
        super(ResNet50, self).__init__(ofmap_desc)

    #-----------------------------------------------------------------
    def gName(self):
        return "ResNet-50"


