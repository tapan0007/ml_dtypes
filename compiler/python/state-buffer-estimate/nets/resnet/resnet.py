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
class ResNet(Network, metaclass = ABCMeta):

    #-----------------------------------------------------------------
    def __init__(self, ofmap_desc, dataType, netName, useRelu):
        super().__init__(dataType, netName)

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

        nm = branch + "2a"
        layer = ConvLayer(Layer.Param(nm, firstLayerBatch, self), layer,
                    convNumOfmaps, firstStride, 1, nm+".npy", "MCRS")
        nm = branch + "2b"
        layer = ConvLayer(Layer.Param(nm, otherLayersBatch, self), layer,
                    convNumOfmaps, 1, 3, nm+".npy", "MCRS")
        nm = branch + "2c"
        layer = ConvLayer(Layer.Param(nm, otherLayersBatch, self), layer,
                    4*convNumOfmaps, 1, 1, nm+".npy", "MCRS")

        ## The next layer is in parallel with the previous 3 layers. It's stride is equal to the first layers stride.
        nm = branch + "1"
        reshapeShort = ConvLayer(Layer.Param(nm, otherLayersBatch, self), forkLayer,
                        4*convNumOfmaps, firstStride, 1, nm+".npy", "MCRS")
        layer = AddLayer(Layer.Param(ResNchar, otherLayersBatch, self), layer, reshapeShort)
        if self.__UseRelu:
            layer = ReluLayer(Layer.Param(ResNchar + "_relu", otherLayersBatch, self), layer)

        x = ord('b')
        for i in range(numRepeat-1):
            forkLayer = layer

            ResNchar = ResN + chr(x)            ## res3a, res4c, res5f, ...
            branch = ResNchar + branchSfx

            nm = branch + "2a"
            layer = ConvLayer(Layer.Param(nm, otherLayersBatch, self), layer,
                        convNumOfmaps, 1, 1, nm+".npy", "MCRS")
            if self.__UseRelu:
                layer = ReluLayer(Layer.Param(branch + "2a_relu", otherLayersBatch, self), layer)
            nm = branch + "2b"
            layer = ConvLayer(Layer.Param(nm, otherLayersBatch, self), layer,
                        convNumOfmaps, 1, 3, nm+".npy", "MCRS")
            if self.__UseRelu:
                layer = ReluLayer(Layer.Param(branch + "2b_relu", otherLayersBatch, self), layer)
            nm = branch + "2c"
            layer = ConvLayer(Layer.Param(nm, otherLayersBatch, self), layer,
                        4*convNumOfmaps, 1, 1, nm+".npy", "MCRS")
            layer = AddLayer(Layer.Param(ResNchar, otherLayersBatch, self), layer, forkLayer)
            if self.__UseRelu:
                layer = ReluLayer(Layer.Param(ResNchar + "_relu", otherLayersBatch, self), layer)
            x += 1

        return layer

    #-----------------------------------------------------------------
    def construct(self):
        ofmap_desc = self.m_Ofmap_desc
        ## (3,224)
        layer = DataLayer(Layer.Param("data0", 1, self), ofmap_desc,
                    "input.npy", "NCHW")

        nm = "conv1"
        layer = ConvLayer(Layer.Param(nm, 1, self), layer, 64,
                2, 7, nm+".npy", "MCRS")
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
        self.__Constructed = True


##########################################################
class ResNet50(ResNet):
    #-----------------------------------------------------------------
    def __init__(self, dataType, useRelu):
        ofmap_desc = OfmapDesc(3, 224)
        super().__init__(ofmap_desc, dataType, "ResNet50", useRelu)


