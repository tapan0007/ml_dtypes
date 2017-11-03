from utils.consts    import *
from utils.fmapdesc import  OfmapDesc
from .layer          import Layer
from .subsamplelayer import SubSampleLayer
from .poollayer      import PoolLayer
from .activlayer     import ActivLayer
import nets.network

##########################################################
class ConvLayer(SubSampleLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, num_ofmaps, stride, kernel,
            filterFileName, filterTensorDimSemantics):
        assert(isinstance(param, Layer.Param))
        assert(isinstance(prev_layer, Layer))

        super().__init__(param, prev_layer,
            num_ofmaps=num_ofmaps, stride=stride, kernel=kernel)

        self.__FilterFileName = filterFileName
        self.__FilterTensorDimSemantics = filterTensorDimSemantics

    #-----------------------------------------------------------------
    def __str__(self):
        ks = str(self.gKernel())
        ss = str(self.gStride())
        baseLayer = self.gBaseLayerStr()
        return (self.gName()
                + baseLayer
                + ", kernel=" + ks + "x" + ks + ", stride=" + ss
                + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gNumPrevLayers() == 1)
        super().verify()
        prevLayer = self.gPrevLayer(0)
        prevMapSize = prevLayer.gOfmapSize()
        ##assert(self.gPrevLayer(0).gOfmapSize() // self.gStride() == self.gOfmapSize())

    #-----------------------------------------------------------------
    def gFilterFileName(self):
        return self.__FilterFileName

    #-----------------------------------------------------------------
    def gFilterTensorDimSemantics(self):
        return self.__FilterTensorDimSemantics

    #-----------------------------------------------------------------
    def gTypeStr(self):
        return "Conv"

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def gNumberWeights(self):
        assert(self.gNumPrevLayers() == 1)
        num_ifmaps = self.gPrevLayer(0).gNumOfmaps()
        return num_ifmaps *  self.gNumberWeightsPerPartition()

    #-----------------------------------------------------------------
    def gNumberWeightsPerPartition(self):
        assert(self.gNumPrevLayers() == 1)
        k = self.gKernel()
        num_ofmaps = self.gNumOfmaps()
        return k*k * num_ofmaps

    #-----------------------------------------------------------------
    def qConvLayer(self):
        return True


