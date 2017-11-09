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
    def gJson(self):
        x = super().gJson()
        y = {
            "filter_file" : self.__FilterFileName,
            "filter_dims" : self.__FilterTensorDimSemantics 
        }
        r = self.combineJson( (x, y) )
        return r

    #-----------------------------------------------------------------
    def __str__(self):
        kw = str(self.gKernelWidth())
        kh = str(self.gKernelHeight())
        ss = str(self.gStride())
        baseLayer = self.gBaseLayerStr()
        return (self.gName()
                + baseLayer
                + ", kernel=" + kh + "x" + kw + ", stride=" + ss
                + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gNumPrevLayers() == 1)
        super().verify()
        prevLayer = self.gPrevLayer(0)
        prevMapWidth = prevLayer.gOfmapWidth()
        prevMapHeight = prevLayer.gOfmapHeight()
        ##assert(self.gPrevLayer(0).gOfmapWidth() // self.gStride() == self.gOfmapWidth())
        ##assert(self.gPrevLayer(0).gOfmapHeight() // self.gStride() == self.gOfmapHeight())

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
        kw = self.gKernelWidth()
        kh = self.gKernelHeight()
        num_ofmaps = self.gNumOfmaps()
        return kw*kh * num_ofmaps

    #-----------------------------------------------------------------
    def qConvLayer(self):
        return True


