from utils.consts    import *
from utils.fmapdesc import  OfmapDesc
from layer          import Layer
from subsamplelayer import SubSampleLayer
import nets.network

##########################################################
class ConvLayer(SubSampleLayer):
    #-----------------------------------------------------------------
    def __init__(self, ntwk, prev_layer, num_ofmaps, stride, kernel):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))

        super(ConvLayer, self).__init__(ntwk, prev_layer,
            num_ofmaps=num_ofmaps, stride=stride, kernel=kernel)

    #-----------------------------------------------------------------
    def __str__(self):
        ks = str(self.gKernel())
        ss = str(self.gStride())
        baseLayer = self.gBaseLayerStr()
        if self.gPrevLayer(0) == self.gNetwork().m_PrevLayer:
            return ("Conv " + baseLayer
                    + ", kernel=" + ks + "x" + ks + ", stride=" + ss
                    + self.gStateSizesStr())
        else:
            p = "[" + self.gPrevLayer(0).gNumberStr() + "]"
            return ("Conv" + p + " " + baseLayer
                    + ", kernel=" + ks + "x" + ks + ", stride=" + ss
                    + self.gStateSizesStr())

    def verify(self):
        assert(self.gNumPrevLayers() == 1)
        prevLayer = self.gPrevLayer(0)
        prevMapSize = prevLayer.gOfmapSize()
        assert(prevMapSize/self.gStride() == self.gOfmapSize())
        ##assert(self.gPrevLayer(0).gOfmapSize()/self.gStride() == self.gOfmapSize())


    #-----------------------------------------------------------------
    def gLayerType(self):
        return LAYER_TYPE_CONV

    def gName(self):
        return "Conv"

    #-----------------------------------------------------------------
    def gNumConvWeights(self):
        k = self.gKernel()
        num_ifmaps = gPrevLayer(0).gNumOfmaps()
        num_ofmaps = self.gNumOfmaps()
        return k*k * num_ifmaps * num_ofmaps

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    def gNumberWeights(self):
        assert(self.gNumPrevLayers() == 1)
        prevLayer = self.gPrevLayer(0)
        return prevLayer.gNumOfmaps() * self.gNumOfmaps() * (self.gKernel()**2)


