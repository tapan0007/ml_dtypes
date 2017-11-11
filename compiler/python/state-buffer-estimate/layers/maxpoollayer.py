from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from .layer           import Layer
from .poollayer       import PoolLayer
import nets.network

##########################################################
class MaxPoolLayer(PoolLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, stride, kernel):
        assert(isinstance(prev_layer, Layer))

        super().__init__(param, prev_layer,
            stride=stride, kernel=kernel)

        #self.m_PoolType = poolType

    #-----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()
        return x

    @classmethod
    def constructFromJson(klass, layerDict, nn):
        layerName = Layer.gLayerNameFromJson(layerDict)
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, nn)

        strideLR = PoolLayer.gStrideLRFromJson(layerDict, nn)
        strideBT = PoolLayer.gStrideBTFromJson(layerDict, nn)
        kernelH = PoolLayer.gKernelHeightFromJson(layerDict, nn)
        kernelW = PoolLayer.gKernelWeightFromJson(layerDict, nn)

        paddingLeft = PoolLayer.gPaddingLeftFromJson(layerDict, nn)
        paddingRight = PoolLayer.gPaddingRightFromJson(layerDict, nn)
        paddingTop = PoolLayer.gPaddingTopFromJson(layerDict, nn)
        paddingBottom = PoolLayer.gPaddingBottomFromJson(layerDict, nn)

        stride = (strideLR + strideBT) // 2
        kernel = (kernelH + kernelW) // 2
        batch = 1
        param = Layer.Param(layerName, batch, nn)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, nn)
        assert isinstance(prevLayers, list) and len(prevLayers)==1
            #def __init__(self, param, prev_layer, stride, kernel):
                #assert(isinstance(prev_layer, Layer))
        layer = MaxPoolLayer(param, prevLayers[0], stride, kernel)


    #-----------------------------------------------------------------
    def __str__(self):
        return self.gPoolLayerStr()

    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gNumPrevLayers() == 1)
        assert(self.gPrevLayer(0).gOfmapWidth()  // self.gStrideLR() == self.gOfmapWidth())
        assert(self.gPrevLayer(0).gOfmapHeight() // self.gStrideBT() == self.gOfmapHeight())


    #-----------------------------------------------------------------
    @classmethod
    def gTypeStr(cls):
        t = "MaxPool"
        return t

    #-----------------------------------------------------------------
    def qMaxPoolLayer(self):
        return True

