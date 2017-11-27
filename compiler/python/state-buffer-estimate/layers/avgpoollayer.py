from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from .layer           import Layer
from .poollayer       import PoolLayer
import nets.network

##########################################################
class AvgPoolLayer(PoolLayer):
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

    #-----------------------------------------------------------------
    @classmethod
    def constructFromJson(klass, layerDict, ntwk):
        layerName = Layer.gLayerNameFromJson(layerDict)
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, ntwk)

        strideLR = PoolLayer.gStrideLRFromJson(layerDict, ntwk)
        strideBT = PoolLayer.gStrideBTFromJson(layerDict, ntwk)
        kernelH = PoolLayer.gKernelHeightFromJson(layerDict, ntwk)
        kernelW = PoolLayer.gKernelWeightFromJson(layerDict, ntwk)

        paddingLeft = PoolLayer.gPaddingLeftFromJson(layerDict, ntwk)
        paddingRight = PoolLayer.gPaddingRightFromJson(layerDict, ntwk)
        paddingTop = PoolLayer.gPaddingTopFromJson(layerDict, ntwk)
        paddingBottom = PoolLayer.gPaddingBottomFromJson(layerDict, ntwk)

        stride = (strideLR + strideBT) // 2
        kernel = (kernelH + kernelW) // 2
        batch = 1
        param = Layer.Param(layerName, batch, ntwk)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, ntwk)
        assert isinstance(prevLayers, list) and len(prevLayers)==1
            #def __init__(self, param, prev_layer, stride, kernel):
                #assert(isinstance(prev_layer, Layer))

        return AvgPoolLayer(param, prevLayers[0], stride, kernel)

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
    def gTypeStr(self):
        t = "AvgPool"
        return t


    #-----------------------------------------------------------------
    def qAvgPoolLayer(self):
        return True

