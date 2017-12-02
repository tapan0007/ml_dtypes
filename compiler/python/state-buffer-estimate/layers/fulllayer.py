
from utils.consts   import *
from utils.fmapdesc import  OfmapDesc
from .layer          import Layer
from .convlayer      import ConvLayer
import nets.network

class FullLayer(ConvLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, numOuts):
        assert(isinstance(prev_layer, Layer))
        super().__init__(param, prev_layer, numOuts, 1, 1,
                        param.gLayerName()+".npy", "MCRS")

    #-----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()
        return x

    #-----------------------------------------------------------------
    @classmethod
    def constructFromJson(cls, layerDict, ntwk):
        layerName = Layer.gLayerNameFromJson(layerDict)
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, ntwk)

        strideLR = ConvLayer.gStrideLRFromJson(layerDict, ntwk)
        strideBT = ConvLayer.gStrideBTFromJson(layerDict, ntwk)
        kernelH = ConvLayer.gKernelHeightFromJson(layerDict, ntwk)
        kernelW = ConvLayer.gKernelWeightFromJson(layerDict, ntwk)

        paddingLeft = ConvLayer.gPaddingLeftFromJson(layerDict, ntwk)
        paddingRight = ConvLayer.gPaddingRightFromJson(layerDict, ntwk)
        paddingTop = ConvLayer.gPaddingTopFromJson(layerDict, ntwk)
        paddingBottom = ConvLayer.gPaddingBottomFromJson(layerDict, ntwk)

        stride = (strideLR + strideBT) // 2
        kernel = (kernelH + kernelW) // 2
        batch = 1
        param = Layer.Param(layerName, batch, ntwk)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, ntwk)
        assert isinstance(prevLayers, list) and len(prevLayers)==1
            #def __init__(self, param, prev_layer, stride, kernel):
                #assert(isinstance(prev_layer, Layer))
        layer = FullLayer(param, prevLayers[0], ofmapDesc.gNumMaps())

    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return (self.gName() + baseLayer)

    #-----------------------------------------------------------------
    @classmethod
    def gTypeStr(cls):
        return "Full"

    #-----------------------------------------------------------------
    def qFullLayer(self):
        return True

