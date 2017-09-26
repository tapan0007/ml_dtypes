from utils.consts    import *
#from utils.fmapdesc  import IfmapDesc
from utils.fmapdesc  import OfmapDesc
from layer           import Layer
from combinelayer    import CombineLayer
import nets.network

##########################################################
class ConcatLayer(CombineLayer):
    #-----------------------------------------------------------------
    def __init__(self, ntwk, prev_layer, earlier_layer):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))
        assert(isinstance(earlier_layer, Layer))
        assert(prev_layer.gOfmapSize() == earlier_layer.gOfmapSize())

        num_ofmaps = prev_layer.gNumOfmaps() + earlier_layer.gNumOfmaps()
        super(ConcatLayer, self).__init__(ntwk, prev_layer, earlier_layer, num_ofmaps)

    #-----------------------------------------------------------------
    def verify(self):
        mapSize = self.gOfmapSize()
        numInputMaps = 0
        for prevLayer in self.gPrevLayers():
            assert(mapSize == prevLayer.gOfmapSize())
            numInputMaps += prevLayer.gNumOfmaps()

        assert(numInputMaps == self.gNumOfmaps())


    #-----------------------------------------------------------------
    def gLayerType(self):
        return LAYER_TYPE_CONCAT

    def gName(self):
        return "Concat"

    #-----------------------------------------------------------------
    def __str__(self):
        numOfmaps = self.gNumOfmaps()
        fromTo = prevLayerIndices = ""
        mapNumStr = ""
        totalNumMaps = 0

        for prevLayer in self.gPrevLayers():
            totalNumMaps += prevLayer.gNumOfmaps()
            assert(prevLayer.gOfmapSize() == self.gOfmapSize())
            if prevLayerIndices == "":
                assert(mapNumStr == "")
                prevLayerIndices = str(prevLayer.gIndex())
                mapNumStr = str(prevLayer.gNumOfmaps())
            else:
                assert(mapNumStr != "")
                prevLayerIndices += "," + str(prevLayer.gIndex())
                mapNumStr += ":" + str(prevLayer.gNumOfmaps())

        assert(totalNumMaps == self.gNumOfmaps())

        fromTo = "(" + mapNumStr + "," + str(self.gOfmapSize()) + ")"

        return ("Concat[" + prevLayerIndices + "] " + fromTo + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return True

