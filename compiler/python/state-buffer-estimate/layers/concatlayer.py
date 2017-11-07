from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from .layer           import Layer
from .combinelayer    import CombineLayer
import nets.network

##########################################################
class ConcatLayer(CombineLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, earlier_layer):
        assert(isinstance(prev_layer, Layer))
        assert(isinstance(earlier_layer, Layer))
        assert(prev_layer.gOfmapWidth() == earlier_layer.gOfmapWidth())
        assert(prev_layer.gOfmapHeight() == earlier_layer.gOfmapHeight())

        num_ofmaps = prev_layer.gNumOfmaps() + earlier_layer.gNumOfmaps()
        super().__init__(param, prev_layer, earlier_layer, num_ofmaps)

    #-----------------------------------------------------------------
    def verify(self):
        mapWidth = self.gOfmapWidth()
        mapHeight = self.gOfmapHeight()
        numInputMaps = 0
        for prevLayer in self.gPrevLayers():
            assert(mapWidth == prevLayer.gOfmapWidth())
            assert(mapHeight == prevLayer.gOfmapHeight())
            numInputMaps += prevLayer.gNumOfmaps()

        assert(numInputMaps == self.gNumOfmaps())


    #-----------------------------------------------------------------
    def gTypeStr(self):
        return "Concat"

    #-----------------------------------------------------------------
    def __str__(self):
        numOfmaps = self.gNumOfmaps()
        fromTo = prevLayerIndices = ""
        mapNumStr = ""
        totalNumMaps = 0

        for prevLayer in self.gPrevLayers():
            totalNumMaps += prevLayer.gNumOfmaps()
            assert(prevLayer.gOfmapWidth() == self.gOfmapWidth())
            assert(prevLayer.gOfmapHeight() == self.gOfmapHeight())
            if prevLayerIndices == "":
                assert(mapNumStr == "")
                prevLayerIndices = prevLayer.gNumberStr()
                mapNumStr = str(prevLayer.gNumOfmaps())
            else:
                assert(mapNumStr != "")
                prevLayerIndices += "," + prevLayer.gNumberStr()
                mapNumStr += "+" + str(prevLayer.gNumOfmaps())

        assert(totalNumMaps == self.gNumOfmaps())

        fromTo = "(" + mapNumStr + "," + 
                str(self.gOfmapWidth()) + '*' + str(self.gOfmapHeight()) + ")"

        return ("Concat[" + prevLayerIndices + "] " + fromTo + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return True

    #-----------------------------------------------------------------
    def qConcatLayer(self):
        return True

