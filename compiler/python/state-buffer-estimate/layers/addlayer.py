from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from .layer           import Layer
from .combinelayer    import CombineLayer
import nets.network

##########################################################
class AddLayer(CombineLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, earlier_layer):
        assert(isinstance(prev_layer, Layer))
        assert(isinstance(earlier_layer, Layer))
        assert(prev_layer.gOfmapDesc() == earlier_layer.gOfmapDesc())

        num_ofmaps = prev_layer.gNumOfmaps()
        super().__init__(param, prev_layer, earlier_layer, num_ofmaps)

    #-----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()
        return x

    #-----------------------------------------------------------------
    @classmethod
    def constructFromJson(klass, layerDict, ntwk):
        layerName = Layer.gLayerNameFromJson(layerDict)
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, ntwk)
        batch = 1
        param = Layer.Param(layerName, batch, ntwk)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, ntwk)
        assert isinstance(prevLayers, list) and len(prevLayers)==2

        return AddLayer(param, prevLayers[0], prevLayers[1])

    #-----------------------------------------------------------------
    @classmethod
    def gTypeStr(klass):
        return "Add"

    #-----------------------------------------------------------------
    def verify(self):
        prevDesc = None
        numOfmaps = 0

        numPreLayers = self.gNumPrevLayers()

        for i in range(numPreLayers):
            prevLayer = self.gPrevLayer(i)
            if not prevDesc:
                prevDesc = prevLayer.gOfmapDesc()
            else:
                assert(prevDesc == prevLayer.gOfmapDesc())

        assert(prevDesc == self.gOfmapDesc())

    #-----------------------------------------------------------------
    def __str__(self):
        numOfmaps = self.gNumOfmaps()

        nPrevs = self.gNumPrevLayers()
        oneMapStr = str(self.gOfmapDesc())
        fromTo = (str(nPrevs) + "*" + oneMapStr + "->" + oneMapStr)

        p = ""

        for prevLayer in self.gPrevLayers():
            if p == "":
                p = prevLayer.gNumberStr()
            else:
                p += "," + prevLayer.gNumberStr()

        return (self.gName()
            + "[" + p + "] " + fromTo + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def qAddLayer(self):
        return True

