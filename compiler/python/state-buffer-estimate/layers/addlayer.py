from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from layer           import Layer
from combinelayer    import CombineLayer
import nets.network

##########################################################
class AddLayer(CombineLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, earlier_layer):
        assert(isinstance(prev_layer, Layer))
        assert(isinstance(earlier_layer, Layer))
        assert(prev_layer.gOfmapDesc() == earlier_layer.gOfmapDesc())

        num_ofmaps = prev_layer.gNumOfmaps()
        super(AddLayer, self).__init__(param, prev_layer, earlier_layer, num_ofmaps)


    #-----------------------------------------------------------------
    def gTypeStr(self):
        return "Add"

    #-----------------------------------------------------------------
    def verify(self):
        prevDesc = None
        mapSize = -1
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

