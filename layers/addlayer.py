from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from layer           import Layer
from combinelayer    import CombineLayer
import nets.network

##########################################################
class AddLayer(CombineLayer):
    #-----------------------------------------------------------------
    def __init__(self, ntwk, prev_layer, earlier_layer):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))
        assert(isinstance(earlier_layer, Layer))
        assert(prev_layer.gOfmapDesc() == earlier_layer.gOfmapDesc())

        num_ofmaps = prev_layer.gNumOfmaps()
        super(AddLayer, self).__init__(ntwk, prev_layer, earlier_layer, num_ofmaps)


    #-----------------------------------------------------------------
    def gLayerType(self):
        return LAYER_TYPE_ADD

    #-----------------------------------------------------------------
    def gName(self):
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

        return ("Add[" + p + "] " + fromTo + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def gBatchInputStateSize(self, batch=1):
        l0 = self.gPrevLayer(0)
        l1 = self.gPrevLayer(1)
        s0 = l0.gSchedule()
        s1 = l1.gSchedule()
        s  = self.gSchedule()
        assert(s0 < s and s1 <s and s0 != s1)

        if s0 == s-1:
            return l1.gRawOutputStateSize()
        elif s1 == s-1:
            return l0.gRawOutputStateSize()
        else:
            return self.gRawInputStateSize()

    #-----------------------------------------------------------------
    def gBatchOutputStateSize(self, batch=1):
        nextSchedLayer = self.gNextSchedLayer()
        if self.qStoreInSB():
            return self.gRawOutputStateSize(batch)
        else:
            return 0


