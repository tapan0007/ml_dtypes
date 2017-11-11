from utils.consts           import  *
from utils.fmapdesc         import  OfmapDesc
from layers.layer           import  Layer
from layers.batchnormlayer  import  BatchNormLayer
from layers.convlayer       import  ConvLayer
from layers.relulayer       import  ReluLayer
from layers.avgpoollayer    import  AvgPoolLayer

import nets.block
import nets.network


##########################################################
class TranBlock(nets.block.Block):
    #-----------------------------------------------------------------
    def __init__(self, batch, ntwk, blockIdx, prev_layer, compression):
        assert(isinstance(prev_layer, Layer))
        super().__init__(ntwk)

        layer = prev_layer

        pfx = "TBk" + str(blockIdx)
        layer = BatchNormLayer(Layer.Param(pfx + "-bn1", batch, ntwk), layer)
        layer.rTranBlockStart(blockIdx)

        layer = ReluLayer(Layer.Param(pfx + "-relu1", batch, ntwk), layer)

        ofmap_desc = layer.gOfmapDesc()
        if compression == 1.0:
            numOfmaps = ofmap_desc.gNumMaps()
        else:
            numOfmaps = int(compression * ofmap_desc.gNumMaps())
        nm = pfx + "-conv1"
        layer = ConvLayer(Layer.Param(nm, batch, ntwk), layer, numOfmaps, 1, 1, nm+".npy", "MCRS")


        layer = AvgPoolLayer(Layer.Param(pfx + "-avg1", batch, ntwk), layer, stride=2, kernel=2)

        layer.rTranBlockEnd(blockIdx)
        self.m_LastLayer = layer

    #-----------------------------------------------------------------
    def gLastLayer(self):
        return self.m_LastLayer

