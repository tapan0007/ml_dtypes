from utils.consts           import  *
from utils.fmapdesc         import  OfmapDesc
from layers.layer           import  Layer
from layers.batchnormlayer  import  BatchNormLayer
from layers.convlayer       import  ConvLayer
from layers.relulayer       import  ReluLayer
from layers.poollayer       import  PoolLayer

import nets.block
import nets.network


##########################################################
class TranBlock(nets.block.Block):
    def __init__(self, ntwk, blockIdx, prev_layer, compression):
        assert(isinstance(prev_layer, Layer))
        super(TranBlock, self).__init__(ntwk)

        layer = prev_layer

        layer = BatchNormLayer(ntwk, layer)
        layer.rTranBlockStart(blockIdx)

        layer = ReluLayer(ntwk, layer)

        ofmap_desc = layer.gOfmapDesc()
        if compression == 1.0:
            numOfmaps = ofmap_desc.gNumMaps()
        else:
            numOfmaps = int(compression * ofmap_desc.gNumMaps())
        layer = ConvLayer(ntwk, layer, numOfmaps, stride=1, kernel=1)


        layer = PoolLayer(ntwk, layer, stride=2, kernel=2, poolType=POOL_TYPE_AVG)

        layer.rTranBlockEnd(blockIdx)
        self.m_LastLayer = layer

    def gLastLayer(self):
        return self.m_LastLayer

