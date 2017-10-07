from utils.consts           import  *
from utils.fmapdesc         import  OfmapDesc
from layers.layer           import  Layer
from layers.batchnormlayer  import  BatchNormLayer
from layers.convlayer       import  ConvLayer
from layers.relulayer       import  ReluLayer
from layers.concatlayer     import  ConcatLayer

import nets.block
import nets.network

##########################################################
class DenseBlock(nets.block.Block):
    #-----------------------------------------------------------------
    def __init__(self, ntwk, blockIdx, numDenseLayers, growthRate, prev_layer):
        assert(isinstance(prev_layer, Layer))
        super(DenseBlock, self).__init__(ntwk)

        self.m_numDenseLayers = numDenseLayers
        growthRate = ntwk.gGrowthRate()
        layer = prev_layer

        for layerNum in range(numDenseLayers):
            first_layer_in_dense_subblock = layer

            pfx = "DBk" + str(blockIdx) + "-LY" + str(layerNum)
            layer = BatchNormLayer(pfx + "-BN1", ntwk, layer)
            if layerNum == 0:
                layer.rDenseBlockStart(blockIdx)

            layer = ReluLayer(pfx + "-RL1", ntwk, layer)

            layer = ConvLayer(pfx + "-CNV1", ntwk, layer, 4*growthRate, stride=1, kernel=1)
            #layer = Drop

            layer = BatchNormLayer(pfx + "-BN2", ntwk, layer)

            layer = ReluLayer(pfx + "-RL2", ntwk, layer)

            layer = ConvLayer(pfx + "-CNV2", ntwk, layer, growthRate, stride=1, kernel=3)
            #layer = Drop

            layer = ConcatLayer(pfx + "-CAT", ntwk, layer, first_layer_in_dense_subblock)


        layer.rDenseBlockEnd(blockIdx)
        self.m_LastLayer = layer

    #-----------------------------------------------------------------
    def gLastLayer(self):
        return self.m_LastLayer


