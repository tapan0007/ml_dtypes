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
    def __init__(self, batch, ntwk, blockIdx, numDenseLayers, growthRate, prev_layer):
        assert(isinstance(prev_layer, Layer))
        super().__init__(ntwk)

        self.m_numDenseLayers = numDenseLayers
        growthRate = ntwk.gGrowthRate()
        layer = prev_layer

        for layerNum in range(numDenseLayers):
            first_layer_in_dense_subblock = layer

            pfx = "DBk" + str(blockIdx) + "-LY" + str(layerNum)
            layer = BatchNormLayer(Layer.Param(pfx + "-bn1", batch, ntwk), layer)
            if layerNum == 0:
                layer.rDenseBlockStart(blockIdx)

            layer = ReluLayer(Layer.Param(pfx + "-relu1", batch, ntwk), layer)

            nm = pfx + "-conv1"
            layer = ConvLayer(Layer.Param(nm, batch, ntwk), layer, 4*growthRate, 1, 1, nm+".npy", "MCRS")
            #layer = Drop

            layer = BatchNormLayer(Layer.Param(pfx + "-bn2", batch, ntwk), layer)

            layer = ReluLayer(Layer.Param(pfx + "-relu2", batch, ntwk), layer)

            nm = pfx + "-conv2"
            layer = ConvLayer(Layer.Param(nm, batch, ntwk), layer, growthRate, 1, 3, nm+".npy", "MCRS")
            #layer = Drop

            nm = pfx + "-cat"
            layer = ConcatLayer(Layer.Param(nm, batch, ntwk), layer, first_layer_in_dense_subblock)


        layer.rDenseBlockEnd(blockIdx)
        self.m_LastLayer = layer

    #-----------------------------------------------------------------
    def gLastLayer(self):
        return self.m_LastLayer


