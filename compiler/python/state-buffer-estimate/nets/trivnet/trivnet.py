
from utils.fmapdesc     import OfmapDesc
from utils.datatype     import *

from layers.layer       import Layer
from layers.datalayer   import DataLayer
from layers.convlayer   import ConvLayer

from nets.network       import Network


class TrivNet(Network):
    def __init__(self):
        super().__init__(DataTypeFloat16())

    def gName(self):
        return "TrivNet"

    def construct(self):
        tensorOrder = "NHWC"  ## produced by TF
        tensorOrder = "NCHW"  ## expected by sim (otherwise sim crashes)

        layer =  DataLayer(Layer.Param("jdr_v3/input", 1, self),
                    OfmapDesc(4, 8), inputDataFileName="out_input:0.npy", dataTensorDimSemantics=tensorOrder)

        layer = ConvLayer(Layer.Param("jdr_v3/i1", 1, self), layer,
                    4, stride=1, kernel=3,
                    filterFileName="out_jdr_v3__weight1__read:0.npy", filterTensorDimSemantics="MCRS")
        # Golden result file  out_jdr_v3__i1:0.npy
        #
        layer = ConvLayer(Layer.Param("jdr_v3/i2", 1, self), layer,
                    4, stride=1, kernel=3,
                    filterFileName="out_jdr_v3__weight2__read:0.npy", filterTensorDimSemantics="MCRS")
        # Golden result file  out_jdr_v3__i2:0.npy

