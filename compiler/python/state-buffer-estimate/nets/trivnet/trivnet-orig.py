
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
        layer =  DataLayer(Layer.Param("input", 1, self),
               OfmapDesc(1, 4), inputDataFileName="out_input:0_NCHW.npy", dataTensorDimSemantics="NCHW")
        layer = ConvLayer(Layer.Param("jdr_v2/i1", 1, self), layer,
                   1, stride=1, kernel=1,
                   filterFileName="out_jdr_v2__weight1__read:0_MCRS.npy", filterTensorDimSemantics="MCRS")
        # Golden result file  out_jdr_v2__i1:0_NCHW.npy
        
