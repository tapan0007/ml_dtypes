from utils.consts    import *
from utils.fmapdesc import  OfmapDesc
from layer          import Layer
import nets.network

##########################################################
class SubSampleLayer(Layer): # abstract class
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, num_ofmaps, stride, kernel):
        ## Stride(2) is larger than kernel(1*1) for some layers in ResNet.
        ## That seems to be a wrong way of aggregating information, but
        ## it does happen.
        ##assert(stride <= kernel)

        ifmap_desc = prev_layer.gOfmapDesc()
        if not num_ofmaps:
            num_ofmaps = ifmap_desc.gNumMaps()
        ofmap_size = ifmap_desc.gMapSize() / stride
        ofmap_desc = OfmapDesc(num_ofmaps, ofmap_size);

        super(SubSampleLayer, self).__init__(param, ofmap_desc, (prev_layer,))

        self.__Stride = stride
        self.__Kernel = kernel


    #-----------------------------------------------------------------
    def gStride(self):
        return self.__Stride

    #-----------------------------------------------------------------
    def gKernel(self):
        return self.__Kernel

