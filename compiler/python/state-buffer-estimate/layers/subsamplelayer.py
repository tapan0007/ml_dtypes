from utils.consts    import *
from utils.fmapdesc import  OfmapDesc
from .layer          import Layer
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
        ofmap_width = ifmap_desc.gMapWidth()  //  stride
        ofmap_height = ifmap_desc.gMapHeight()  //  stride
        ofmap_desc = OfmapDesc(num_ofmaps, (ofmap_width, ofmap_height));

        super().__init__(param, ofmap_desc, (prev_layer,))

        ## Stride
        if isinstance(stride, tuple):
            assert len(stride)==2 and isinstance(stride[0], int) and isinstance(stride[1], int)
            self.__StrideBT = stride[0]
            self.__StrideLR = stride[1]
            assert isinstance(self.__StrideLR, int)
        else:
            assert isinstance(stride, int)
            self.__StrideLR = self.__StrideBT = stride

        if isinstance(kernel, tuple):
            assert len(kernel)==2 and isinstance(kernel[0], int) and isinstance(kernel[1], int)
            self.__KernelHeight = kernel[0]
            self.__KernelWidth  = kernel[1]
        else:
            assert isinstance(kernel, int)
            self.__KernelHeight = self.__KernelWidth = kernel 

        ## Padding
        kh = self.gKernelHeight()
        if kh % 2 == 1:
            self.__PaddingBottom = self.__PaddingTop = (kh - 1) // 2
        else:
            self.__PaddingBottom = 0
            self.__PaddingTop    = kh // 2

        kw = self.gKernelWidth()
        if kw % 2 == 1:
            self.__PaddingRight  = self.__PaddingLeft  = (kw - 1) // 2
        else:
            self.__PaddingLeft   = 0
            self.__PaddingRight  = kw // 2

    #-----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()

        if self.gNetwork().gUseDimList():
            kernel = [1, 1, self.gKernelHeight(), self.gKernelWidth()]
            stride = [1, 1, self.gStrideBT(), self.gStrideLR()]
            padding = [ [0,0], [0,0], [self.gPaddingBottom(), self.gPaddingTop()], [self.gPaddingLeft(), self.gPaddingRight()]]
        else:
            kernel  = {
                "height" : self.gKernelHeight(),
                "width" : self.gKernelWidth()
            }
            padding = {
                "top" : self.gPaddingTop(),
                "bottom" : self.gPaddingBottom(),
                "left" : self.gPaddingLeft(),
                "right" : self.gPaddingRight()
            }
            stride = {
                "BT" : self.gStrideBT(),
                "LR" : self.gStrideLR()
            }

        y = {
            "kernel" : kernel,
            "stride" : stride,
            "padding" : padding
        }
        r = self.combineJson( (x, y) )
        return r

    #-----------------------------------------------------------------
    def gStrideLR(self):
        return self.__StrideLR

    #-----------------------------------------------------------------
    def gStrideBT(self):
        return self.__StrideBT

    #-----------------------------------------------------------------
    def gKernelHeight(self):
        return self.__KernelHeight

    #-----------------------------------------------------------------
    def gKernelWidth(self):
        return self.__KernelWidth

    #-----------------------------------------------------------------
    def gPaddingLeft(self):
        return self.__PaddingLeft

    #-----------------------------------------------------------------
    def gPaddingRight(self):
        return self.__PaddingRight

    #-----------------------------------------------------------------
    def gPaddingBottom(self):
        return self.__PaddingBottom

    #-----------------------------------------------------------------
    def gPaddingTop(self):
        return self.__PaddingTop

    #-----------------------------------------------------------------
    def qSubSampleLayer(self):
        return True

    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gPrevLayer(0).gOfmapWidth()  // self.gStrideLR() == self.gOfmapWidth())
        assert(self.gPrevLayer(0).gOfmapHeight() // self.gStrideBT() == self.gOfmapHeight())

