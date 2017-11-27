from utils.consts    import *
from utils.fmapdesc import  OfmapDesc
from .layer          import Layer
from .subsamplelayer import SubSampleLayer
from .poollayer      import PoolLayer
from .activlayer     import ActivLayer
import nets.network

##########################################################
class ConvLayer(SubSampleLayer):
    filter_file_key   = "kernel_file" 
    kernel_format_key = "kernel_format"

    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, num_ofmaps, stride, kernel,
            filterFileName, filterTensorDimSemantics):
        assert(isinstance(param, Layer.Param))
        assert(isinstance(prev_layer, Layer))

        super().__init__(param, prev_layer,
            num_ofmaps=num_ofmaps, stride=stride, kernel=kernel)

        self.__FilterFileName = filterFileName
        self.__FilterTensorDimSemantics = filterTensorDimSemantics

    #-----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()
        y = {
            ConvLayer.filter_file_key   : self.__FilterFileName,
            ConvLayer.kernel_format_key : self.__FilterTensorDimSemantics 
        }
        r = self.combineJson( (x, y) )
        return r

    #-----------------------------------------------------------------
    @classmethod
    def constructFromJson(klass, layerDict, nn):
        layerName = Layer.gLayerNameFromJson(layerDict)
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, nn)
        num_ofmaps = ofmapDesc.gNumMaps()

        strideLR = SubSampleLayer.gStrideLRFromJson(layerDict, nn)
        strideBT = SubSampleLayer.gStrideBTFromJson(layerDict, nn)
        kernelH = SubSampleLayer.gKernelHeightFromJson(layerDict, nn)
        kernelW = SubSampleLayer.gKernelWeightFromJson(layerDict, nn)

        paddingLeft = SubSampleLayer.gPaddingLeftFromJson(layerDict, nn)
        paddingRight = SubSampleLayer.gPaddingRightFromJson(layerDict, nn)
        paddingTop = SubSampleLayer.gPaddingTopFromJson(layerDict, nn)
        paddingBottom = SubSampleLayer.gPaddingBottomFromJson(layerDict, nn)

        stride = (strideLR + strideBT) // 2
        kernel = (kernelH + kernelW) // 2

        filterFileName = layerDict[ConvLayer.filter_file_key]
        tensorSemantics = layerDict[ConvLayer.kernel_format_key]
        batch = 1

        param = Layer.Param(layerName, batch, nn)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, nn)
        assert isinstance(prevLayers, list) and len(prevLayers)==1

        layer = ConvLayer(param, prevLayers[0], num_ofmaps, stride, kernel,
                    filterFileName, tensorSemantics)
        return layer

    #-----------------------------------------------------------------
    def __str__(self):
        kw = str(self.gKernelWidth())
        kh = str(self.gKernelHeight())
        ss = str(self.gStrideBT()) + "/" + str(self.gStrideLR()) 
        baseLayer = self.gBaseLayerStr()
        return (self.gName()
                + baseLayer
                + ", kernel=" + kh + "x" + kw + ", stride=" + ss
                + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gNumPrevLayers() == 1)
        super().verify()
        prevLayer = self.gPrevLayer(0)
        prevMapWidth = prevLayer.gOfmapWidth()
        prevMapHeight = prevLayer.gOfmapHeight()
        ##assert(self.gPrevLayer(0).gOfmapWidth() // self.gStrideLR() == self.gOfmapWidth())
        ##assert(self.gPrevLayer(0).gOfmapHeight() // self.gStrideBT() == self.gOfmapHeight())

    #-----------------------------------------------------------------
    def gFilterFileName(self):
        return self.__FilterFileName

    #-----------------------------------------------------------------
    def gFilterTensorDimSemantics(self):
        return self.__FilterTensorDimSemantics

    #-----------------------------------------------------------------
    @classmethod
    def gTypeStr(klass):
        return "Conv"

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def gNumberWeights(self):
        assert(self.gNumPrevLayers() == 1)
        num_ifmaps = self.gPrevLayer(0).gNumOfmaps()
        return num_ifmaps *  self.gNumberWeightsPerPartition()

    #-----------------------------------------------------------------
    def gNumberWeightsPerPartition(self):
        assert(self.gNumPrevLayers() == 1)
        kw = self.gKernelWidth()
        kh = self.gKernelHeight()
        num_ofmaps = self.gNumOfmaps()
        return kw*kh * num_ofmaps

    #-----------------------------------------------------------------
    def qConvLayer(self):
        return True


