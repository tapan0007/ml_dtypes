#include <sstream>

#include "convlayer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

//--------------------------------------------------------
const char* const
ConvLayer::filter_file_key   = "kernel_file";

const char* const
ConvLayer::kernel_format_key = "kernel_format";

//--------------------------------------------------------
ConvLayer::ConvLayer(const Params& params, Layer* prev_layer, int num_ofmaps,
        const std::tuple<int,int>& stride, const std::tuple<int,int>& kernel,
        const char* filterFileName, const char* filterTensorDimSemantics)
    : SubSampleLayer(params, prev_layer, num_ofmaps, stride, kernel)
{
        m_FilterFileName = filterFileName;
        m_FilterTensorDimSemantics = filterTensorDimSemantics;
}

//--------------------------------------------------------
string
ConvLayer::gString() const
{
    std::stringstream ss;
    const int kh = gKernelHeight();
    const int kw = gKernelWidth();
    ss << gName() << gBaseLayerStr() << ", kernel=" << kh << "x" << kw
       << ",stride=" << gStrideBT() << "/" << gStrideLR() << gStateSizesStr();
    return ss.str();
}

//--------------------------------------------------------
bool
ConvLayer::verify() const
{
    assert(gNumPrevLayers() == 1);
    const bool ok = this->SubSampleLayer::verify();
    const Layer* prevLayer = gPrevLayer(0);
    const int prevMapWidth = prevLayer->gOfmapWidth();
    const int prevMapHeight = prevLayer->gOfmapHeight();
    assert(prevMapWidth ==  gStrideLR() * gOfmapWidth());
    assert(prevMapHeight == gStrideBT() * gOfmapHeight());
    return ok;
}

//--------------------------------------------------------
int 
ConvLayer::gNumberWeights() const
{
    assert(gNumPrevLayers() == 1);
    const int num_ifmaps = gPrevLayer(0)->gNumOfmaps();
    return num_ifmaps * gNumberWeightsPerPartition();
}

//--------------------------------------------------------
long
ConvLayer::gNumberWeightsPerPartition() const
{
    assert(gNumPrevLayers() == 1);
    const int kw = gKernelWidth();
    const int kh = gKernelHeight();
    const int num_ofmaps = gNumOfmaps();
    return kw*kh * num_ofmaps;
}


#if 0
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
    def constructFromJson(cls, layerDict, nn):
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

        stride = (strideLR + strideBT) / 2
        kernel = (kernelH + kernelW) / 2

        filterFileName = layerDict[ConvLayer.filter_file_key]
        tensorSemantics = layerDict[ConvLayer.kernel_format_key]
        batch = 1

        param = Layer.Param(layerName, batch, nn)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, nn)
        assert isinstance(prevLayers, list) and len(prevLayers)==1

        layer = ConvLayer(param, prevLayers[0], num_ofmaps, stride, kernel,
                    filterFileName, tensorSemantics)
        return layer
#endif

} // namespace layers
} // namespace kcc


