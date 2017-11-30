#include <sstream>

#include "datatype.hpp"
#include "network.hpp"
#include "subsamplelayer.hpp"



namespace kcc {
namespace layers {

//--------------------------------------------------------
const char* const
SubSampleLayer::kernel_key         = "kernel_shape";
const char* const
SubSampleLayer::stride_key         = "stride";
const char* const
SubSampleLayer::padding_key        = "padding";

const char* const
SubSampleLayer::kernel_height_key  = "height";
const char* const
SubSampleLayer::kernel_width_key   = "width";
const char* const
SubSampleLayer::stride_lr_key      = "LR";
const char* const
SubSampleLayer::stride_bt_key      = "BT";
const char* const
SubSampleLayer::padding_left_key   = "left";
const char* const
SubSampleLayer::padding_right_key  = "right";
const char* const
SubSampleLayer::padding_top_key    = "top";
const char* const
SubSampleLayer::padding_bottom_key = "bottom";

//--------------------------------------------------------
SubSampleLayer::SubSampleLayer (const Params& param, Layer* prev_layer,
         int num_ofmaps, const std::tuple<int,int>& stride, const std::tuple<int,int>& kernel)
    : Layer(param, 
            FmapDesc(
                (num_ofmaps >= 1 ? num_ofmaps : prev_layer->gNumOfmaps()),
                prev_layer->gOfmapWidth(),
                prev_layer->gOfmapHeight()),
            vector<Layer*>(1, prev_layer))
{
    // Stride(2) is larger than kernel(1*1) for some layers in ResNet.
    // That seems to be a wrong way of aggregating information, but
    // it does happen.
    // assert(stride <= kernel)


    // Stride
    m_StrideBT = std::get<0>(stride);
    m_StrideLR = std::get<1>(stride);

    m_KernelHeight = std::get<0>(kernel);
    m_KernelWidth  = std::get<1>(kernel);

    // Padding
    const int kh = gKernelHeight();
    if (kh % 2 == 1) {
        m_PaddingBottom = m_PaddingTop = (kh - 1) / 2;
    } else {
        m_PaddingBottom = 0;
        m_PaddingTop    = kh / 2;
    }

    const int kw = gKernelWidth();
    if (kw % 2 == 1) {
        m_PaddingRight  = m_PaddingLeft  = (kw - 1) / 2;
    } else {
        m_PaddingLeft   = 0;
        m_PaddingRight  = kw / 2;
    }
}



#if 0
    #-----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()

        if self.gNetwork().gUseDimList():
            kernel = [1, 1, self.gKernelHeight(), self.gKernelWidth()]
            stride = [1, 1, self.gStrideBT(), self.gStrideLR()]
            padding = [ [0,0], [0,0], [self.gPaddingBottom(), self.gPaddingTop()], [self.gPaddingLeft(), self.gPaddingRight()]]
        else:
            kernel  = {
                SubSampleLayer.kernel_height_key : self.gKernelHeight(),
                SubSampleLayer.kernel_width_key  : self.gKernelWidth()
            }
            padding = {
                SubSampleLayer.padding_top_key    : self.gPaddingTop(),
                SubSampleLayer.padding_bottom_key : self.gPaddingBottom(),
                SubSampleLayer.padding_left_key   : self.gPaddingLeft(),
                SubSampleLayer.padding_right_key  : self.gPaddingRight()
            }
            stride = {
                SubSampleLayer.stride_bt_key : self.gStrideBT(),
                SubSampleLayer.stride_lr_key : self.gStrideLR()
            }

        y = {
            SubSampleLayer.kernel_key : kernel,
            SubSampleLayer.stride_key : stride,
            SubSampleLayer.padding_key : padding
        }
        r = self.combineJson( (x, y) )
        return r




    #-----------------------------------------------------------------
    @classmethod
    def gStrideLRFromJson(cls, layerDict, nn):
        stride = layerDict[SubSampleLayer.stride_key]
        if nn.gUseDimList():
            return stride[3]
        else:
            return stride[SubSampleLayer.stride_lr_key]

    #-----------------------------------------------------------------
    @classmethod
    def gStrideBTFromJson(cls, layerDict, nn):
        stride = layerDict[SubSampleLayer.stride_key]
        if nn.gUseDimList():
            return stride[2]
        else:
            return stride[SubSampleLayer.stride_bt_key]

    #-----------------------------------------------------------------
    @classmethod
    def gKernelHeightFromJson(cls, layerDict, nn):
        kernel = layerDict[SubSampleLayer.kernel_key]
        if nn.gUseDimList():
            return kernel[2]
        else:
            return kernel[SubSampleLayer.kernel_height_key]

    #-----------------------------------------------------------------
    @classmethod
    def gKernelWeightFromJson(cls, layerDict, nn):
        kernel = layerDict[SubSampleLayer.kernel_key]
        if nn.gUseDimList():
            return kernel[3]
        else:
            return kernel[SubSampleLayer.kernel_weight_key]


    #-----------------------------------------------------------------
    @classmethod
    def gPaddingLeftFromJson(cls, layerDict, nn):
        padding = layerDict[SubSampleLayer.padding_key]
        if nn.gUseDimList():
            return padding[3][0]
        else:
            return padding[SubSampleLayer.padding_left_key]


    #-----------------------------------------------------------------
    @classmethod
    def gPaddingRightFromJson(cls, layerDict, nn):
        padding = layerDict[SubSampleLayer.padding_key]
        if nn.gUseDimList():
            return padding[3][1]
        else:
            return padding[SubSampleLayer.padding_right_key]

    #-----------------------------------------------------------------
    @classmethod
    def gPaddingBottomFromJson(cls, layerDict, nn):
        padding = layerDict[SubSampleLayer.padding_key]
        if nn.gUseDimList():
            return padding[2][0]
        else:
            return padding[SubSampleLayer.padding_bottom_key]

    #-----------------------------------------------------------------
    @classmethod
    def gPaddingTopFromJson(cls, layerDict, nn):
        padding = layerDict[SubSampleLayer.padding_key]
        if nn.gUseDimList():
            return padding[2][1]
        else:
            return padding[SubSampleLayer.padding_top_key]
#endif


//--------------------------------------------------------
bool
SubSampleLayer::verify() const
{
    assert(gPrevLayer(0)->gOfmapWidth()  == gStrideLR() * gOfmapWidth());
    assert(gPrevLayer(0)->gOfmapHeight() == gStrideBT() * gOfmapHeight());
    return true;
}

} // namespace layers
} // namespace kcc

