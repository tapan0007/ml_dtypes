#pragma once

#ifndef KCC_LAYERS_SUBSAMPLELAYER_H
#define KCC_LAYERS_SUBSAMPLELAYER_H

#include <tuple>

#include "types.hpp"
#include "datatype.hpp"
#include "fmapdesc.hpp"
#include "layer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

//--------------------------------------------------------
class SubSampleLayer : public Layer {
public:
    static const char* const kernel_key;
    static const char* const stride_key;
    static const char* const padding_key;

    static const char* const kernel_height_key;
    static const char* const kernel_width_key;
    static const char* const stride_lr_key;
    static const char* const stride_bt_key;
    static const char* const padding_left_key;
    static const char* const padding_right_key;
    static const char* const padding_top_key;
    static const char* const padding_bottom_key;

    //----------------------------------------------------------------
    SubSampleLayer (const Params& param, Layer* prev_layer,
         int num_ofmaps, const std::tuple<int,int>& stride, const std::tuple<int,int>& kernel);

#if 0
    #-----------------------------------------------------------------
    def gJson(self):

    #-----------------------------------------------------------------
    @classmethod
    def gStrideLRFromJson(cls, layerDict, nn):

    #-----------------------------------------------------------------
    @classmethod
    def gStrideBTFromJson(cls, layerDict, nn):

    #-----------------------------------------------------------------
    @classmethod
    def gKernelHeightFromJson(cls, layerDict, nn):

    #-----------------------------------------------------------------
    @classmethod
    def gKernelWeightFromJson(cls, layerDict, nn):


    #-----------------------------------------------------------------
    @classmethod
    def gPaddingLeftFromJson(cls, layerDict, nn):

    #-----------------------------------------------------------------
    @classmethod
    def gPaddingRightFromJson(cls, layerDict, nn):

    #-----------------------------------------------------------------
    @classmethod
    def gPaddingBottomFromJson(cls, layerDict, nn):

    #-----------------------------------------------------------------
    @classmethod
    def gPaddingTopFromJson(cls, layerDict, nn):
#endif


    //----------------------------------------------------------------
    int gStrideLR() const {
        return m_StrideLR;
    }

    //----------------------------------------------------------------
    int gStrideBT() const {
        return m_StrideBT;
    }

    //----------------------------------------------------------------
    int gKernelHeight() const {
        return m_KernelHeight;
    }

    //----------------------------------------------------------------
    int gKernelWidth() const {
        return m_KernelWidth;
    }

    //----------------------------------------------------------------
    int gPaddingLeft() const {
        return m_PaddingLeft;
    }

    //----------------------------------------------------------------
    int gPaddingRight() const {
        return m_PaddingRight;
    }

    //----------------------------------------------------------------
    int gPaddingBottom() const {
        return m_PaddingBottom;
    }

    //----------------------------------------------------------------
    int gPaddingTop() const {
        return m_PaddingTop;
    }

    //----------------------------------------------------------------
    bool qSubSampleLayer() const {
        return true;
    }

    //----------------------------------------------------------------
    bool verify() const;

private:
    int m_StrideLR;
    int m_StrideBT;

    int m_KernelHeight;
    int m_KernelWidth;

    int m_PaddingLeft;
    int m_PaddingRight;
    int m_PaddingBottom;
    int m_PaddingTop;
};


} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_SUBSAMPLELAYER_H

