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
// Base layer for all layers that can stride or dilate.
// Examples: Conv, Pool
//--------------------------------------------------------
class SubSampleLayer : public Layer {
private:
    typedef Layer SubClass;
public:

    //----------------------------------------------------------------
    SubSampleLayer (const Params& param, Layer* prev_layer,
         kcc_int32 num_ofmaps, const string& dataTensorSemantics,
         const std::tuple<kcc_int32,kcc_int32>& stride,
         const std::tuple<kcc_int32,kcc_int32>& kernel,
         const std::tuple<kcc_int32,kcc_int32,kcc_int32,kcc_int32>& padding);

private:
    SubSampleLayer() = delete;
    SubSampleLayer(const SubSampleLayer&) = delete;

    SubSampleLayer& operator= (const SubSampleLayer&) const = delete;


public:

    //----------------------------------------------------------------
    // Horizontal (Left-to-Right) stride
    kcc_int32 gStrideLeftRight() const {
        return m_Stride.m_LeftRight;
    }

    //----------------------------------------------------------------
    // Vertical (Bottom-to-Top) stride
    kcc_int32 gStrideTopBottom() const {
        return m_Stride.m_TopBottom;
    }

    //----------------------------------------------------------------
    kcc_int32 gKernelHeight() const {
        return m_Kernel.m_Height;
    }

    //----------------------------------------------------------------
    kcc_int32 gKernelWidth() const {
        return m_Kernel.m_Width;
    }

    //----------------------------------------------------------------
    kcc_int32 gPaddingLeft() const {
        return m_Padding.m_Left;
    }

    //----------------------------------------------------------------
    kcc_int32 gPaddingRight() const {
        return m_Padding.m_Right;
    }

    //----------------------------------------------------------------
    kcc_int32 gPaddingBottom() const {
        return m_Padding.m_Bottom;
    }

    //----------------------------------------------------------------
    kcc_int32 gPaddingTop() const {
        return m_Padding.m_Top;
    }

    //----------------------------------------------------------------
    bool qSubSampleLayer() const  override{
        return true;
    }

    //----------------------------------------------------------------
    bool verify() const override;

private:
    struct Stride {
        kcc_int32 m_TopBottom;
        kcc_int32 m_LeftRight;
    };
    Stride m_Stride;

    struct Kernel {
        kcc_int32 m_Height;
        kcc_int32 m_Width;
    };
    Kernel m_Kernel;

    struct Padding {
        kcc_int32 m_Top;
        kcc_int32 m_Bottom;
        kcc_int32 m_Left;
        kcc_int32 m_Right;
    };
    Padding m_Padding;
};


} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_SUBSAMPLELAYER_H

