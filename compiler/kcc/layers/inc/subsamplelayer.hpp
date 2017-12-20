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
    kcc_int32 gStrideLR() const {
        return m_StrideLR;
    }

    //----------------------------------------------------------------
    // Vertical (Bottom-to-Top) stride
    kcc_int32 gStrideBT() const {
        return m_StrideBT;
    }

    //----------------------------------------------------------------
    kcc_int32 gKernelHeight() const {
        return m_KernelHeight;
    }

    //----------------------------------------------------------------
    kcc_int32 gKernelWidth() const {
        return m_KernelWidth;
    }

    //----------------------------------------------------------------
    kcc_int32 gPaddingLeft() const {
        return m_PaddingLeft;
    }

    //----------------------------------------------------------------
    kcc_int32 gPaddingRight() const {
        return m_PaddingRight;
    }

    //----------------------------------------------------------------
    kcc_int32 gPaddingBottom() const {
        return m_PaddingBottom;
    }

    //----------------------------------------------------------------
    kcc_int32 gPaddingTop() const {
        return m_PaddingTop;
    }

    //----------------------------------------------------------------
    bool qSubSampleLayer() const  override{
        return true;
    }

    //----------------------------------------------------------------
    bool verify() const override;

private:
    kcc_int32 m_StrideLR;
    kcc_int32 m_StrideBT;

    kcc_int32 m_KernelHeight;
    kcc_int32 m_KernelWidth;

    kcc_int32 m_PaddingLeft;
    kcc_int32 m_PaddingRight;
    kcc_int32 m_PaddingBottom;
    kcc_int32 m_PaddingTop;
};


} // namespace layers
} // namespace kcc

#endif // KCC_LAYERS_SUBSAMPLELAYER_H

