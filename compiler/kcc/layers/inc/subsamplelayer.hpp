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
         int num_ofmaps, const string& dataTensorSemantics,
         const std::tuple<int,int>& stride, const std::tuple<int,int>& kernel);

private:
    SubSampleLayer() = delete;
    SubSampleLayer(const SubSampleLayer&) = delete;

    SubSampleLayer& operator= (const SubSampleLayer&) const = delete;


public:

    //----------------------------------------------------------------
    // Horizontal (Left-to-Right) stride
    int gStrideLR() const {
        return m_StrideLR;
    }

    //----------------------------------------------------------------
    // Vertical (Bottom-to-Top) stride
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
    bool qSubSampleLayer() const  override{
        return true;
    }

    //----------------------------------------------------------------
    bool verify() const override;

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

