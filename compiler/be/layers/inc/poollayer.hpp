#pragma once

#ifndef KCC_LAYERS_POOLLAYER_H
#define KCC_LAYERS_POOLLAYER_H 1

#include "subsamplelayer.hpp"

namespace kcc {
namespace layers {

//--------------------------------------------------------
// Base class of various pooling layers.
//--------------------------------------------------------
class PoolLayer : public SubSampleLayer {
private:
    using SubClass = SubSampleLayer;
public:
    //----------------------------------------------------------------
    PoolLayer(const Params& params, Layer* prev_layer,
            const FmapDesc& fmapDesc, const string& dataTensorSemantics,
            const std::tuple<kcc_int32,kcc_int32>& stride, const std::tuple<kcc_int32,kcc_int32>& kernel,
            const std::tuple<kcc_int32,kcc_int32,kcc_int32,kcc_int32>& padding)
        : SubSampleLayer(params, prev_layer,fmapDesc, dataTensorSemantics, stride, kernel, padding)
    { }

    //----------------------------------------------------------------
    bool verify() const override
    {
        assert(gNumPrevLayers() == 1);
        return this->SubClass::verify();
    }


    //----------------------------------------------------------------
    bool qPassThrough() const override {
        return false;
    }

    //----------------------------------------------------------------
    std::string gPoolLayerStr() const;

    //----------------------------------------------------------------
    bool qPoolLayer() const override {
        return true;
    }
};

}}

#endif // KCC_LAYERS_POOLLAYER_H

