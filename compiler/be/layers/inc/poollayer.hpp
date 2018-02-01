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
            const FmapDesc& fmapDesc,
            const std::tuple<kcc_int32,kcc_int32>& stride, const std::tuple<kcc_int32,kcc_int32>& kernel,
            const std::tuple<kcc_int32,kcc_int32,kcc_int32,kcc_int32>& padding)
        : SubClass(params, prev_layer,fmapDesc, stride, kernel, padding)
    { }

    //----------------------------------------------------------------
    bool verify() const override
    {
        assert(gNumPrevLayers() == 1 && "Pool layer: number of previous layers not 1");
        assert((gPrevLayer(0)->gOfmapWidth()  / gStrideLeftRight()) == gOfmapWidth() && "Pool layer: Ifmap width not multiple of stride");
        assert((gPrevLayer(0)->gOfmapHeight() / gStrideTopBottom()) == gOfmapHeight() && "Pool layer: Ifmap height not multiple of stride");
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

