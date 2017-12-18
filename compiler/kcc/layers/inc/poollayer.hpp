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
    typedef SubSampleLayer SubClass;
public:
    //----------------------------------------------------------------
    PoolLayer(const Params& params, Layer* prev_layer,
         kcc_int32 num_ofmaps, const string& dataTensorSemantics,
         const std::tuple<kcc_int32,kcc_int32>& stride, const std::tuple<kcc_int32,kcc_int32>& kernel)
        : SubSampleLayer(params, prev_layer,num_ofmaps, dataTensorSemantics, stride, kernel)
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

