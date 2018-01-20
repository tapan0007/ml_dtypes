#pragma once

#ifndef KCC_LAYERS_MAXPOOLLAYER_H
#define KCC_LAYERS_MAXPOOLLAYER_H 1

#include "poollayer.hpp"

namespace kcc {
namespace layers {


//--------------------------------------------------------
// Max pooling layer
//--------------------------------------------------------
class MaxPoolLayer : public PoolLayer {
private:
    using SubClass = PoolLayer;
public:
    //----------------------------------------------------------------
    MaxPoolLayer(const Params& params,
                 Layer* prev_layer,
                 const FmapDesc& fmapDesc,
                 const string& dataTensorSemantics,
                 const std::tuple<kcc_int32,kcc_int32>& stride,
                 const std::tuple<kcc_int32,kcc_int32>& kernel,
                 const std::tuple<kcc_int32,kcc_int32,kcc_int32,kcc_int32>& padding)
        : PoolLayer(params, prev_layer, fmapDesc, dataTensorSemantics, stride, kernel, padding)
    {}

    //----------------------------------------------------------------
    string gString() const override {
        return gPoolLayerStr();
    }

    //----------------------------------------------------------------
    bool verify() const override;


    //----------------------------------------------------------------
    static const char* TypeStr() {
        return "MaxPool";
    }

    //----------------------------------------------------------------
    const char* gTypeStr() const override {
        return TypeStr();
    }


    //----------------------------------------------------------------
    bool qMaxPoolLayer() const override {
        return true;
    }
};


}}


#endif // KCC_LAYERS_MAXPOOLLAYER_H

