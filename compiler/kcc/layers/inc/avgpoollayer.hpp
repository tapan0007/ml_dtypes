#pragma once

#ifndef KCC_LAYERS_AVGPOOLLAYER_H
#define KCC_LAYERS_AVGPOOLLAYER_H 1

#include "poollayer.hpp"

namespace kcc {
namespace layers {

//########################################################
class AvgPoolLayer : public PoolLayer {
private:
    typedef PoolLayer SubClass;
public:
    //----------------------------------------------------------------
    AvgPoolLayer(const Params& params, Layer* prev_layer,
            int num_ofmaps, const string& dataTensorSemantics,
            const std::tuple<int,int>& stride, const std::tuple<int,int>& kernel)
        : PoolLayer(params, prev_layer, num_ofmaps, dataTensorSemantics, stride, kernel)
    {}

    //----------------------------------------------------------------
    string gString() const override {
        return gPoolLayerStr();
    }

    //----------------------------------------------------------------
    bool verify() const override;


    //----------------------------------------------------------------
    static const char* TypeStr() {
        return "AvgPool";
    }

    //----------------------------------------------------------------
    const char* gTypeStr() const override {
        return TypeStr();
    }


    //----------------------------------------------------------------
    bool qAvgPoolLayer() const override {
        return true;
    }
};

}}


#endif // KCC_LAYERS_AVGPOOLLAYER_H

