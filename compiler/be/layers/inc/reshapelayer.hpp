#pragma once

#ifndef KCC_LAYERS_RESHAPELAYER_H
#define KCC_LAYERS_RESHAPELAYER_H

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "layers/inc/layer.hpp"

namespace kcc {
namespace layers {


class ReshapeLayer : public Layer {
private:
    using SubClass = Layer;

public:
    class Params;

    ReshapeLayer(const ReshapeLayer::Params& params, Layer* prev_layer, const FmapDesc& fmapDesc);

    //----------------------------------------------------------------
    std::string gString() const override;



    //----------------------------------------------------------------
    static const char* TypeStr() {
        return LayerTypeStr_Reshape;
    }
    const char* gTypeStr() const override;

    //----------------------------------------------------------------
    bool qReshapeLayer() const override {
        return true;
    }

    bool verify() const override;
};


class ReshapeLayer::Params : public Layer::Params {
public:
    Params(const Layer::Params& params);
    bool verify() const;
};


}}


#endif
