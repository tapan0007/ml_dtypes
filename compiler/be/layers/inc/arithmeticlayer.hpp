#pragma once

#ifndef KCC_LAYERS_ARITHMETICLAYER_H
#define KCC_LAYERS_ARITHMETICLAYER_H


#include "layer.hpp"
//#include "network.hpp"

namespace kcc {
namespace layers {

//########################################################
class ArithmeticLayer : public Layer {
private:
    using SubClass = Layer;
public:
    //----------------------------------------------------------------
    // Represents input FMAP. The input FMAP is stored as the *OUTPUT*
    // of the input layer.
//--------------------------------------------------------
    ArithmeticLayer(const Params& param, const FmapDesc& fmap_desc,
                  const std::vector<Layer*>& prevLayers)
        : SubClass(param, fmap_desc, prevLayers)
    {
    }

    //----------------------------------------------------------------
    string gString() const override {
        string baseLayer = gBaseLayerStr();
        return (gName() + baseLayer
               + gStateSizesStr());
    }


    //----------------------------------------------------------------
    bool qPassThrough() const override {
        return false;
    }


    //----------------------------------------------------------------
    bool qArithmeticLayer() const override {
        return true;
    }

    //----------------------------------------------------------------
    bool qStoreInSB() const {
        return true;
    }

    //----------------------------------------------------------------


private:
};

}}

#endif // KCC_LAYERS_ARITHMETICLAYER_H



