#pragma once

#ifndef KCC_LAYERS_ADDLAYER_H
#define KCC_LAYERS_ADDLAYER_H


#include "arithmeticlayer.hpp"
//#include "network.hpp"

namespace kcc {
namespace layers {

//########################################################
class AddLayer : public ArithmeticLayer {
private:
    using SubClass = ArithmeticLayer;
public:
    //----------------------------------------------------------------
    // Represents input FMAP. The input FMAP is stored as the *OUTPUT*
    // of the input layer.
//--------------------------------------------------------
    AddLayer(const Params& param, const FmapDesc& fmap_desc,
                  const std::vector<Layer*>& prevLayers)
        : SubClass(param, fmap_desc, prevLayers)
    {
    }

    //----------------------------------------------------------------
    std::string gString() const override {
        std::string baseLayer = gBaseLayerStr();
        return (gName() + baseLayer
               + gStateSizesStr());
    }


    //----------------------------------------------------------------
    bool qPassThrough() const override {
        return false;
    }


    //----------------------------------------------------------------
    bool qAddLayer() const override {
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

#endif // KCC_LAYERS_ADDLAYER_H




