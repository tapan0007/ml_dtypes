#pragma once

#ifndef KCC_LAYERS_DATALAYER_H
#define KCC_LAYERS_DATALAYER_H


#include "layers/inc/layer.hpp"

namespace kcc {
namespace layers {

//########################################################
class DataLayer : public Layer {
private:
    using SubClass = Layer;
public:
    //----------------------------------------------------------------
    // Represents input FMAP. The input FMAP is stored as the *OUTPUT*
    // of the input layer.
//--------------------------------------------------------
    DataLayer(const Params& param, const FmapDesc& fmap_desc)
        : SubClass(param, fmap_desc, std::vector<Layer*>())
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
    bool qDataLayer() const override {
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

#endif // KCC_LAYERS_DATALAYER_H


