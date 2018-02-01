#pragma once

#ifndef KCC_LAYERS_RESADDLAYER_H
#define KCC_LAYERS_RESADDLAYER_H


#include "addlayer.hpp"
//#include "network.hpp"

namespace kcc {
namespace layers {

//########################################################
class ResAddLayer : public AddLayer {
private:
    using SubClass = AddLayer;
public:
    //----------------------------------------------------------------
    // Represents input FMAP. The input FMAP is stored as the *OUTPUT*
    // of the input layer.
//--------------------------------------------------------
    ResAddLayer(const Params& param, const FmapDesc& fmap_desc,
                  const std::vector<Layer*>& prevLayers)
        : SubClass(param, fmap_desc, prevLayers)
    {
    }

    //----------------------------------------------------------------
    std::string gString() const override;

    //----------------------------------------------------------------
    static const char* gTypeStrStatic() {
        return TypeStr_ResAdd;
    }
    const char*  gTypeStr() const  override {
        return gTypeStrStatic();
    }



    //----------------------------------------------------------------
    bool qPassThrough() const override {
        return false;
    }


    //----------------------------------------------------------------
    bool qResAddLayer() const override {
        return true;
    }

    //----------------------------------------------------------------
    bool qStoreInSB() const {
        return true;
    }


    //----------------------------------------------------------------
    bool verify() const override;


private:
};

}}

#endif // KCC_LAYERS_RESADDLAYER_H



