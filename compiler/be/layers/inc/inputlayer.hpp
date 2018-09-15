#pragma once

#ifndef KCC_LAYERS_INPUTLAYER_H
#define KCC_LAYERS_INPUTLAYER_H


#include "layers/inc/layerconsts.hpp"
#include "layers/inc/datalayer.hpp"

namespace kcc {
namespace layers {

//########################################################
class InputLayer : public DataLayer {
private:
    using SubClass = DataLayer;
public:
    //----------------------------------------------------------------
    // Represents input FMAP. The input FMAP is stored as the *OUTPUT*
    // of the input layer.
//--------------------------------------------------------
    InputLayer(const Params& param, const FmapDesc& fmap_desc)
        : SubClass(param, fmap_desc)
    { }

    //----------------------------------------------------------------
    std::string gString() const override;

    //----------------------------------------------------------------
    static const char* gTypeStrStatic() {
        return LayerTypeStr_Input;
    }
    const char*  gTypeStr() const  override {
        return gTypeStrStatic();
    }


    //----------------------------------------------------------------
    bool verify() const override {
        return true;
    }

    //----------------------------------------------------------------
    bool qPassThrough() const override {
        return false;
    }


    //----------------------------------------------------------------
    bool qInputLayer() const override {
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

#endif // KCC_LAYERS_INPUTLAYER_H

