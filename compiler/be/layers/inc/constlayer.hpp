#pragma once

#ifndef KCC_LAYERS_CONSTLAYER_H
#define KCC_LAYERS_CONSTLAYER_H


#include "layers/inc/datalayer.hpp"

namespace kcc {
namespace layers {

//########################################################
class ConstLayer : public DataLayer {
private:
    using SubClass = DataLayer;
public:
    //----------------------------------------------------------------
    // Represents input FMAP. The input FMAP is stored as the *OUTPUT*
    // of the input layer.
//--------------------------------------------------------
    ConstLayer(const Params& param, const FmapDesc& fmap_desc)
        : SubClass(param, fmap_desc)
    { }

    //----------------------------------------------------------------
    std::string gString() const override;


    //----------------------------------------------------------------
    static const char* gTypeStrStatic() {
        return LayerTypeStr_Const;
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
    bool qConstLayer() const override {
        return true;
    }

    //----------------------------------------------------------------
    bool qStoreInSB() const {
        return true;
    }

    //----------------------------------------------------------------
    kcc_int64 gNumberWeightsPerPartition() const override;

private:
};

}}

#endif // KCC_LAYERS_CONSTLAYER_H

