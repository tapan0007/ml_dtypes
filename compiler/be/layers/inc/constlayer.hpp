#pragma once

#ifndef KCC_LAYERS_CONSTLAYER_H
#define KCC_LAYERS_CONSTLAYER_H


#include "datalayer.hpp"

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
    {
        assert(fmap_desc.gMapWidth() == 1 && "Const layer must have width == 1");
        assert(fmap_desc.gMapHeight() == 1 && "Const layer must have height == 1");
    }

    //----------------------------------------------------------------
    std::string gString() const override;


    //----------------------------------------------------------------
    static const char* gTypeStrStatic() {
        return TypeStr_Const;
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

