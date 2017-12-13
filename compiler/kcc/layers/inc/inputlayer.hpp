#pragma once

#ifndef KCC_LAYERS_INPUTLAYER_H
#define KCC_LAYERS_INPUTLAYER_H


#include "layer.hpp"
//#include "network.hpp"

namespace kcc {
namespace layers {

//########################################################
class InputLayer : public Layer {
private:
    typedef Layer SubClass;
public:
    //----------------------------------------------------------------
    //TODO: remove default values for input data file name and tensor dimension meaning string
    InputLayer(const Params& param, const FmapDesc& fmap_desc, 
                  const char* inputDataFileName,
                  const string& dataTensorDimSemantics)
        : Layer(param, fmap_desc, dataTensorDimSemantics, vector<Layer*>())
    {
        m_RefFileName = inputDataFileName;
    }

    //----------------------------------------------------------------
    string gString() const override {
        string baseLayer = gBaseLayerStr();
        return (gName() + baseLayer
               + gStateSizesStr());
    }

    //----------------------------------------------------------------
    const string gInputDataFileName() const {
        return m_RefFileName;
    }

    //----------------------------------------------------------------
    static const char* gTypeStrStatic() {
        return utils::TypeStr_Input;
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
    long gNumberWeightsPerPartition() const override {
        return 0;
    }


private:
};

}}

#endif // KCC_LAYERS_INPUTLAYER_H

