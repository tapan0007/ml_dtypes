#pragma once

#ifndef KCC_LAYERS_INPUTLAYER_H
#define KCC_LAYERS_INPUTLAYER_H


#include "layer.hpp"
//#include "network.hpp"

namespace kcc {
namespace layers {

//########################################################
class InputLayer : public Layer {
public:
    //----------------------------------------------------------------
    //TODO: remove default values for input data file name and tensor dimension meaning string
    InputLayer(const Params& param, const FmapDesc& fmap_desc, 
                  const char* inputDataFileName,
                  const string& dataTensorDimSemantics)
        : Layer(param, fmap_desc, dataTensorDimSemantics, vector<Layer*>())
    {
        m_InputDataFileName = inputDataFileName;
    }

    //----------------------------------------------------------------
    string gString() const override {
        string baseLayer = gBaseLayerStr();
        return (gName() + baseLayer
               + gStateSizesStr());
    }

    //----------------------------------------------------------------
    const string gInputDataFileName() const {
        return m_InputDataFileName;
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


#if 0
    #-----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()
        y = {
            Layer.ref_file_key        : self.__InputDataFileName,
            InputLayer.input_dims_key  : self.__DataTensorDimSemantics
        }
        r = self.combineJson( (x, y) )
        return r

    #-----------------------------------------------------------------
    @classmethod
    def constructFromJson(cls, layerDict, nn):
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, nn)
        layerName = Layer.gLayerNameFromJson(layerDict)

        inputFileName = layerDict[Layer.ref_file_key]
        tensorSemantics = layerDict[InputLayer.input_dims_key]
        batch = 1

        param = Layer.Param(layerName, batch, nn)
        layer = InputLayer(param, ofmapDesc, inputFileName, tensorSemantics)
        return layer
#endif

private:
    string m_InputDataFileName;
};

}}

#endif // KCC_LAYERS_INPUTLAYER_H

