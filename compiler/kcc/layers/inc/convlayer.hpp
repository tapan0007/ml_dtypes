#pragma once

#ifndef KCC_LAYERS_CONVLAYER_H
#define KCC_LAYERS_CONVLAYER_H

#include "consts.hpp"
#include "types.hpp"
#include "datatype.hpp"
#include "fmapdesc.hpp"
#include "subsamplelayer.hpp"


namespace kcc {

namespace nets {
    class Network;
}

namespace layers {

//--------------------------------------------------------
class ConvLayer : public SubSampleLayer {
public:

    //----------------------------------------------------------------
    ConvLayer(const Params& params, Layer* prev_layer, int num_ofmaps,
        const string& dataTensorSemantics,
        const std::tuple<int,int>& stride, const std::tuple<int,int>& kernel,
        const char* filterFileName, const char* filterTensorDimSemantics);

    //----------------------------------------------------------------
    string gString() const;

    //----------------------------------------------------------------
    bool verify() const;

    //----------------------------------------------------------------
    string gFilterFileName() const {
        return m_FilterFileName;
    }

    //----------------------------------------------------------------
    string gFilterTensorDimSemantics() const {
        return m_FilterTensorDimSemantics;
    }

    static const char* TypeStr() {
        return utils::TypeStr_Conv;
    }

    //----------------------------------------------------------------
    const char* gTypeStr() const {
        return TypeStr();
    }

    //----------------------------------------------------------------
    bool qPassThrough() const override {
        return false;
    }

    //----------------------------------------------------------------
    int gNumberWeights() const;

    //----------------------------------------------------------------
    long gNumberWeightsPerPartition() const;

    //----------------------------------------------------------------
    bool qConvLayer() const override {
        return true;
    }

private:
    string m_FilterFileName;
    string m_FilterTensorDimSemantics;
};

} // namespace layers
} // namespace kcc


#endif // KCC_LAYERS_CONVLAYER_H

