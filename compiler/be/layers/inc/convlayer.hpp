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
// Convolution layer.
//--------------------------------------------------------
class ConvLayer : public SubSampleLayer {
private:
    using SubClass = SubSampleLayer;
public:

    //----------------------------------------------------------------
    ConvLayer(const Params& params, Layer* prev_layer, const FmapDesc& fmapDesc,
        const string& dataTensorSemantics,
        const std::tuple<kcc_int32,kcc_int32>& stride, const std::tuple<kcc_int32,kcc_int32>& kernel,
        const std::tuple<kcc_int32,kcc_int32,kcc_int32,kcc_int32>& padding,
        const char* filterFileName, const char* filterTensorDimSemantics);

    //----------------------------------------------------------------
    string gString() const override;

    //----------------------------------------------------------------
    bool verify() const override;

    //----------------------------------------------------------------
    // Numpy file where the filter weights are stored (for Inkling simulator).
    //----------------------------------------------------------------
    string gFilterFileName() const {
        return m_FilterFileName;
    }

    //----------------------------------------------------------------
    // AKA filter format
    //----------------------------------------------------------------
    string gFilterTensorDimSemantics() const {
        return m_FilterTensorDimSemantics;
    }

    //----------------------------------------------------------------
    static const char* TypeStr() {
        return TypeStr_Conv;
    }

    //----------------------------------------------------------------
    const char* gTypeStr() const override {
        return TypeStr();
    }

    //----------------------------------------------------------------
    bool qPassThrough() const override {
        return false;
    }

    StateBufferAddress gWeightAddress() const {
        return m_WeightAddress;
    }

    void rWeightAddress(StateBufferAddress address) {
        m_WeightAddress = address;
    }

    //----------------------------------------------------------------
    // ConvLayer must override this method with correct values
    StateBufferAddress gNumberWeightsPerPartition();

    //----------------------------------------------------------------
    kcc_int32 gNumberWeights() const;

    //----------------------------------------------------------------
    kcc_int64 gNumberWeightsPerPartition() const override;

    //----------------------------------------------------------------
    bool qConvLayer() const override {
        return true;
    }

private:
    string              m_FilterFileName;
    string              m_FilterTensorDimSemantics;
    StateBufferAddress  m_WeightAddress;
};

} // namespace layers
} // namespace kcc


#endif // KCC_LAYERS_CONVLAYER_H

