#pragma once

#ifndef KCC_LAYERS_CONVLAYER_H
#define KCC_LAYERS_CONVLAYER_H

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "layers/inc/subsamplelayer.hpp"


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
        const std::tuple<kcc_int32,kcc_int32>& stride, const std::tuple<kcc_int32,kcc_int32>& kernel,
        const std::tuple<kcc_int32,kcc_int32,kcc_int32,kcc_int32>& padding,
        const char* filterFileName, const char* filterTensorDimSemantics);

    //----------------------------------------------------------------
    std::string gString() const override;

    //----------------------------------------------------------------
    bool verify() const override;

    //----------------------------------------------------------------
    // Numpy file where the filter weights are stored (for Inkling simulator).
    //----------------------------------------------------------------
    std::string gFilterFileName() const {
        return m_FilterFileName;
    }

    //----------------------------------------------------------------
    const std::string& gFilterTensorDimSemantics() const {
        return m_FilterTensorFormat;
    }

    const std::string& gFilterTensorFormat() const {
        return m_FilterTensorFormat;
    }

    //----------------------------------------------------------------
    static const char* TypeStr() {
        return LayerTypeStr_Conv;
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
    kcc_int32 gNumberWeights() const;

    //----------------------------------------------------------------
    kcc_int64 gNumberWeightsPerPartition() const override;

    //----------------------------------------------------------------
    bool qConvLayer() const override {
        return true;
    }

private:
    std::string              m_FilterFileName;
    std::string              m_FilterTensorFormat;
    StateBufferAddress  m_WeightAddress;
};

} // namespace layers
} // namespace kcc


#endif // KCC_LAYERS_CONVLAYER_H

