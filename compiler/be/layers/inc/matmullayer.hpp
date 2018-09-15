#pragma once

#ifndef KCC_LAYERS_MATMULLAYER_H
#define KCC_LAYERS_MATMULLAYER_H

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "layers/inc/layerconsts.hpp"
#include "layers/inc/layer.hpp"

namespace kcc {
namespace layers {


class MatmulLayer : public Layer {
private:
    using SubClass = Layer;

public:
    class Params;

    MatmulLayer(const MatmulLayer::Params& params, Layer* prev_layer, const FmapDesc& fmapDesc,
        const std::tuple<kcc_int32,kcc_int32>& kernel,
        const char* filterFileName, const char* filterTensorDimSemantics);

    //----------------------------------------------------------------
    std::string gString() const override;


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
        return LayerTypeStr_Matmul;
    }
    virtual const char* gTypeStr() const override;

    //----------------------------------------------------------------
    bool qMatmulLayer() const override {
        return true;
    }

    bool verify() const override;

    //----------------------------------------------------------------
    kcc_int32 gKernelHeight() const {
        return m_Kernel.m_Height;
    }

    //----------------------------------------------------------------
    kcc_int32 gKernelWidth() const {
        return m_Kernel.m_Width;
    }

private:
    struct Kernel {
        kcc_int32           m_Height;
        kcc_int32           m_Width;
    };
    Kernel                  m_Kernel;
    std::string             m_FilterFileName;
    std::string             m_FilterTensorFormat;
};


class MatmulLayer::Params : public Layer::Params {
public:
    Params(const Layer::Params& params);
    bool verify() const;
};


}}


#endif
