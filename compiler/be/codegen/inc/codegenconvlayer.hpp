#pragma once

#ifndef KCC_CODEGEN_CODEGENCONVLAYER_H
#define KCC_CODEGEN_CODEGENCONVLAYER_H

#include <cstdio>

#include "codegenlayer.hpp"

namespace kcc {
namespace layers {
    class Layer;
}

namespace codegen {

class CodeGenConvLayer : public CodeGenLayer {
public:
    //----------------------------------------------------------------
    CodeGenConvLayer(CodeGen* codegen)
        : CodeGenLayer(codegen)
    {}

    //----------------------------------------------------------------
    void generate(layers::Layer* layer) override;

private:
    kcc_int32 m_FilterIndex_M = FilterIndex_M; // filter num ofmaps",
    kcc_int32 m_FilterIndex_C = FilterIndex_C; // filter num ifmaps",
    kcc_int32 m_FilterIndex_R = FilterIndex_R; // filter height",
    kcc_int32 m_FilterIndex_S = FilterIndex_S; // filter width",

    addr_t   m_FilterAddr[2];
    uint64_t m_FilterDims[FMAP_TENSOR_RANK];
    std::string   m_FilterFormat;
    std::string   m_FilterFileNames[2]; // more when #ofmaps > #cols

    uint8_t  m_ConvolveStride[2];
};

}}

#endif // KCC_CODEGEN_CODEGENCONVLAYER_H

