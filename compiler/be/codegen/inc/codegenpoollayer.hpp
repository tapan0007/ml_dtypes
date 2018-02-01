#pragma once

#ifndef KCC_CODEGEN_CODEGENPOOLLAYER_H
#define KCC_CODEGEN_CODEGENPOOLLAYER_H

#include <cstdio>

#include "codegenlayer.hpp"

namespace kcc {
namespace layers {
    class PoolLayer;
}

namespace codegen {


class CodeGenPoolLayer : public CodeGenLayer {
public:
    //----------------------------------------------------------------
    CodeGenPoolLayer(CodeGen* codegen);

protected:
public:
    //----------------------------------------------------------------
    void Generate(layers::Layer* layer, POOLFUNC poolFunc);

protected:
    uint64_t m_PoolStride[4];
    uint64_t m_KernelDims[FMAP_TENSOR_RANK];
};

}}

#endif // KCC_CODEGEN_CODEGENPOOLLAYER_H


