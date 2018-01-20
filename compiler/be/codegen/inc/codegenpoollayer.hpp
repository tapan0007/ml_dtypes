#pragma once

#ifndef KCC_CODEGEN_CODEGENPOOLLAYER_H
#define KCC_CODEGEN_CODEGENPOOLLAYER_H

#include <cstdio>

#include "codegenlayer.hpp"

namespace kcc {
namespace layers {
    class Layer;
    class PoolLayer;
}

namespace codegen {
using layers::Layer;
using layers::PoolLayer;


class CodeGenPoolLayer : public CodeGenLayer {
public:
    //----------------------------------------------------------------
    CodeGenPoolLayer(CodeGen* codegen);

    //----------------------------------------------------------------

protected:
    uint64_t m_PoolStride[4];
    uint64_t m_KernelDims[FMAP_TENSOR_RANK];
};

}}

#endif // KCC_CODEGEN_CODEGENPOOLLAYER_H



