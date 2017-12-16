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
using layers::Layer;

class CodeGenConvLayer : public CodeGenLayer {
public:
    //----------------------------------------------------------------
    CodeGenConvLayer(CodeGen* codegen)
        : CodeGenLayer(codegen)
    {}

    //----------------------------------------------------------------
    void generate(Layer* layer) override;

private:
    uint8_t  m_Convolve_stride[2];
};

}}

#endif // KCC_CODEGEN_CODEGENCONVLAYER_H

