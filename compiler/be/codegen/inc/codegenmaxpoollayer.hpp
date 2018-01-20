#pragma once

#ifndef KCC_CODEGEN_CODEGENMAXPOOLLAYER_H
#define KCC_CODEGEN_CODEGENMAXPOOLLAYER_H

#include <cstdio>

#include "codegenpoollayer.hpp"

namespace kcc {
namespace layers {
    class Layer;
    class MaxPoolLayer;
}

namespace codegen {
using layers::Layer;
using layers::MaxPoolLayer;


class CodeGenMaxPoolLayer : public CodeGenPoolLayer {
public:
    //----------------------------------------------------------------
    CodeGenMaxPoolLayer(CodeGen* codegen)
        : CodeGenPoolLayer(codegen)
    {}

    //----------------------------------------------------------------
    void generate(Layer* layer) override;

};

}}

#endif // KCC_CODEGEN_CODEGENMAXPOOLLAYER_H


