#pragma once

#ifndef KCC_CODEGEN_CODEGENMAXPOOLLAYER_H
#define KCC_CODEGEN_CODEGENMAXPOOLLAYER_H

#include <cstdio>

#include "codegen/inc/codegenpoollayer.hpp"

namespace kcc {
namespace layers {
    class Layer;
}

namespace codegen {


class CodeGenMaxPoolLayer : public CodeGenPoolLayer {
public:
    //----------------------------------------------------------------
    CodeGenMaxPoolLayer(CodeGen* codegen)
        : CodeGenPoolLayer(codegen)
    {}

    //----------------------------------------------------------------
    void generate(layers::Layer* layer) override;

};

}}

#endif // KCC_CODEGEN_CODEGENMAXPOOLLAYER_H


