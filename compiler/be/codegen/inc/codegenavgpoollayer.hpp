#pragma once

#ifndef KCC_CODEGEN_CODEGENAVGPOOLLAYER_H
#define KCC_CODEGEN_CODEGENAVGPOOLLAYER_H

#include <cstdio>

#include "codegen/inc/codegenpoollayer.hpp"

namespace kcc {

namespace layers {
    class Layer;
}

namespace codegen {


class CodeGenAvgPoolLayer : public CodeGenPoolLayer {
public:
    //----------------------------------------------------------------
    CodeGenAvgPoolLayer(CodeGen* codegen)
        : CodeGenPoolLayer(codegen)
    {}

    //----------------------------------------------------------------
    void generate(layers::Layer* layer) override;

};

}}

#endif // KCC_CODEGEN_CODEGENAVGPOOLLAYER_H



