#pragma once

#ifndef KCC_CODEGEN_CODEGENBIASADDLAYER_H
#define KCC_CODEGEN_CODEGENBIASADDLAYER_H

#include "aws_tonga_isa_tpb_activate.h"

#include "codegen/inc/codegenaddlayer.hpp"

namespace kcc {
namespace layers {
    class InputLayer;
    class ConstLayer;
}

namespace codegen {

//########################################################
class CodeGenBiasAddLayer : public CodeGenAddLayer {
private:
    using SubClass = CodeGenAddLayer;
public:
    //----------------------------------------------------------------
    CodeGenBiasAddLayer(CodeGen* codegen)
        : SubClass(codegen)
    {}

    void generate(layers::Layer* layer) override;
};

}}

#endif // KCC_CODEGEN_CODEGENBIASADDLAYER_H



