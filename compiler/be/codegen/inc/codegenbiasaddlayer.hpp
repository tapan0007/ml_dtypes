#pragma once

#ifndef KCC_CODEGEN_CODEGENBIASADDLAYER_H
#define KCC_CODEGEN_CODEGENBIASADDLAYER_H

#include "tpb_isa_activate.hpp"

#include "codegenaddlayer.hpp"

namespace kcc {
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



