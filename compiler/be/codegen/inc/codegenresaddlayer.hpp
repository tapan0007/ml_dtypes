#pragma once

#ifndef KCC_CODEGEN_CODEGENRESADDLAYER_H
#define KCC_CODEGEN_CODEGENRESADDLAYER_H

#include "tpb_isa_activate.hpp"

#include "codegenaddlayer.hpp"

namespace kcc {
namespace codegen {

//########################################################
class CodeGenResAddLayer : public CodeGenAddLayer {
private:
    using SubClass = CodeGenAddLayer;
public:
    //----------------------------------------------------------------
    CodeGenResAddLayer(CodeGen* codegen)
        : SubClass(codegen)
    {}

    void generate(layers::Layer* layer) override;
};

}}

#endif // KCC_CODEGEN_CODEGENRESADDLAYER_H



