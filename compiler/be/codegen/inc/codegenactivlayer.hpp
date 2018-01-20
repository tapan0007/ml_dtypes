#pragma once

#ifndef KCC_CODEGEN_CODEGENACTIVLAYER_H
#define KCC_CODEGEN_CODEGENACTIVLAYER_H

#include "tpb_isa_activate.hpp"

#include "codegenlayer.hpp"

namespace kcc {
namespace codegen {

//########################################################
class CodeGenActivLayer : public CodeGenLayer {
public:
    //----------------------------------------------------------------
    CodeGenActivLayer(CodeGen* codegen)
        : CodeGenLayer(codegen)
    {}

    virtual ACTIVATIONFUNC gActivFunc() const = 0;

    void generate(layers::Layer* layer) override;
};

}}

#endif // KCC_CODEGEN_CODEGENACTIVLAYER_H


