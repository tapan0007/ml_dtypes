#pragma once

#ifndef KCC_CODEGEN_CODEGENACTIVLAYER_H
#define KCC_CODEGEN_CODEGENACTIVLAYER_H

#include "aws_tonga_isa_tpb_common.h"

#include "codegen/inc/codegenlayer.hpp"

namespace kcc {
namespace codegen {

//########################################################
class CodeGenActivLayer : public CodeGenLayer {
public:
    //----------------------------------------------------------------
    CodeGenActivLayer(CodeGen* codegen)
        : CodeGenLayer(codegen)
    {}

    virtual TONGA_ISA_TPB_ACTIVATION_FUNC gActivFunc() const = 0;

    void generate(layers::Layer* layer) override;
};

}}

#endif // KCC_CODEGEN_CODEGENACTIVLAYER_H


