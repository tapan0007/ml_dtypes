#pragma once

#ifndef KCC_CODEGEN_CODEGENRELULAYER_H
#define KCC_CODEGEN_CODEGENRELULAYER_H

#include "codegen/inc/codegenactivlayer.hpp"

namespace kcc {
namespace codegen {

//########################################################
class CodeGenReluLayer : public CodeGenActivLayer {
public:
    //----------------------------------------------------------------
    CodeGenReluLayer(CodeGen* codegen)
        : CodeGenActivLayer(codegen)
    {}

    ACTIVATIONFUNC gActivFunc() const override;
};

}}

#endif // KCC_CODEGEN_CODEGENRELULAYER_H


