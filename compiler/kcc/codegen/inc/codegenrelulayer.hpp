#pragma once

#ifndef KCC_CODEGEN_CODEGENRELULAYER_H
#define KCC_CODEGEN_CODEGENRELULAYER_H

#include "codegenlayer.hpp"

namespace kcc {
namespace codegen {

//########################################################
class CodeGenReluLayer : public CodeGenActivLayer {
public:
    //----------------------------------------------------------------
    CodeGenReluLayer(CodeGen* codegen)
        : CodeGenActivLayer(codegen)
    {}

    static string gActivFunc();
};

}}

#endif // KCC_CODEGEN_CODEGENRELULAYER_H


