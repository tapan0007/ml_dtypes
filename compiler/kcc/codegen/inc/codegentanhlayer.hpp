#pragma once

#ifndef KCC_CODEGEN_CODEGENTANHLAYER_H
#define KCC_CODEGEN_CODEGENTANHLAYER_H

#include "codegenlayer.hpp"

namespace kcc {
namespace codegen {

//########################################################
class CodeGenTanhLayer : public CodeGenActivLayer {
public:
    //----------------------------------------------------------------
    CodeGenTanhLayer(CodeGen* codegen)
        : CodeGenActivLayer(codegen)
    {}

    static string gActivFunc();
};

}}

#endif // KCC_CODEGEN_CODEGENTANHLAYER_H



