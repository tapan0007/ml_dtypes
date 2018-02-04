#pragma once

#ifndef KCC_CODEGEN_CODEGENTANHLAYER_H
#define KCC_CODEGEN_CODEGENTANHLAYER_H

#include "codegen/inc/codegenactivlayer.hpp"

namespace kcc {
namespace codegen {

//########################################################
class CodeGenTanhLayer : public CodeGenActivLayer {
public:
    //----------------------------------------------------------------
    CodeGenTanhLayer(CodeGen* codegen)
        : CodeGenActivLayer(codegen)
    {}

    ACTIVATIONFUNC gActivFunc() const override;
};

}}

#endif // KCC_CODEGEN_CODEGENTANHLAYER_H



