#pragma once

#ifndef KCC_CODEGEN_CODEGENADDLAYER_H
#define KCC_CODEGEN_CODEGENADDLAYER_H


#include "codegen/inc/codegenarithmeticlayer.hpp"

namespace kcc {
namespace codegen {

//########################################################
class CodeGenAddLayer : public CodeGenArithmeticLayer {
private:
    using SubClass = CodeGenArithmeticLayer;
public:
    //----------------------------------------------------------------
    CodeGenAddLayer(CodeGen* codegen)
        : SubClass(codegen)
    {}

};

}}

#endif // KCC_CODEGEN_CODEGENADDLAYER_H




