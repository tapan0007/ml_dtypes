#pragma once

#ifndef KCC_CODEGEN_CODEGENARITHMETICLAYER_H
#define KCC_CODEGEN_CODEGENARITHMETICLAYER_H

#include "tpb_isa_activate.hpp"

#include "codegen/inc/codegenlayer.hpp"

namespace kcc {
namespace codegen {

//########################################################
class CodeGenArithmeticLayer : public CodeGenLayer {
private:
    using SubClass = CodeGenLayer;
public:
    //----------------------------------------------------------------
    CodeGenArithmeticLayer(CodeGen* codegen)
        : SubClass(codegen)
    {}

};

}}

#endif // KCC_CODEGEN_CODEGENARITHMETICLAYER_H


