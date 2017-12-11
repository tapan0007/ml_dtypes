#pragma once

#ifndef KCC_CODEGEN_CODEGENINPUTLAYER_H
#define KCC_CODEGEN_CODEGENINPUTLAYER_H

#include <cstdio>

#include "codegenlayer.hpp"

namespace kcc {
namespace codegen {

class CodeGenInputLayer : public CodeGenLayer {
public:
    CodeGenInputLayer(CodeGen* codegen)
        : CodeGenLayer(codegen)
    {}

    void generate(Layer* layer) override;
};


}}

#endif // KCC_CODEGEN_CODEGENINPUTLAYER_H

