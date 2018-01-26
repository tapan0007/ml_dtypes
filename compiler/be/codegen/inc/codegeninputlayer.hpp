#pragma once

#ifndef KCC_CODEGEN_CODEGENINPUTLAYER_H
#define KCC_CODEGEN_CODEGENINPUTLAYER_H

#include <cstdio>

#include "codegendatalayer.hpp"

namespace kcc {
using namespace utils;

namespace codegen {

class CodeGenInputLayer : public CodeGenDataLayer {
private:
    using SubClass = CodeGenDataLayer;
public:
    //----------------------------------------------------------------
    CodeGenInputLayer(CodeGen* codegen)
        : SubClass(codegen)
    {}

    //----------------------------------------------------------------
    void generate(layers::Layer* layer) override;
};


}}

#endif // KCC_CODEGEN_CODEGENINPUTLAYER_H

