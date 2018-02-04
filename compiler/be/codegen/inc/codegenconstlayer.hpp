#pragma once

#ifndef KCC_CODEGEN_CODEGENCONSTLAYER_H
#define KCC_CODEGEN_CODEGENCONSTLAYER_H

#include <cstdio>

#include "codegen/inc/codegendatalayer.hpp"

namespace kcc {
using namespace utils;

namespace codegen {

class CodeGenConstLayer : public CodeGenDataLayer {
private:
    using SubClass = CodeGenDataLayer;
public:
    //----------------------------------------------------------------
    CodeGenConstLayer(CodeGen* codegen)
        : SubClass(codegen)
    {}

    //----------------------------------------------------------------
    void generate(layers::Layer* layer) override;
};


}}

#endif // KCC_CODEGEN_CODEGENCONSTLAYER_H



