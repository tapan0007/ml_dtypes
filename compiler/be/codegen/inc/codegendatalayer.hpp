#pragma once

#ifndef KCC_CODEGEN_CODEGENDATALAYER_H
#define KCC_CODEGEN_CODEGENDATALAYER_H

#include <cstdio>

#include "codegen/inc/codegenlayer.hpp"

namespace kcc {
using namespace utils;

namespace codegen {

class CodeGenDataLayer : public CodeGenLayer {
private:
    using SubClass = CodeGenLayer;
public:
    //----------------------------------------------------------------
    CodeGenDataLayer(CodeGen* codegen)
        : SubClass(codegen)
    {}

protected:
    void Generate(layers::DataLayer* dataLayer, addr_t sbAddress);
};


}}

#endif // KCC_CODEGEN_CODEGENDATALAYER_H


