#pragma once

#ifndef KCC_CODEGEN_CODEGEN_H
#define KCC_CODEGEN_CODEGEN_H

#include<string>
#include<memory>


#include "layer.hpp"

namespace kcc {

namespace layers {
    class InputLayer;
    class ConvLayer;
    class MaxPoolLayer;
    class AvgPoolLayer;
}
namespace arch {
    class Arch;
}
namespace nets {
    class Network;
}

namespace codegen {


class CodeGenLayer;
class CodeGenInputLayer;
class CodeGenConvLayer;
class CodeGenReluLayer;
class CodeGenTanhLayer;
class CodeGenMaxPoolLayer;
class CodeGenAvgPoolLayer;

//########################################################
class CodeGen {
public:
    //----------------------------------------------------------------
    CodeGen(nets::Network* ntwk, arch::Arch* arch);

    //----------------------------------------------------------------
    FILE* gObjFile() const {
        return m_ObjFile;
    }

    //----------------------------------------------------------------
    void generate(const char* fileName);


private:
    //----------------------------------------------------------------
    void createGenMap();

    //----------------------------------------------------------------
    string gDataTypeName(const layers::Layer* layer) const {
        string ret("ARBPRECTYPE::");
        ret += layer->gDataType().gTccName();
        return  ret;
    }

    //----------------------------------------------------------------
    CodeGenLayer& gGenFunc(const layers::Layer* layer);


private:
    FILE*                                   m_ObjFile = nullptr;
    const nets::Network*                    m_Network = nullptr;
    const arch::Arch*                       m_Arch = nullptr;
    std::unique_ptr<CodeGenInputLayer>      m_InputLayer;
    std::unique_ptr<CodeGenConvLayer>       m_ConvLayer;
    std::unique_ptr<CodeGenReluLayer>       m_ReluLayer;
    std::unique_ptr<CodeGenTanhLayer>       m_TanhLayer;
    std::unique_ptr<CodeGenMaxPoolLayer>    m_MaxPoolLayer;
    std::unique_ptr<CodeGenAvgPoolLayer>    m_AvgPoolLayer;
};


}}

#endif // KCC_CODEGEN_CODEGEN_H

