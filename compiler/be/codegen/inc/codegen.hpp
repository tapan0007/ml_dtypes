#pragma once

#ifndef KCC_CODEGEN_CODEGEN_H
#define KCC_CODEGEN_CODEGEN_H

#include<string>
#include<memory>


#include "layer.hpp"

namespace kcc {

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
class CodeGenResAddLayer;
class CodeGenBiasAddLayer;
class CodeGenConstLayer;


//########################################################
class CodeGen {
public:
    //----------------------------------------------------------------
    CodeGen(nets::Network* ntwk, const arch::Arch& arch);

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
    std::string gDataTypeName(const layers::Layer* layer) const {
        std::string ret("ARBPRECTYPE::");
        ret += layer->gDataType().gTccName();
        return  ret;
    }

    //----------------------------------------------------------------
    CodeGenLayer& gGenFunc(const layers::Layer* layer);


private:
    FILE*                                   m_ObjFile = nullptr;
    const nets::Network*                    m_Network = nullptr;
    const arch::Arch&                       m_Arch;

    std::unique_ptr<CodeGenInputLayer>      m_InputLayer;
    std::unique_ptr<CodeGenConvLayer>       m_ConvLayer;
    std::unique_ptr<CodeGenReluLayer>       m_ReluLayer;
    std::unique_ptr<CodeGenTanhLayer>       m_TanhLayer;
    std::unique_ptr<CodeGenMaxPoolLayer>    m_MaxPoolLayer;
    std::unique_ptr<CodeGenAvgPoolLayer>    m_AvgPoolLayer;
    std::unique_ptr<CodeGenResAddLayer>     m_ResAddLayer;
    std::unique_ptr<CodeGenBiasAddLayer>    m_BiasAddLayer;
    std::unique_ptr<CodeGenConstLayer>      m_ConstLayer;
};


}}

#endif // KCC_CODEGEN_CODEGEN_H
