#pragma once

#ifndef KCC_CODEGEN_CODEGEN_H
#define KCC_CODEGEN_CODEGEN_H

#include<string>
#include<memory>

using namespace std;

#include "layer.hpp"

namespace kcc {

namespace layers {
    class InputLayer;
    class ConvLayer;
}
namespace arch {
    class Arch;
}
namespace nets {
    class Network;
}

namespace codegen {

using nets::Network;
using arch::Arch;
using layers::Layer;
using layers::InputLayer;
using layers::ConvLayer;

class CodeGenLayer;
class CodeGenInputLayer;
class CodeGenConvLayer;
class CodeGenReluLayer;
class CodeGenTanhLayer;
//class CodeGenMaxPoolLayer;
//class CodeGenAvgPoolLayer;

//########################################################
class CodeGen {
public:
    //----------------------------------------------------------------
    CodeGen(Network* ntwk, Arch* arch);

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
    string gDataTypeName(Layer* layer) const {
        string ret("ARBPRECTYPE::");
        ret += layer->gDataType().gTccName();
        return  ret;
    }

    //----------------------------------------------------------------
    CodeGenLayer& gGenFunc(const Layer* layer);


private:
    FILE* m_ObjFile = nullptr;
    Network* m_Network = nullptr;
    Arch* m_Arch = nullptr;
    unique_ptr<CodeGenInputLayer> m_InputLayer;
    unique_ptr<CodeGenConvLayer>  m_ConvLayer;
    unique_ptr<CodeGenReluLayer> m_ReluLayer;
    unique_ptr<CodeGenTanhLayer>  m_TanhLayer;
    //unique_ptr<CodeGenMaxPoolLayer>  m_MaxPoolLayer;
    //unique_ptr<CodeGenAvgPoolLayer>  m_AvgPoolLayer;
};


}}

#endif // KCC_CODEGEN_CODEGEN_H

