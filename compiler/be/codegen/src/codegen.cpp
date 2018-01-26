#include <cstdio>


#include "network.hpp"
#include "inputlayer.hpp"
#include "convlayer.hpp"
#include "relulayer.hpp"
#include "tanhlayer.hpp"
#include "maxpoollayer.hpp"
#include "avgpoollayer.hpp"
#include "constlayer.hpp"
#include "resaddlayer.hpp"
#include "biasaddlayer.hpp"

#include "codegen.hpp"
#include "codegeninputlayer.hpp"
#include "codegenconvlayer.hpp"
#include "codegenrelulayer.hpp"
#include "codegentanhlayer.hpp"
#include "codegenmaxpoollayer.hpp"
#include "codegenavgpoollayer.hpp"
#include "codegenconstlayer.hpp"
#include "codegenresaddlayer.hpp"
#include "codegenbiasaddlayer.hpp"

namespace kcc {

namespace layers {
    class InputLayer;
    class ConvLayer;
    class MaxPoolLayer;
    class AvgPoolLayer;
    class ResAddLayer;
    class BiasAddLayer;
    class ConstLayer;
}

namespace codegen {


//########################################################
CodeGen::CodeGen(nets::Network* ntwk, arch::Arch* arch)
{
    m_Network = ntwk;
    m_Arch = arch;
    createGenMap();
}

//----------------------------------------------------------------
void
CodeGen::createGenMap()
{
    m_InputLayer   = std::make_unique<CodeGenInputLayer>(this);
    m_ConvLayer    = std::make_unique<CodeGenConvLayer>(this);
    m_ReluLayer    = std::make_unique<CodeGenReluLayer>(this);
    m_TanhLayer    = std::make_unique<CodeGenTanhLayer>(this);
    m_MaxPoolLayer = std::make_unique<CodeGenMaxPoolLayer>(this);
    m_AvgPoolLayer = std::make_unique<CodeGenAvgPoolLayer>(this);
    m_ResAddLayer  = std::make_unique<CodeGenResAddLayer>(this);
    m_BiasAddLayer = std::make_unique<CodeGenBiasAddLayer>(this);
    m_ConstLayer   = std::make_unique<CodeGenConstLayer>(this);
}

CodeGenLayer&
CodeGen::gGenFunc(const layers::Layer* layer)
{
    if (dynamic_cast<const layers::InputLayer*>(layer)) {
        return *m_InputLayer;
    } else if (dynamic_cast<const layers::ConvLayer*>(layer)) {
        return *m_ConvLayer;
    } else if (dynamic_cast<const layers::ReluLayer*>(layer)) {
        return *m_ReluLayer;
    } else if (dynamic_cast<const layers::TanhLayer*>(layer)) {
        return *m_TanhLayer;
    } else if (dynamic_cast<const layers::MaxPoolLayer*>(layer)) {
        return *m_MaxPoolLayer;
    } else if (dynamic_cast<const layers::AvgPoolLayer*>(layer)) {
        return *m_AvgPoolLayer;
    } else if (dynamic_cast<const layers::BiasAddLayer*>(layer)) {
        return *m_BiasAddLayer;
    } else if (dynamic_cast<const layers::ResAddLayer*>(layer)) {
        return *m_ResAddLayer;
    } else if (dynamic_cast<const layers::ConstLayer*>(layer)) {
        return *m_ConstLayer;
    } else {
        assert("CodeGen::generate: Unknown layer");
    }
    assert("CodeGen::generate: Unknown layer");
    return *m_InputLayer;
}

void
CodeGen::generate(const char* objFileName)
{
    m_ObjFile = std::fopen(objFileName, "w");
    assert(m_ObjFile && "Object file is null");

    for (auto layer : m_Network->gSchedForwLayers()) {
        CodeGenLayer& layerGen = gGenFunc(layer);
        layerGen.generate(layer);
    }

    fclose(m_ObjFile); m_ObjFile = nullptr;
}



}}



