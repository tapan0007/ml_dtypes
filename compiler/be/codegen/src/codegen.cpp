#include <cstdio>


#include "network.hpp"
#include "inputlayer.hpp"
#include "convlayer.hpp"
#include "relulayer.hpp"
#include "tanhlayer.hpp"
#include "maxpoollayer.hpp"
#include "avgpoollayer.hpp"

#include "codegen.hpp"
#include "codegeninputlayer.hpp"
#include "codegenconvlayer.hpp"
#include "codegenrelulayer.hpp"
#include "codegentanhlayer.hpp"
#include "codegenmaxpoollayer.hpp"
#include "codegenavgpoollayer.hpp"

namespace kcc {

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
    m_InputLayer.reset(new CodeGenInputLayer(this));
    m_ConvLayer.reset(new CodeGenConvLayer(this));
    m_ReluLayer.reset(new CodeGenReluLayer(this));
    m_TanhLayer.reset(new CodeGenTanhLayer(this));
    m_MaxPoolLayer.reset(new CodeGenMaxPoolLayer(this));
    m_AvgPoolLayer.reset(new CodeGenAvgPoolLayer(this));
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
    } else {
        assert(false || "CodeGen::generate: Unknown layer");
    }
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



