#include <cstdio>

using namespace std;

#include "network.hpp"
#include "inputlayer.hpp"
#include "convlayer.hpp"
#include "relulayer.hpp"
#include "tanhlayer.hpp"

#include "codegen.hpp"
#include "codegeninputlayer.hpp"
#include "codegenconvlayer.hpp"
#include "codegenrelulayer.hpp"
#include "codegentanhlayer.hpp"

namespace kcc {
using layers::InputLayer;
using layers::ConvLayer;
using layers::ReluLayer;
using layers::TanhLayer;

namespace codegen {


//########################################################
CodeGen::CodeGen(Network* ntwk, Arch* arch)
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
    //m_MaxPoolLayer.reset(new CodeGenMaxPoolLayer());
    //m_AvgPoolLayer.reset(new CodeGenAvgPoolLayer());
}

CodeGenLayer&
CodeGen::gGenFunc(const Layer* layer)
{
    if (dynamic_cast<const InputLayer*>(layer)) {
        return *m_InputLayer;
    } else if (dynamic_cast<const ConvLayer*>(layer)) {
        return *m_ConvLayer;
    } else if (dynamic_cast<const ReluLayer*>(layer)) {
        return *m_ReluLayer;
    } else if (dynamic_cast<const TanhLayer*>(layer)) {
        return *m_TanhLayer;
    //} else if (dynamic_cast<const MaxPoolLayer*>(layer)) {
    //    return *m_MaxPoolLayer;
    //} else if (dynamic_cast<const AvgPoolhLayer*>(layer)) {
    //    return *m_AvgPoolLayer;
    } else {
        assert(false);
    }
}

void
CodeGen::generate(const char* objFileName)
{
    m_ObjFile = std::fopen(objFileName, "w");
    assert(m_ObjFile);

    for (auto layer : m_Network->gSchedForwLayers()) {
        CodeGenLayer& layerGen = gGenFunc(layer);
        layerGen.generate(layer);
    }

    fclose(m_ObjFile); m_ObjFile = nullptr;
}



}}



