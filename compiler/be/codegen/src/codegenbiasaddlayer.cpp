
#include "layers/inc/inputlayer.hpp"
#include "layers/inc/constlayer.hpp"
#include "layers/inc/biasaddlayer.hpp"
#include "codegen/inc/codegenbiasaddlayer.hpp"

namespace kcc {
namespace codegen {

void
CodeGenBiasAddLayer::generate(layers::Layer* layer)
{
    FILE* const objFile = gObjFile();
    assert(objFile);
    layers::BiasAddLayer* const biasaddLayer = dynamic_cast<layers::BiasAddLayer*>(layer);
    assert(biasaddLayer && "CodeGenLayer::generate: layer is not a BiasAdd layer");

    const std::vector<layers::Layer*>& prevLayers = biasaddLayer->gPrevLayers();
    assert(prevLayers.size() == 2);
    layers::InputLayer* inLayer;
    layers::ConstLayer* constLayer;
    if ( (inLayer = dynamic_cast<layers::InputLayer*>(prevLayers[0])) ) {
        constLayer = dynamic_cast<layers::ConstLayer*>(prevLayers[1]);
        assert(constLayer);
    } else {
        constLayer = dynamic_cast<layers::ConstLayer*>(prevLayers[0]);
        assert(constLayer);
        inLayer = dynamic_cast<layers::InputLayer*>(prevLayers[1]);
        assert(inLayer);
    }

    const ARBPRECTYPE inDataType    = inLayer->gDataType().gSimTypeId();
    const ARBPRECTYPE outDataType   = biasaddLayer->gDataType().gSimTypeId();
    const unsigned int ofmapWidth   = biasaddLayer->gOfmapWidth();
    const unsigned int ofmapHeight  = biasaddLayer->gOfmapHeight();
    m_IfmapAddrs[0]                 = inLayer->gOfmapAddress();
    m_IfmapDims[m_FmapIndex_N]      = 1;
    m_IfmapDims[m_FmapIndex_C]      = inLayer->gNumOfmaps();
    m_IfmapDims[m_FmapIndex_H]      = inLayer->gOfmapHeight();
    m_IfmapDims[m_FmapIndex_W]      = inLayer->gOfmapWidth();
    m_OfmapAddrs                    = biasaddLayer->gOfmapAddress();

    /* Must use 4 file API becaue 1 file API cannot specify addition address */
    compile_activation(
        objFile,
        m_IfmapAddrs[0], m_IfmapDims,
        m_OfmapAddrs, m_OfmapDims,
        inDataType, outDataType,
        ACTIVATIONFUNC::IDENTITY,
        1.0f,
        constLayer->gOfmapAddress());
    assert(ofmapWidth == m_OfmapDims[m_FmapIndex_W] && "BiasAdd layer: Ofmap width mismatch");
    assert(ofmapHeight == m_OfmapDims[m_FmapIndex_H] && "BiasAdd layer: Ofmap height mismatch");

    epilogue(layer);
}

}}


