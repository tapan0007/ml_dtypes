#include "utils/inc/asserter.hpp"

#include "layers/inc/activlayer.hpp"
#include "codegenactivlayer.hpp"


namespace kcc {
namespace codegen {



// Generate Tonga code for an activation layer
void
CodeGenActivLayer::generate(layers::Layer* layer)
{
    FILE* const objFile = gObjFile();
    const auto activLayer = dynamic_cast<layers::ActivLayer*>(layer);
    Assert(activLayer, "CodeGen::generate: layer is not an activiation layer: ", layer->gTypeStr());

    const TONGA_ISA_TPB_ACTIVATION_FUNC activFunc  = gActivFunc();
    layers::Layer* const prevLayer  = activLayer->gPrevLayer(0);
    const unsigned numIfmaps        = prevLayer->gNumOfmaps();
    const unsigned ifmapWidth       = prevLayer->gOfmapWidth();
    const unsigned ifmapHeight      = prevLayer->gOfmapHeight();
    const unsigned numOfmaps        = activLayer->gNumOfmaps();
    const unsigned ofmapWidth       = activLayer->gOfmapWidth();
    const unsigned ofmapHeight      = activLayer->gOfmapHeight();
    const unsigned numBatches       = 1;

    const TONGA_ISA_TPB_DTYPE inDataType    = prevLayer->gDataType().gSimTypeId();
    const TONGA_ISA_TPB_DTYPE outDataType   = activLayer->gDataType().gSimTypeId();

    m_IfmapAddrs[0] = activLayer->gIfmapAddress();

    m_IfmapDims[m_FmapIndex_N] = numBatches;
    m_IfmapDims[m_FmapIndex_C] = numIfmaps;
    m_IfmapDims[m_FmapIndex_H] = ifmapHeight;
    m_IfmapDims[m_FmapIndex_W] = ifmapWidth;

    m_OfmapAddrs = activLayer->gOfmapAddress();

    compile_activation(objFile,
            m_IfmapAddrs[0], m_IfmapDims,
            m_OfmapAddrs, m_OfmapDims,
            inDataType, outDataType,
            activFunc);


    assert(m_OfmapDims[m_FmapIndex_N] == numBatches && "Number of batches not matching after activation calculation");
    assert(m_OfmapDims[m_FmapIndex_C] == numOfmaps && "Number of OFMAPs not matching after activation calculation");
    assert(m_OfmapDims[m_FmapIndex_H] == ofmapHeight && "OFMAP height not matching after activation calculation");
    assert(m_OfmapDims[m_FmapIndex_W] == ofmapWidth && "OFMAP width  not matching after activation calculation");

    epilogue(layer);
}

}}
