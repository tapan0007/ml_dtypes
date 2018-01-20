
#include "codegenactivlayer.hpp"

#include "activlayer.hpp"

namespace kcc {
namespace codegen {



// Generate Tonga code for an activation layer
void
CodeGenActivLayer::generate(layers::Layer* layer)
{
    FILE* const objFile = gObjFile();
    const auto activLayer = dynamic_cast<layers::ActivLayer*>(layer);
    assert(activLayer);

    const ACTIVATIONFUNC activFunc  = gActivFunc();
    layers::Layer* const prevLayer  = activLayer->gPrevLayer(0);
    const unsigned numIfmaps        = prevLayer->gNumOfmaps();
    const unsigned ifmapWidth       = prevLayer->gOfmapWidth();
    const unsigned ifmapHeight      = prevLayer->gOfmapHeight();
    const unsigned numOfmaps        = activLayer->gNumOfmaps();
    const unsigned ofmapWidth       = activLayer->gOfmapWidth();
    const unsigned ofmapHeight      = activLayer->gOfmapHeight();
    const unsigned numBatches       = 1;

    const ARBPRECTYPE inDataType    = prevLayer->gDataType().gTypeId();
    const ARBPRECTYPE outDataType   = activLayer->gDataType().gTypeId();

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


    assert(m_OfmapDims[m_FmapIndex_N] == numBatches);
    assert(m_OfmapDims[m_FmapIndex_C] == numOfmaps);
    assert(m_OfmapDims[m_FmapIndex_H] == ofmapHeight);
    assert(m_OfmapDims[m_FmapIndex_W] == ofmapWidth);

    epilogue(layer);
}

}}
