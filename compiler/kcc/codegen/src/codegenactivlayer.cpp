
#include "codegenactivlayer.hpp"

#include "activlayer.hpp"

namespace kcc {
namespace codegen {

using layers::ActivLayer;

//  void
//  compile_activation(FILE *out_binary,
//            const addr_t ifmap_addr,
//            const uint64_t ifmapDims[4], /* NCHW */
//            const addr_t ofmap_addr,
//            uint64_t ofmapDims[4], /* output NCHW */
//            const ARBPRECTYPE in_dtype,
//            const ARBPRECTYPE out_dtype,
//            ACTIVATIONFUNC act_func);

void
CodeGenActivLayer::generate(Layer* layer)
{
    FILE* const objFile = gObjFile();
    ActivLayer* const activLayer = dynamic_cast<ActivLayer*>(layer);
    assert(activLayer);

    const ACTIVATIONFUNC activFunc = gActivFunc();
    Layer* const prevLayer      = activLayer->gPrevLayer(0);
    const unsigned numIfmaps    = prevLayer->gNumOfmaps();
    const unsigned ifmapWidth   = prevLayer->gOfmapWidth();
    const unsigned ifmapHeight  = prevLayer->gOfmapHeight();
    const unsigned numOfmaps    = activLayer->gNumOfmaps();
    const unsigned ofmapWidth   = activLayer->gOfmapWidth();
    const unsigned ofmapHeight  = activLayer->gOfmapHeight();
    const unsigned numBatches   = 1;

    const ARBPRECTYPE inDataType  = prevLayer->gDataType().gTypeId();
    const ARBPRECTYPE outDataType  = activLayer->gDataType().gTypeId();

    // const addr_t *ifmapAddrs, const uint64_t ifmapDims[4], // NCHW
    // N: batch size
    // C: number of ifmaps / channels
    // H: height of ifmap
    // W: width of ifmap
    m_IfmapAddrs[0] = activLayer->gIfmapAddress();

    m_IfmapDims[m_IfmapIndex_N] = numBatches;
    m_IfmapDims[m_IfmapIndex_C] = numIfmaps;
    m_IfmapDims[m_IfmapIndex_H] = ifmapHeight;
    m_IfmapDims[m_IfmapIndex_W] = ifmapWidth;

    m_OfmapAddrs = activLayer->gOfmapAddress();

    compile_activation(objFile,
            m_IfmapAddrs[0], m_IfmapDims,
            m_OfmapAddrs, m_OfmapDims,
            inDataType, outDataType,
            activFunc);


    // const addr_t ofmap_addr, uint64_t ofmapDims[4], // output NCHW
    // N: batch size
    // C: number of ifmaps / channels
    // H: height of ofmap
    // W: width of ofmap
    assert(m_OfmapDims[m_OfmapIndex_N] == numBatches);
    assert(m_OfmapDims[m_OfmapIndex_C] == numOfmaps);
    assert(m_OfmapDims[m_OfmapIndex_H] == ofmapHeight);
    assert(m_OfmapDims[m_OfmapIndex_W] == ofmapWidth);

    epilogue(layer);
}

}}
