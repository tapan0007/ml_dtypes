#include "poollayer.hpp"
#include "codegenpoollayer.hpp"

namespace kcc {
namespace layers {
    class Layer;
    class PoolLayer;
}

namespace codegen {

CodeGenPoolLayer::CodeGenPoolLayer(CodeGen* codegen)
    : CodeGenLayer(codegen)
{}


void
CodeGenPoolLayer::Generate(layers::Layer* layer, POOLFUNC poolFunc)
{
    FILE* const objFile = gObjFile();
    const auto poolLayer = dynamic_cast<layers::PoolLayer*>(layer);
    assert(poolLayer);

    layers::Layer* const prevLayer  = poolLayer->gPrevLayer(0);
    const unsigned numIfmaps        = prevLayer->gNumOfmaps();
    const unsigned ifmapWidth       = prevLayer->gOfmapWidth();
    const unsigned ifmapHeight      = prevLayer->gOfmapHeight();
    const unsigned numOfmaps        = poolLayer->gNumOfmaps();
    const unsigned ofmapWidth       = poolLayer->gOfmapWidth();
    const unsigned ofmapHeight      = poolLayer->gOfmapHeight();
    const unsigned kernelHeight     = poolLayer->gKernelHeight();
    const unsigned kernelWidth      = poolLayer->gKernelWidth();
    const unsigned numBatches       = 1;

    const ARBPRECTYPE inDataType    = prevLayer->gDataType().gTypeId();
    const ARBPRECTYPE outDataType   = poolLayer->gDataType().gTypeId();


    m_IfmapAddrs[0] = poolLayer->gIfmapAddress();
    m_IfmapDims[m_FmapIndex_N] = numBatches;
    m_IfmapDims[m_FmapIndex_C] = numIfmaps;
    m_IfmapDims[m_FmapIndex_H] = ifmapHeight;
    m_IfmapDims[m_FmapIndex_W] = ifmapWidth;

    // filterAddr
    m_KernelDims[m_FmapIndex_N] = 1; // no pooling across batches
    m_KernelDims[m_FmapIndex_C] = 1; // no pooling across ifmaps
    m_KernelDims[m_FmapIndex_H] = kernelHeight;
    m_KernelDims[m_FmapIndex_W] = kernelWidth;

    m_OfmapAddrs = poolLayer->gOfmapAddress();

    m_Padding[PaddingIndex_Top]     = poolLayer->gPaddingTop();
    m_Padding[PaddingIndex_Bottom]  = poolLayer->gPaddingBottom();
    m_Padding[PaddingIndex_Left]    = poolLayer->gPaddingLeft();
    m_Padding[PaddingIndex_Right]   = poolLayer->gPaddingRight();

    //    const uint64_t stride_dims[4], /* NCHW */
    m_PoolStride[FmapIndex_N]       = 1; // batches is 1 today
    m_PoolStride[FmapIndex_C]       = 1; // input channels/fmaps don't pool today
    m_PoolStride[FmapIndex_H]       = poolLayer->gStrideTopBottom();
    m_PoolStride[FmapIndex_W]       = poolLayer->gStrideLeftRight();

    m_Dilate[0]                     = 1;
    m_Dilate[1]                     = 1;

    assert(inDataType == outDataType);
    compile_pool(objFile,
            m_IfmapAddrs[0], m_IfmapDims,
            m_KernelDims,
            m_OfmapAddrs, m_OfmapDims,
            m_PoolStride,
            outDataType,
            poolFunc);

    assert(m_OfmapDims[m_FmapIndex_N] == numBatches);
    assert(m_OfmapDims[m_FmapIndex_C] == numOfmaps);
    assert(m_OfmapDims[m_FmapIndex_H] == ofmapHeight);
    assert(m_OfmapDims[m_FmapIndex_W] == ofmapWidth);

    epilogue(poolLayer);
}

}}

