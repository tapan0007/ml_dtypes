#include "layers/inc/poollayer.hpp"
#include "codegen/inc/codegenpoollayer.hpp"

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
CodeGenPoolLayer::Generate(layers::Layer* layer, TONGA_ISA_TPB_POOL_TYPE poolFunc)
{
    FILE* const objFile = gObjFile();
    const auto poolLayer = dynamic_cast<layers::PoolLayer*>(layer);
    assert(poolLayer && "CodeGen::generate: layer is not a Pool layer");

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

    const TONGA_ISA_TPB_DTYPE inDataType    = prevLayer->gDataType().gSimTypeId();
    const TONGA_ISA_TPB_DTYPE outDataType   = poolLayer->gDataType().gSimTypeId();


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

    assert(inDataType == outDataType && "Pool layer's in and out data types should be identical");
    compile_pool(objFile,
            m_IfmapAddrs[0], m_IfmapDims,
            m_KernelDims,
            m_OfmapAddrs, m_OfmapDims,
            m_PoolStride,
            outDataType,
            poolFunc);

    assert(m_OfmapDims[m_FmapIndex_N] == numBatches && "Number of batches not matching after pool calculation");
    assert(m_OfmapDims[m_FmapIndex_C] == numOfmaps && "Number of OFMAPs not matching after pool calculation");
    assert(m_OfmapDims[m_FmapIndex_H] == ofmapHeight && "OFMAP height not matching after pool calculation");
    assert(m_OfmapDims[m_FmapIndex_W] == ofmapWidth && "OFMAP width  not matching after pool calculation");

    epilogue(poolLayer);
}

}}

