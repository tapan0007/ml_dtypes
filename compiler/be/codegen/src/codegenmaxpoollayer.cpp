#include "maxpoollayer.hpp"

#include "codegenmaxpoollayer.hpp"

//-----------------------------------------------------------------
// *[ifmap/filter]Addrs are arrays of statebuffer addresses.  Arrays
// * deal with cases iwhen with the number of ifmap channels is > number of rows.
// * In this case, the ifmaps and filters must be "wrapped".  Each address in the
// * array is the wrap offset
//-----------------------------------------------------------------

namespace kcc {
namespace codegen {

using layers::Layer;
using layers::MaxPoolLayer;

void
CodeGenMaxPoolLayer::generate(Layer* layer)
{
    FILE* const objFile = gObjFile();
    MaxPoolLayer* const maxpoolLayer = dynamic_cast<MaxPoolLayer*>(layer);
    assert(maxpoolLayer);

    Layer* const prevLayer  = maxpoolLayer->gPrevLayer(0);
    const unsigned numIfmaps     = prevLayer->gNumOfmaps();
    const unsigned ifmapWidth    = prevLayer->gOfmapWidth();
    const unsigned ifmapHeight   = prevLayer->gOfmapHeight();
    const unsigned numOfmaps     = maxpoolLayer->gNumOfmaps();
    const unsigned ofmapWidth    = maxpoolLayer->gOfmapWidth();
    const unsigned ofmapHeight   = maxpoolLayer->gOfmapHeight();
    const unsigned kernelHeight  = maxpoolLayer->gKernelHeight();
    const unsigned kernelWidth   = maxpoolLayer->gKernelWidth();
    const unsigned numBatches    = 1;

    const ARBPRECTYPE inDataType  = prevLayer->gDataType().gTypeId();
    const ARBPRECTYPE outDataType = maxpoolLayer->gDataType().gTypeId();


    m_IfmapAddrs[0] = maxpoolLayer->gIfmapAddress();
    m_IfmapDims[m_FmapIndex_N] = numBatches;
    m_IfmapDims[m_FmapIndex_C] = numIfmaps;
    m_IfmapDims[m_FmapIndex_H] = ifmapHeight;
    m_IfmapDims[m_FmapIndex_W] = ifmapWidth;

    // filterAddr
    m_KernelDims[m_FmapIndex_N] = 1; // no pooling across batches
    m_KernelDims[m_FmapIndex_C] = 1; // no pooling across ifmaps
    m_KernelDims[m_FmapIndex_H] = kernelHeight;
    m_KernelDims[m_FmapIndex_W] = kernelWidth;

    m_OfmapAddrs = maxpoolLayer->gOfmapAddress();

    m_Padding[PaddingIndex_Top]     = maxpoolLayer->gPaddingTop();
    m_Padding[PaddingIndex_Bottom]  = maxpoolLayer->gPaddingBottom();
    m_Padding[PaddingIndex_Left]    = maxpoolLayer->gPaddingLeft();
    m_Padding[PaddingIndex_Right]   = maxpoolLayer->gPaddingRight();

    //    const uint64_t stride_dims[4], /* NCHW */
    m_PoolStride[FmapIndex_N]       = 1; // batches is 1 today
    m_PoolStride[FmapIndex_C]       = 1; // input channels/fmaps don't pool today
    m_PoolStride[FmapIndex_H]       = maxpoolLayer->gStrideTopBottom();
    m_PoolStride[FmapIndex_W]       = maxpoolLayer->gStrideLeftRight();

    m_Dilate[0]                     = 1;
    m_Dilate[1]                     = 1;

    assert(inDataType == outDataType);
    compile_pool(objFile,
            m_IfmapAddrs[0], m_IfmapDims,
            m_KernelDims,
            m_OfmapAddrs, m_OfmapDims,
            m_PoolStride,
            outDataType,
            POOLFUNC::MAX_POOL);

    assert(m_OfmapDims[m_FmapIndex_N] == numBatches);
    assert(m_OfmapDims[m_FmapIndex_C] == numOfmaps);
    assert(m_OfmapDims[m_FmapIndex_H] == ofmapHeight);
    assert(m_OfmapDims[m_FmapIndex_W] == ofmapWidth);

    epilogue(layer);
}

}}



