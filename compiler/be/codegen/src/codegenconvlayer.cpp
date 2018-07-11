#include "layers/inc/convlayer.hpp"

#include "codegen/inc/codegenconvlayer.hpp"

//-----------------------------------------------------------------
// *[ifmap/filter]Addrs are arrays of statebuffer addresses.  Arrays
// * deal with cases iwhen with the number of ifmap channels is > number of rows.
// * In this case, the ifmaps and filters must be "wrapped".  Each address in the
// * array is the wrap offset
//-----------------------------------------------------------------

namespace kcc {
namespace codegen {


void
CodeGenConvLayer::generate(layers::Layer* layer)
{
    FILE* const objFile = gObjFile();
    const auto convLayer = dynamic_cast<layers::ConvLayer*>(layer);
    assert(convLayer && "CodeGenLayer::generate: layer is not a convolution layer");

    layers::Layer* const prevLayer  = convLayer->gPrevLayer(0);
    const unsigned numIfmaps     = prevLayer->gNumOfmaps();
    const unsigned ifmapWidth    = prevLayer->gOfmapWidth();
    const unsigned ifmapHeight   = prevLayer->gOfmapHeight();
    const unsigned numOfmaps     = convLayer->gNumOfmaps();
    const unsigned ofmapWidth    = convLayer->gOfmapWidth();
    const unsigned ofmapHeight   = convLayer->gOfmapHeight();
    const unsigned kernelHeight  = convLayer->gKernelHeight();
    const unsigned kernelWidth   = convLayer->gKernelWidth();
    const unsigned numBatches    = 1;

    const TONGA_ISA_TPB_DTYPE inDataType  = prevLayer->gDataType().gSimTypeId();
    const TONGA_ISA_TPB_DTYPE outDataType = convLayer->gDataType().gSimTypeId();

    m_FilterAddr[0] = convLayer->gWeightAddress();
    m_FilterFileNames[0] = convLayer->gFilterFileName();

    compile_read_filter(objFile,
            m_FilterAddr[0], m_FilterFileNames[0].c_str(),
            convLayer->gFilterTensorDimSemantics().c_str());

    m_IfmapAddrs[0] = convLayer->gIfmapAddress();
    m_IfmapDims[m_FmapIndex_N] = numBatches;
    m_IfmapDims[m_FmapIndex_C] = numIfmaps;
    m_IfmapDims[m_FmapIndex_H] = ifmapHeight;
    m_IfmapDims[m_FmapIndex_W] = ifmapWidth;

    // filterAddr
    m_FilterDims[m_FilterIndex_M] = numOfmaps;
    m_FilterDims[m_FilterIndex_C] = numIfmaps;
    m_FilterDims[m_FilterIndex_R] = kernelHeight;
    m_FilterDims[m_FilterIndex_S] = kernelWidth;

    m_OfmapAddrs = convLayer->gOfmapAddress();

    m_Padding[PaddingIndex_Top]     = convLayer->gPaddingTop();
    m_Padding[PaddingIndex_Bottom]  = convLayer->gPaddingBottom();
    m_Padding[PaddingIndex_Left]    = convLayer->gPaddingLeft();
    m_Padding[PaddingIndex_Right]   = convLayer->gPaddingRight();
    m_ConvolveStride[0]             = convLayer->gStrideTopBottom();
    m_ConvolveStride[1]             = convLayer->gStrideLeftRight();
    m_Dilate[0]                     = 1;
    m_Dilate[1]                     = 1;

    compile_convolve(objFile,
            m_IfmapAddrs, m_IfmapDims,
            m_FilterAddr, m_FilterDims,
            m_OfmapAddrs, m_OfmapDims,
            inDataType, outDataType,
            m_Padding,
            m_ConvolveStride,
            m_Dilate);

    assert(m_OfmapDims[m_FmapIndex_N] == numBatches && "Number of batches not matching after convolution calculation");
    assert(m_OfmapDims[m_FmapIndex_C] == numOfmaps && "Number of OFMAPs not matching after convolution calculation");
    assert(m_OfmapDims[m_FmapIndex_H] == ofmapHeight && "OFMAP height not matching after convolution calculation");
    assert(m_OfmapDims[m_FmapIndex_W] == ofmapWidth && "OFMAP width  not matching after convolution calculation");

    epilogue(layer);
}

}}


