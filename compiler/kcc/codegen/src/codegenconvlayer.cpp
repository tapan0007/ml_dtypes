#include "convlayer.hpp"

#include "codegenconvlayer.hpp"

//-----------------------------------------------------------------
// *[ifmap/filter]Addrs are arrays of statebuffer addresses.  Arrays
// * deal with cases iwhen with the number of ifmap channels is > number of rows.
// * In this case, the ifmaps and filters must be "wrapped".  Each address in the
// * array is the wrap offset
//
// void
// compile_convolve(FILE *out_binary,
//         const addr_t *ifmapAddrs, const uint64_t ifmapDims[4], // NCHW
//         const addr_t *filterAddr, const uint64_t filterDims[4], // MCRS
//         const addr_t ofmapAddr, uint64_t ofmapDims[4], // output NCHW
//         const ARBPRECTYPE in_dtype, const ARBPRECTYPE out_dtype,
//         const uint8_t padding[2],  // Height,Width
//         const uint8_t stride[2],   // Height,Width
//         const uint8_t dilate[2]);  // Height,Width
//-----------------------------------------------------------------

namespace kcc {
namespace codegen {

using layers::Layer;
using layers::ConvLayer;

void
CodeGenConvLayer::generate(Layer* layer)
{
    FILE* const objFile = gObjFile();
    ConvLayer* const convLayer = dynamic_cast<ConvLayer*>(layer);
    assert(convLayer);
        
    Layer* const prevLayer  = convLayer->gPrevLayer(0);
    const unsigned numIfmaps     = prevLayer->gNumOfmaps();
    const unsigned ifmapWidth    = prevLayer->gOfmapWidth();
    const unsigned ifmapHeight   = prevLayer->gOfmapHeight();
    const unsigned numOfmaps     = convLayer->gNumOfmaps();
    const unsigned ofmapWidth    = convLayer->gOfmapWidth();
    const unsigned ofmapHeight   = convLayer->gOfmapHeight();
    const unsigned kernelHeight  = convLayer->gKernelHeight();
    const unsigned kernelWidth   = convLayer->gKernelWidth();
    const unsigned numBatches    = 1;

    const ARBPRECTYPE inDataType  = prevLayer->gDataType().gTypeId();
    const ARBPRECTYPE outDataType = convLayer->gDataType().gTypeId();

    // const addr_t *filterAddr, const uint64_t filterDims[4], // MCRS
    m_FilterAddr[0] = convLayer->gWeightAddress();
    m_FilterFileNames[0] = convLayer->gFilterFileName();

    compile_read_filter(objFile,
            m_FilterAddr[0], m_FilterFileNames[0].c_str(),
            convLayer->gFilterTensorDimSemantics().c_str());

    // N: batch size
    // C: number of ifmaps / channels
    // H: height of ifmap
    // W: width of ifmap
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

    m_Padding[0]         = convLayer->gPaddingRight();
    m_Padding[1]         = convLayer->gPaddingTop();
    m_Convolve_stride[0] = convLayer->gStrideLR();
    m_Convolve_stride[1] = convLayer->gStrideBT();
    m_Dilate[0]          = 0;
    m_Dilate[1]          = 0;

    compile_convolve(objFile,
            m_IfmapAddrs, m_IfmapDims,
            m_FilterAddr, m_FilterDims,
            m_OfmapAddrs, m_OfmapDims,
            inDataType, outDataType,
            m_Padding,
            m_Convolve_stride,
            m_Dilate);

    // const addr_t ofmapAddr, uint64_t ofmapDims[4], // output NCHW
    // N: batch size
    // C: number of ifmaps / channels
    // H: height of ofmap
    // W: width of ofmap
    assert(m_OfmapDims[m_FmapIndex_N] == numBatches);
    assert(m_OfmapDims[m_FmapIndex_C] == numOfmaps);
    assert(m_OfmapDims[m_FmapIndex_H] == ofmapHeight);
    assert(m_OfmapDims[m_FmapIndex_W] == ofmapWidth);

    epilogue(layer);
}

}}


