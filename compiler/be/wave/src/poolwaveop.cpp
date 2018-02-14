#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/poolwaveop.hpp"



namespace kcc {
namespace wave {

PoolWaveOp::PoolWaveOp(const PoolWaveOp::Params& params,
                       const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, prevWaveOps)
    , m_DstSbAtomId(params.m_DstSbAtomId)
    , m_DstSbOffsetInAtom(params.m_DstSbOffsetInAtom)
    , m_DstXNum(params.m_DstXNum)
    , m_DstXStep(params.m_DstXStep)
    , m_DstYNum(params.m_DstYNum)
    , m_DstYStep(params.m_DstYStep)
    , m_DstZNum(params.m_DstZNum)
    , m_DstZStep(params.m_DstZStep)
    , m_InDtype(DataType::dataTypeId2DataType(params.m_InDtype))
    // "layername;
    , m_NumPartitions(params.m_NumPartitions)
    , m_OutDtype(DataType::dataTypeId2DataType(params.m_OutDtype))
    , m_PoolFrequency(params.m_PoolFrequency)
    , m_PoolFunc(params.m_PoolFunc)
    // previouswaveops;
    //  1conv/i1/MatMuln0m0h0w0c0r0s0"
    // ],
    , m_SrcIsPsum(params.m_SrcIsPsum)
    , m_SrcPsumBankId(params.m_SrcPsumBankId)
    , m_SrcPsumBankOffset(params.m_SrcPsumBankOffset)
    , m_SrcSbAtomId(params.m_SrcSbAtomId)
    , m_SrcSbOffsetInAtom(params.m_SrcSbOffsetInAtom)
    , m_SrcWNum(params.m_SrcWNum)
    , m_SrcWStep(params.m_SrcWStep)
    , m_SrcXNum(params.m_SrcXNum)
    , m_SrcXStep(params.m_SrcXStep)
    , m_SrcYNum(params.m_SrcYNum)
    , m_SrcYStep(params.m_SrcYStep)
    , m_SrcZNum(params.m_SrcZNum)
    , m_SrcZStep(params.m_SrcZStep)

    , m_TileId(params.m_TileId)
    , m_TileIdFormat(params.m_TileIdFormat)
    //waveopname;
    //waveoptype;
{
    assert(verify());
}

bool 
PoolWaveOp::verify() const
{
    if (! this->WaveOp::verify()) {
        return false;
    }
    if (m_DstSbAtomId < 0) {
        return false;
    }
    if (m_DstSbOffsetInAtom < 0) {
        return false;
    }
    if (m_DstXNum < 1) {
        return false;
    }
    if (m_DstXStep < 0) {
        return false;
    }
    if (m_DstYNum < 1) {
        return false;
    }
    if (m_DstYStep < 0) {
        return false;
    }
    if (m_DstZNum < 1) {
        return false;
    }
    if (m_DstZStep < 0) {
        return false;
    }
    // "layername;
    if (m_NumPartitions < 1) {
        return false;
    }
    if (m_PoolFrequency < 1) {
        return false;
    }
    if (m_PoolFunc != PoolType_Max && m_PoolFunc != PoolType_Avg) {
        return false;
    }
    // previouswaveops: [ 1conv/i1/MatMuln0m0h0w0c0r0s0" ]
    // m_SrcIsPsum;
    if (m_SrcPsumBankId < 0) {
        return false;
    }
    if (m_SrcPsumBankOffset < 0) {
        return false;
    }
    if (m_SrcSbAtomId < 0) {
        return false;
    }
    if (m_SrcSbOffsetInAtom < 0) {
        return false;
    }
    if (m_SrcWNum < 1) {
        return false;
    }
    if (m_SrcWStep < 0) {
        return false;
    }
    if (m_SrcXNum < 1) {
        return false;
    }
    if (m_SrcXStep < 0) {
        return false;
    }
    if (m_SrcYNum < 1) {
        return false;
    }
    if (m_SrcYStep < 0) {
        return false;
    }
    if (m_SrcZNum < 1) {
        return false;
    }
    if (m_SrcZStep < 0) {
        return false;
    }
    for (auto n : m_TileId) {
        if (n < 1) {
            return false;
        }
    }
    if (m_TileIdFormat == "") {
        return false;
    }
    //waveopname;
    //waveoptype;
    return true;
}




bool 
PoolWaveOp::Params::verify() const
{
    return true;
}

}}

