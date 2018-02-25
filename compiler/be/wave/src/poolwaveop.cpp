#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/poolwaveop.hpp"


// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


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

    , m_TileId(params.m_TileId)
    , m_TileIdFormat(params.m_TileIdFormat)
    //waveopname;
    //waveoptype;
{
    m_SrcIsPsum         = params.m_SrcIsPsum;
    if (m_SrcIsPsum) {
        m_SrcPsumBankId     = params.m_SrcPsumBankId;
        m_SrcPsumBankOffset = params.m_SrcPsumBankOffset;
    } else {
        m_SrcSbAtomId       = params.m_SrcSbAtomId;
        m_SrcSbOffsetInAtom = params.m_SrcSbOffsetInAtom;
    }

    m_SrcWNum           = params.m_SrcWNum;
    m_SrcWStep          = params.m_SrcWStep;
    m_SrcXNum           = params.m_SrcXNum;
    m_SrcXStep          = params.m_SrcXStep;
    m_SrcYNum           = params.m_SrcYNum;
    m_SrcYStep          = params.m_SrcYStep;
    m_SrcZNum           = params.m_SrcZNum;
    m_SrcZStep          = params.m_SrcZStep;
    assert(verify());
}

bool
PoolWaveOp::verify() const
{
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());
    if (! this->WaveOp::verify()) {
        RETURN_ASSERT(false);
    }
    if (m_DstSbAtomId < 0) {
        RETURN_ASSERT(false);
    }
    if (m_DstSbOffsetInAtom < 0) {
        RETURN_ASSERT(false);
    }
    if (m_DstXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstXStep < 0) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep < 0) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep < 0) {
        RETURN_ASSERT(false);
    }
    // "layername;
    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    if (m_PoolFrequency < 1) {
        RETURN_ASSERT(false);
    }
    switch (m_PoolFunc) {
    case PoolType_Max:
    case PoolType_Avg:
        break;
    default:
        RETURN_ASSERT(false);
    }
    // previouswaveops: [ 1conv/i1/MatMuln0m0h0w0c0r0s0" ]

    if (m_SrcIsPsum) {
        if (m_SrcPsumBankId < 0 || m_SrcPsumBankId >= psumBuf.gNumberBanks()) {
            RETURN_ASSERT(false);
        }
        if (m_SrcPsumBankOffset < 0 || m_SrcPsumBankOffset >= psumBuf.gNumberBankEntries()) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_SrcSbAtomId < 0) {
            RETURN_ASSERT(false);
        }
        if (m_SrcSbOffsetInAtom < 0) {
            RETURN_ASSERT(false);
        }
    }

    if (m_SrcWNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcWStep < 0) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXStep < 0) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep < 0) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep < 0) {
        RETURN_ASSERT(false);
    }

    for (auto n : m_TileId) {
        if (n < 0) {
            RETURN_ASSERT(false);
        }
    }
    if (m_TileIdFormat == "") {
        RETURN_ASSERT(false);
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

