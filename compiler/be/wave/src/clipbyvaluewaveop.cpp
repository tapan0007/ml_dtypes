#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "wave/inc/waveconsts.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/clipbyvaluewaveop.hpp"

// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

ClipByValueWaveOp::ClipByValueWaveOp(const ClipByValueWaveOp::Params& params,
                       const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, prevWaveOps)
    , m_InDtype(DataType::dataTypeId2DataType(params.m_InDtypeId))
    , m_OutDtype(DataType::dataTypeId2DataType(params.m_OutDtypeId))
    , m_SrcIsPsum(params.m_SrcIsPsum)
    , m_DstIsPsum(params.m_DstIsPsum)

    , m_SrcXNum(params.m_SrcXNum)
    , m_SrcXStep(params.m_SrcXStep)
    , m_SrcYNum(params.m_SrcYNum)
    , m_SrcYStep(params.m_SrcYStep)
    , m_SrcZNum(params.m_SrcZNum)
    , m_SrcZStep(params.m_SrcZStep)
    , m_DstXNum(params.m_DstXNum)
    , m_DstXStep(params.m_DstXStep)
    , m_DstYNum(params.m_DstYNum)
    , m_DstYStep(params.m_DstYStep)
    , m_DstZNum(params.m_DstZNum)
    , m_DstZStep(params.m_DstZStep)

    , m_NumPartitions(params.m_NumPartitions)
    , m_MinValue(params.m_MinValue)
    , m_MaxValue(params.m_MaxValue)

    , m_TileId(params.m_TileId)
    , m_TileIdFormat(params.m_TileIdFormat)
{

    if (m_DstIsPsum) {
        m_DstPsumBankId         = params.m_DstPsumBankId;
        m_DstPsumBankOffset     = params.m_DstPsumBankOffset;
    } else {
        m_DstSbAddress          = params.m_DstSbAddress;
        m_DstStartAtMidPart     = params.m_DstStartAtMidPart;
    }
    if (m_SrcIsPsum) {
        m_SrcPsumBankId         = params.m_SrcPsumBankId;
        m_SrcPsumBankOffset     = params.m_SrcPsumBankOffset;
    } else {
        m_SrcSbAddress          = params.m_SrcSbAddress;
        m_SrcStartAtMidPart     = params.m_SrcStartAtMidPart;
    }
    assert(verify());
}

bool
ClipByValueWaveOp::verify() const
{
    if (! this->WaveOp::verify()) {
        RETURN_ASSERT(false);
    }
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());

    if (m_MinValue > m_MaxValue) {
        RETURN_ASSERT(false);
    }

    if (m_DstXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstXStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstYStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_DstZStep < 1) {
        RETURN_ASSERT(false);
    }

    if (m_DstIsPsum) {
        if (m_DstPsumBankId < 0 || m_DstPsumBankId >= psumBuf.gNumberBanks()) {
            RETURN_ASSERT(false);
        }
        if (m_DstPsumBankOffset < 0 || m_DstPsumBankOffset >= psumBuf.gNumberBankEntries()) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_DstSbAddress < 0) {
            RETURN_ASSERT(false);
        }
    }

    if (m_NumPartitions < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcXStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcYStep < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZNum < 1) {
        RETURN_ASSERT(false);
    }
    if (m_SrcZStep <1 ){
        RETURN_ASSERT(false);
    }

    if (m_SrcIsPsum) {
        if (m_SrcPsumBankId < 0 || m_SrcPsumBankId >= psumBuf.gNumberBanks()) {
            RETURN_ASSERT(false);
        }
        if (m_SrcPsumBankOffset < 0 || m_SrcPsumBankOffset >= psumBuf.gNumberBankEntries()) {
            RETURN_ASSERT(false);
        }
    } else {
        if (m_SrcSbAddress < 0) {
            RETURN_ASSERT(false);
        }
    }

    if (m_TileIdFormat == "") {
        RETURN_ASSERT(false);
    }

    for (auto n : m_TileId) {
        if (n < 0) {
            RETURN_ASSERT(false);
        }
    }

    return true;
}



bool
ClipByValueWaveOp::Params::verify() const
{
    return true;
}

std::string
ClipByValueWaveOp::gTypeStrStatic()
{
    return WaveOpTypeStr_ClipByValue;
}

}}

