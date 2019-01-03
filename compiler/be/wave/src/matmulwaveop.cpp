#include <sstream>



#include "utils/inc/datatype.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/waveconsts.hpp"
#include "wave/inc/matmulwaveop.hpp"



namespace kcc {
namespace wave {

MatMulWaveOp::MatMulWaveOp(const MatMulWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps)
    : BaseClass(params, prevWaveOps)
    , m_FmapXNum(params.m_FmapXNum)
    , m_FmapXStep(params.m_FmapXStep)
    , m_FmapYNum(params.m_FmapYNum)
    , m_FmapYStep(params.m_FmapYStep)
    , m_FmapZNum(params.m_FmapZNum)
    , m_FmapZStep(params.m_FmapZStep)
    , m_IfmapsSbAddress(params.m_IfmapsSbAddress)
    , m_InDtype(DataType::dataTypeId2DataType(params.m_InDtypeId))
    , m_NumColumnPartitions(params.m_NumColumnPartitions)
    , m_NumRowPartitions(params.m_NumRowPartitions)
    , m_OutDtype(DataType::dataTypeId2DataType(params.m_OutDtypeId))
    , m_PsumBankId(params.m_PsumBankId)
    , m_PsumBankOffset(params.m_PsumBankOffset)
    , m_PsumXNum(params.m_PsumXNum)
    , m_PsumXStep(params.m_PsumXStep)
    , m_PsumYNum(params.m_PsumYNum)
    , m_PsumYStep(params.m_PsumYStep)
    , m_PsumZNum(params.m_PsumZNum)
    , m_PsumZStep(params.m_PsumZStep)
    , m_StartTensorCalc(params.m_StartTensorCalc)
    , m_StopTensorCalc(params.m_StopTensorCalc)
    // waveop name
    // waveop type
    , m_WeightsSbAddress(params.m_WeightsSbAddress)
    , m_IfmapReplicationNumRows(params.m_IfmapReplicationNumRows)
    , m_IfmapReplicationResolution(params.m_IfmapReplicationResolution)
    , m_IfmapReplicationShiftAmnt(params.m_IfmapReplicationShiftAmnt)
    , m_QuantOffsetIfmaps(params.m_QuantOffsetIfmaps)
    , m_QuantOffsetWeights(params.m_QuantOffsetWeights)
    , m_PEPerfOptMode(params.m_PEPerfOptMode)
    , m_IsDynamicWeights(params.m_IsDynamicWeights)
{
    assert(params.verify());
    assert(verify());
}



bool
MatMulWaveOp::verify() const
{
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());
    if (! this->BaseClass::verify()) {
        return false;
    }
    if (m_FmapXNum < 1) {
        return false;
    }
    if (m_FmapYNum < 1) {
        return false;
    }
    if (m_FmapZNum < 1) {
        return false;
    }
    if (m_IfmapsSbAddress < 0) {
        return false;
    }
    if (m_NumColumnPartitions < 1) {
        return false;
    }
    if (m_NumRowPartitions < 1) {
        return false;
    }
    if (m_PsumBankId < 0 || m_PsumBankId >= psumBuf.gNumberBanks()) {
        return false;
    }
    if (m_PsumBankOffset < 0
            || m_PsumBankOffset >= psumBuf.gNumberBankEntries(gOutDtype().gDataTypeId()))
    {
        return false;
    }
    if (m_PsumXNum < 1) {
        return false;
    }
    if (m_PsumXStep == 0 && m_PsumXNum != 1) {
        return false;
    }

    if (m_PsumYNum < 1) {
        return false;
    }
    if (m_PsumYStep == 0 && m_PsumYNum != 1) {
        return false;
    }
    if (m_PsumZNum < 1) {
        return false;
    }
    if (m_PsumZStep == 0 && m_PsumZNum != 1) {
        return false;
    }
    if (m_WeightsSbAddress < -1) {
        return false;
    }
    if (m_IfmapReplicationNumRows < 0) {
        return false;
    }
    if (m_IfmapReplicationResolution < 0) {
        return false;
    }
    if (m_IfmapReplicationShiftAmnt < 0) {
        return false;
    }
    return true;
}

bool
MatMulWaveOp::Params::verify() const
{
    if (! this-> MatMulWaveOp::BaseClass::Params::verify()) {
        return false;
    }
    if (m_FmapXNum < 1) {
        return false;
    }
    if (m_FmapYNum < 1) {
        return false;
    }
    if (m_FmapZNum < 1) {
        return false;
    }
    if (m_IfmapsSbAddress < 0) {
        return false;
    }
    if (m_NumColumnPartitions < 1) {
        return false;
    }
    if (m_NumRowPartitions < 1) {
        return false;
    }
    if (m_PsumBankId < 0) {
        return false;
    }
    if (m_PsumBankOffset < 0) {
        return false;
    }
    if (m_PsumXNum < 1) {
        return false;
    }
    if (m_PsumXStep == 0 && m_PsumXNum != 1) {
        return false;
    }
    if (m_PsumYNum < 1) {
        return false;
    }
    if (m_PsumYStep == 0 && m_PsumYNum != 1) {
        return false;
    }
    if (m_PsumZNum < 1) {
        return false;
    }
    if (m_PsumZStep == 0 && m_PsumZNum != 1) {
        return false;
    }
    if (m_WeightsSbAddress < -1) {
        return false;
    }
    if (DataTypeId::Uint8 == m_InDtypeId) {
        switch (m_PEPerfOptMode) {
            case PEPerfOptType::None:
                break;
            case PEPerfOptType::DoubleRow:
                if (m_FmapXStep < 2) {
                    // double_row: matmul src_x_step must be at least 2
                    // TODO: we need to check ldweight src_x_step as well
                    // but right now it is hard coded
                    return false;
                }
                break;
            case PEPerfOptType::DoubleColumn:
                if (m_PsumXStep < 2) {
                    // double_column: dst_x_step must be at least 2
                    return false;
                }
                break;
            case PEPerfOptType::DoublePixel:
                if (m_FmapXStep < 2 || m_PsumXStep < 2) {
                    // double_pixel: both matmul src_x_step and dst_x_step
                    // must be at least 2
                    return false;
                }
                break;
            default:
                return false;
        }
    }
    return true;
}

std::string 
MatMulWaveOp::gTypeStrStatic()
{
    return WaveOpTypeStr::MatMul;
}

}}

