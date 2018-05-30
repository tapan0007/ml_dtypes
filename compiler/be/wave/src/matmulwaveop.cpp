#include <sstream>



#include "utils/inc/datatype.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/matmulwaveop.hpp"



namespace kcc {
namespace wave {

MatMulWaveOp::MatMulWaveOp(const MatMulWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, prevWaveOps)
    , m_BatchingInWave(params.m_BatchingInWave)
    , m_FmapXNum(params.m_FmapXNum)
    , m_FmapXStep(params.m_FmapXStep)
    , m_FmapYNum(params.m_FmapYNum)
    , m_FmapYStep(params.m_FmapYStep)
    , m_FmapZNum(params.m_FmapZNum)
    , m_FmapZStep(params.m_FmapZStep)
    , m_IfmapCount(params.m_IfmapCount)
    , m_IfmapTileHeight(params.m_IfmapTileHeight)
    , m_IfmapTileWidth(params.m_IfmapTileWidth)
    , m_IfmapsSbAddress(params.m_IfmapsSbAddress)
    , m_InDtype(DataType::dataTypeId2DataType(params.m_InDtypeId))
    // layer name
    , m_NumColumnPartitions(params.m_NumColumnPartitions)
    , m_NumRowPartitions(params.m_NumRowPartitions)
    , m_OfmapCount(params.m_OfmapCount)
    , m_OfmapTileHeight(params.m_OfmapTileHeight)
    , m_OfmapTileWidth(params.m_OfmapTileWidth)
    , m_OutDtype(DataType::dataTypeId2DataType(params.m_OutDtypeId))
    // previous layers
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
    , m_StrideX(params.m_StrideX)
    , m_StrideY(params.m_StrideY)
    , m_WaveId(params.m_WaveId)
    , m_WaveIdFormat(params.m_WaveIdFormat)
    // waveop name
    // waveop type
    , m_WeightsSbAddress(params.m_WeightsSbAddress)
    , m_IfmapReplicationNumRows(params.m_IfmapReplicationNumRows)
    , m_IfmapReplicationResolution(params.m_IfmapReplicationResolution)
    , m_IfmapReplicationShiftAmnt(params.m_IfmapReplicationShiftAmnt)
{
    assert(params.verify());
    assert(verify());
}



bool
MatMulWaveOp::verify() const
{
    const arch::PsumBuffer& psumBuf(arch::Arch::gArch().gPsumBuffer());
    if (! this->WaveOp::verify()) {
        return false;
    }

    if (m_BatchingInWave < 1) {
        return false;
    }
    if (m_FmapXNum < 1) {
        return false;
    }
    if (m_FmapXStep < 1) {
        return false;
    }
    if (m_FmapYNum < 1) {
        return false;
    }
    if (m_FmapYStep < 1) {
        return false;
    }
    if (m_FmapZNum < 1) {
        return false;
    }
    if (m_FmapZStep < 1) {
        return false;
    }
    if (m_IfmapCount <= 0) {
        return false;
    }
    if (m_IfmapTileHeight <= 0) {
        return false;
    }
    if (m_IfmapTileWidth <= 0) {
        return false;
    }
    if (m_IfmapsSbAddress < 0) {
        return false;
    }
    // layer name
    if (m_NumColumnPartitions < 1) {
        return false;
    }
    if (m_NumRowPartitions < 1) {
        return false;
    }
    if (m_OfmapCount <= 0) {
        return false;
    }
    if (m_OfmapTileHeight <= 0) {
        return false;
    }
    if (m_OfmapTileWidth <= 0) {
        return false;
    }
    // previous layers
    if (m_PsumBankId < 0 || m_PsumBankId >= psumBuf.gNumberBanks()) {
        return false;
    }
    if (m_PsumBankOffset < 0 || m_PsumBankOffset >= psumBuf.gNumberBankEntries()) {
        return false;
    }
    if (m_PsumXNum < 1) {
        return false;
    }
    if (m_PsumXStep < 1) {
        return false;
    }

    if (m_PsumYNum < 1) {
        return false;
    }
    if (m_PsumYStep < 1) {
        return false;
    }
    if (m_PsumZNum < 1) {
        return false;
    }
    if (m_PsumZStep < 1) {
        return false;
    }
    // start
    // stop
    if (m_StrideX < 1) {
        return false;
    }
    if (m_StrideY < 1) {
        return false;
    }
    if (! m_WaveId.verify()) {
        return false;
    }
    if (m_WaveIdFormat == "") {
        return false;
    }
    // waveop name
    // waveop type
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







void
MatMulWaveOp::WaveId::convertFrom(const std::string& fmt, const std::vector<int>& waveId)
{
    // "wave_id_format": "nmhwcrs",
    const int N = fmt.size();

    for (int i = 0; i < N; ++i) {
        switch (fmt[i]) {
        case 'n':
            rBatchIdx(waveId[i]);
            break;
        case 'm':
            rOfmapFoldIdx(waveId[i]);
            break;
        case 'h':
            rTileY(waveId[i]);
            break;
        case 'w':
            rTileX(waveId[i]);
            break;
        case 'c':
            rIfmapFoldIdx(waveId[i]);
            break;
        case 'r':
            rFilterPixelX(waveId[i]);
            break;
        case 's':
            rFilterPixelY(waveId[i]);
            break;
        default:
            assert(false && "Wrong MatMulWaveOp format character");
        }
    }
}

void
MatMulWaveOp::WaveId::convertTo(const std::string& fmt, std::vector<int>& waveId) const
{
    // "wave_id_format": "nmhwcrs",
    const int N = fmt.size();

    for (int i = 0; i < N; ++i) {
        switch (fmt[i]) {
        case 'n':
            waveId[i] = gBatchIdx();
            break;
        case 'm':
            waveId[i] = gOfmapFoldIdx();
            break;
        case 'h':
            waveId[i] = gTileY();
            break;
        case 'w':
            waveId[i] = gTileX();
            break;
        case 'c':
            waveId[i] = gIfmapFoldIdx();
            break;
        case 'r':
            waveId[i] = gFilterPixelX();
            break;
        case 's':
            waveId[i] = gFilterPixelY();
            break;
        default:
            assert(false && "Wrong MatMulWaveOp format character");
        }
    }
}

bool
MatMulWaveOp::WaveId::verify() const
{
    if (m_BatchIdx < 0) {
        return false;
    }
    if (m_OfmapFoldIdx < 0) {
        return false;
    }
    if (m_TileY < 0) {
        return false;
    }
    if (m_TileX < 0) {
        return false;
    }
    if (m_IfmapFoldIdx < 0) {
        return false;
    }
    if (m_FilterPixelX < 0) {
        return false;
    }
    if (m_FilterPixelY < 0) {
        return false;
    }
    return true;
}


bool
MatMulWaveOp::Params::verify() const
{
    if (! this-> WaveOp::Params::verify()) {
        return false;
    }

    if (m_BatchingInWave < 1) {
        return false;
    }
    if (m_FmapXNum < 1) {
        return false;
    }
    if (m_FmapXStep < 1) {
        return false;
    }
    if (m_FmapYNum < 1) {
        return false;
    }
    if (m_FmapYStep < 1) {
        return false;
    }
    if (m_FmapZNum < 1) {
        return false;
    }
    if (m_FmapZStep < 1) {
        return false;
    }
    if (m_IfmapCount <= 0) {
        return false;
    }
    if (m_IfmapTileHeight <= 0) {
        return false;
    }
    if (m_IfmapTileWidth <= 0) {
        return false;
    }
    if (m_IfmapsSbAddress < 0) {
        return false;
    }
    // layer name
    if (m_NumColumnPartitions < 1) {
        return false;
    }
    if (m_NumRowPartitions < 1) {
        return false;
    }
    if (m_OfmapCount <= 0) {
        return false;
    }
    if (m_OfmapTileHeight <= 0) {
        return false;
    }
    if (m_OfmapTileWidth <= 0) {
        return false;
    }
    // previous layers
    if (m_PsumBankId < 0) {
        return false;
    }
    if (m_PsumBankOffset < 0) {
        return false;
    }
    if (m_PsumXNum < 1) {
        return false;
    }
    if (m_PsumXStep < 1) {
        return false;
    }
    if (m_PsumYNum < 1) {
        return false;
    }
    if (m_PsumYStep < 1) {
        return false;
    }
    // start
    // stop
    if (m_StrideX < 1) {
        return false;
    }
    if (m_StrideY < 1) {
        return false;
    }
    if (! m_WaveId.verify()) {
        return false;
    }
    if (m_WaveIdFormat == "") {
        return false;
    }
    // waveop name
    // waveop type
    if (m_WeightsSbAddress < -1) {
        return false;
    }
    return true;
}


}}

