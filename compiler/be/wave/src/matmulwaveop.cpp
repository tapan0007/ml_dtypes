#include <sstream>



#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"
#include "wave/inc/matmulwaveop.hpp"



namespace kcc {
namespace wave {

MatMulWaveOp::MatMulWaveOp(const MatMulWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps)
    : WaveOp(params, prevWaveOps)
    , m_IfmapTileHeight(params.m_IfmapTileHeight)
    , m_IfmapTileWidth(params.m_IfmapTileWidth)
    , m_IfmapsAtomId(params.m_IfmapsAtomId)
    , m_IfmapsOffsetInAtom(params.m_IfmapsOffsetInAtom)
    , m_OfmapTileHeight(params.m_OfmapTileHeight)
    , m_OfmapTileWidth(params.m_OfmapTileWidth)
    , m_PsumBankId(params.m_PsumBankId)
    , m_PsumBankOffset(params.m_PsumBankOffset)
    , m_Start(params.m_Start)
    , m_WaveIdFormat(params.m_WaveIdFormat)
    , m_WeightsAtomId(params.m_WeightsAtomId)
    , m_WeightsOffsetInAtom(params.m_WeightsOffsetInAtom)
    , m_WaveId(params.m_WaveId)
{
    assert(params.verify());
}


// calculate the number of ofmaps in this wave
kcc_int32
MatMulWaveOp::gNumOfmapsInFold() const
{
    const arch::Arch& arch(m_Layer->gArch());
    const kcc_int32 numPeArrayCols      = arch.gNumberPeArrayColumns();
    const kcc_int32 numOfmapsInLayer    = m_Layer->gNumOfmaps();
    const kcc_int32 ofmapFoldIdx        = m_WaveId.gOfmapFoldIdx();
    if (numOfmapsInLayer % numPeArrayCols == 0) {
        return numOfmapsInLayer;
    } else {
        if ( (numOfmapsInLayer / numPeArrayCols) == ofmapFoldIdx ) { // last
            return numOfmapsInLayer % numPeArrayCols;
        } else {
            return numPeArrayCols;
        }
    }
}

bool
MatMulWaveOp::verify() const
{
    if (! m_WaveId.verify()) {
        return false;
    }
    if (m_WaveIdFormat == "") {
        return false;
    }
    if (m_IfmapsAtomId < 0) {
        return false;
    }
    if (m_IfmapsOffsetInAtom < 0) {
        return false;
    }
    if (m_WeightsAtomId < 0) {
        return false;
    }
    // m_WeightsOffsetInAtom is negative for waves that do NOT reload weights
    if (m_PsumBankId < 0) {
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
    if (! m_WaveId.verify()) {
        return false;
    }
    if (m_WaveIdFormat == "") {
        return false;
    }
    if (m_IfmapsAtomId < 0) {
        return false;
    }
    if (m_IfmapsOffsetInAtom < 0) {
        return false;
    }
    if (m_WeightsAtomId < 0) {
        return false;
    }
    // m_WeightsOffsetInAtom is negative for waves that do NOT reload weights
    if (m_PsumBankId < 0) {
        return false;
    }
    return true;
}


}}

