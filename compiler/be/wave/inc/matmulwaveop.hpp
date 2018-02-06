#pragma once

#ifndef KCC_WAVE_MATMULWAVEOP_H
#define KCC_WAVE_MATMULWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "wave/inc/waveop.hpp"


namespace kcc {
namespace wave {


class MatMulWaveOp : public WaveOp {
public:
    class Params;

    class WaveId {
        //"wave_id_format": "nmhwcrs",
        kcc_int16 m_BatchIdx;
        kcc_int16 m_OfmapFoldIdx;
        kcc_int16 m_TileY;
        kcc_int16 m_TileX;
        kcc_int16 m_IfmapFoldIdx;
        kcc_int16 m_FilterPixelX;
        kcc_int16 m_FilterPixelY;
    };
public:
    MatMulWaveOp(const MatMulWaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps);


    //----------------------------------------------------------------
    bool qMatMultWaveOp() const override {
        return true;
    }

private:
    MatMulWaveOp::WaveId    m_WaveId;
    std::string             m_WaveIdFormat;
    kcc_int16               m_IfmapsAtomId;
    kcc_int16               m_IfmapsOffsetInAtom;
    kcc_int16               m_PsumBankId;
    bool                    m_Start;
};

class MatMulWaveOp::Params {
public:
    WaveOp::Params  m_WaveOpParams;
    WaveId          m_WaveId;
    std::string     m_WaveIdFormat;
    kcc_int16       m_IfmapsAtomId;
    kcc_int16       m_IfmapsOffsetInAtom;
    kcc_int16       m_PsumBankId;
    bool            m_Start;
};

}}


#endif


