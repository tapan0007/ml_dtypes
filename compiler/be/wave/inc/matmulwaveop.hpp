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
    public:
        kcc_int16 gBatchIdx() const {
            return m_BatchIdx;
        }
        void rBatchIdx(kcc_int16 batchIdx) {
            m_BatchIdx = batchIdx;
        }

        kcc_int16 gOfmapFoldIdx() const {
            return m_OfmapFoldIdx;
        }
        void rOfmapFoldIdx(kcc_int16 ofmapFoldIdx) {
            m_OfmapFoldIdx = ofmapFoldIdx;
        }

        kcc_int16 gTileY() const {
            return m_TileY;
        }
        void rTileY(kcc_int16 tileY) {
            m_TileY = tileY;
        }

        kcc_int16 gTileX() const {
            return m_TileX;
        }
        void rTileX(kcc_int16 tileX) {
            m_TileX = tileX;
        }

        kcc_int16 gIfmapFoldIdx() const {
            return m_IfmapFoldIdx;
        }
        void rIfmapFoldIdx(kcc_int16 ifmapFoldIdx) {
            m_IfmapFoldIdx = ifmapFoldIdx;
        }

        kcc_int16 gFilterPixelX() const {
            return m_FilterPixelX;
        }
        void rFilterPixelX(kcc_int16 filterPixelX) {
            m_FilterPixelX = filterPixelX;
        }

        kcc_int16 gFilterPixelY() const {
            return m_FilterPixelY;
        }
        void rFilterPixelY(kcc_int16 filterPixelY) {
            m_FilterPixelY = filterPixelY;
        }

    private:
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

