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

#include "layers/inc/convlayer.hpp"

#include "wave/inc/waveop.hpp"


namespace kcc {

namespace wave {


class MatMulWaveOp : public WaveOp {
public:
    class Params;

    class WaveId { // must be defined inside MatMulWaveOp because it is instantiated in MatMulWaveOp
    public:
        bool verify() const;
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

        void convertFrom(const std::string& fmt, const std::vector<int>& waveId);
        void convertTo(const std::string& fmt, std::vector<int>& waveId) const;

    private:
        kcc_int16 m_BatchIdx        = -1;
        kcc_int16 m_OfmapFoldIdx    = -1;
        kcc_int16 m_TileY           = -1;
        kcc_int16 m_TileX           = -1;
        kcc_int16 m_IfmapFoldIdx    = -1;
        kcc_int16 m_FilterPixelX    = -1;
        kcc_int16 m_FilterPixelY    = -1;
    };



public:
    MatMulWaveOp(const MatMulWaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps);

    const std::string& gWaveIdFormat() const {
        return m_WaveIdFormat;
    }

    const WaveId& gWaveId() const {
        return m_WaveId;
    }

    kcc_int16 gIfmapsAtomId() const {
        return m_IfmapsAtomId;
    }

    kcc_int32 gIfmapsAtomSize () const {
        return m_IfmapsAtomSize;
    }

    kcc_int16 gIfmapsOffsetInAtom() const {
        return m_IfmapsOffsetInAtom;
    }

    kcc_int16 gWeightsAtomId() const {
        return m_WeightsAtomId;
    }

    kcc_int16 gWeightsOffsetInAtom() const {
        return m_WeightsOffsetInAtom;
    }

    kcc_int16 gPsumBankId() const {
        return m_PsumBankId;
    }

    kcc_int16 gPsumBankOffset () const {
        return m_PsumBankOffset;
    }

    bool qStartTensorCalc() const {
        return m_StartTensorCalc;
    }

    bool qStopTensorCalc() const {
        return m_StopTensorCalc;
    }

    kcc_int16 gIfmapTileHeight () const {
        return m_IfmapTileHeight;
    }
    void rIfmapTileHeight (kcc_int16 ifmapTileHeight) {
        m_IfmapTileHeight = ifmapTileHeight;
    }

    kcc_int16 gIfmapTileWidth () const {
        return m_IfmapTileWidth;
    }
    void rIfmapTileWidth (kcc_int16 ifmapTileWidth) {
        m_IfmapTileWidth = ifmapTileWidth;
    }

    kcc_int16 gOfmapTileHeight () const {
        return m_OfmapTileHeight;
    }
    void rOfmapTileHeight (kcc_int16 ofmapTileHeight) {
        m_OfmapTileHeight = ofmapTileHeight;
    }

    kcc_int16 gOfmapTileWidth () const {
        return m_OfmapTileWidth;
    }
    void rOfmapTileWidth (kcc_int16 ofmapTileWidth) {
        m_OfmapTileWidth = ofmapTileWidth;
    }

    kcc_int16 gIfmapCount () const {
        return m_IfmapCount;
    }
    void rIfmapCount (kcc_int16 ifmapCount) {
        m_IfmapCount = ifmapCount;
    }

    kcc_int16 gOfmapCount () const {
        return m_OfmapCount;
    }
    void rOfmapCount (kcc_int16 ofmapCount) {
        m_OfmapCount = ofmapCount;
    }

    kcc_int16 gBatchingInWave () const {
        return m_BatchingInWave;
    }
    void rBatchingInWave (kcc_int16 batchingInWave) {
        m_BatchingInWave = batchingInWave;
    }

    //----------------------------------------------------------------
    bool qMatMulWaveOp() const override {
        return true;
    }

    EngineId gEngineId() const override {
        return EngineId::PeArray;
    }

    static std::string gTypeStrStatic() {
        return WaveOpTypeStr_MatMul;
    }
    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::MatMul;
    }

    bool verify() const override;


    kcc_int16 gFmapXNum () const {
        return m_FmapXNum;
    }

    kcc_int16 gFmapXStep () const {
        return m_FmapXStep;
    }

    kcc_int16 gFmapYNum () const {
        return m_FmapYNum;
    }

    kcc_int16 gFmapYStep () const {
        return m_FmapYStep;
    }

    kcc_int16 gFmapZNum () const {
        return m_FmapZNum;
    }

    kcc_int16 gFmapZStepAtoms () const {
        return m_FmapZStepAtoms;
    }

    kcc_int16 gPsumXNum () const {
        return m_PsumXNum;
    }

    kcc_int16 gPsumXStep () const {
        return m_PsumXStep;
    }

    kcc_int16 gPsumYNum () const {
        return m_PsumYNum;
    }

    kcc_int16 gPsumYStep () const {
        return m_PsumYStep;
    }

    kcc_int16 gNumColumnPartitions () const {
        return m_NumColumnPartitions;
    }

    kcc_int16 gNumRowPartitions () const {
        return m_NumRowPartitions;
    }

    kcc_int16 gStrideX () const {
        return m_StrideX;
    }
    kcc_int16 gStrideY () const {
        return m_StrideY;
    }

    const DataType& gInDtype () const {
        return m_InDtype;
    }
    const DataType& gOutDtype () const {
        return m_OutDtype;
    }

private:
    MatMulWaveOp() = delete;
    MatMulWaveOp(const MatMulWaveOp&) = delete;

    MatMulWaveOp& operator= (const MatMulWaveOp&) const = delete;

private:
    kcc_int16       m_BatchingInWave        = -1;
    kcc_int16       m_FmapXNum              = -1;
    kcc_int16       m_FmapXStep             = -1;
    kcc_int16       m_FmapYNum              = -1;
    kcc_int16       m_FmapYStep             = -1;
    kcc_int16       m_FmapZNum              = -1;
    kcc_int16       m_FmapZStepAtoms        = -1;
    kcc_int16       m_IfmapCount            = -1;
    kcc_int16       m_IfmapTileHeight       = -1;
    kcc_int16       m_IfmapTileWidth        = -1;
    kcc_int16       m_IfmapsAtomId          = -1;
    kcc_int32       m_IfmapsAtomSize        = -1; // in bytes
    kcc_int16       m_IfmapsOffsetInAtom    = -1;
    const DataType& m_InDtype;
    // layer name
    kcc_int16       m_NumColumnPartitions   = -1;
    kcc_int16       m_NumRowPartitions      = -1;
    kcc_int16       m_OfmapCount            = -1;
    kcc_int16       m_OfmapTileHeight       = -1;
    kcc_int16       m_OfmapTileWidth        = -1;
    const DataType& m_OutDtype;
    // previous layers
    kcc_int16       m_PsumBankId            = -1;
    kcc_int16       m_PsumBankOffset        = -1;
    kcc_int16       m_PsumXNum              = -1;
    kcc_int16       m_PsumXStep             = -1;
    kcc_int16       m_PsumYNum              = -1;
    kcc_int16       m_PsumYStep             = -1;
    bool            m_StartTensorCalc       = true;
    bool            m_StopTensorCalc        = true;
    kcc_int16       m_StrideX               = -1;
    kcc_int16       m_StrideY               = -1;
    WaveId          m_WaveId;
    std::string     m_WaveIdFormat          = "";
    // waveop name
    // waveop type
    kcc_int16       m_WeightsAtomId         = -2; // -1 means do not load weights
    kcc_int16       m_WeightsOffsetInAtom   = -2; // -1 means do not load weights
}; // class MatMulWaveOp : public WaveOp






class MatMulWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
public:
    kcc_int16       m_BatchingInWave        = -1;
    kcc_int16       m_FmapXNum              = -1;
    kcc_int16       m_FmapXStep             = -1;
    kcc_int16       m_FmapYNum              = -1;
    kcc_int16       m_FmapYStep             = -1;
    kcc_int16       m_FmapZNum              = -1;
    kcc_int16       m_FmapZStepAtoms        = -1;
    kcc_int16       m_IfmapCount            = -1;
    kcc_int16       m_IfmapTileHeight       = -1;
    kcc_int16       m_IfmapTileWidth        = -1;
    kcc_int16       m_IfmapsAtomId          = -1;
    kcc_int32       m_IfmapsAtomSize        = -1;
    kcc_int16       m_IfmapsOffsetInAtom    = -1;
    DataTypeId      m_InDtypeId             = DataTypeId::None;
    // layer name
    kcc_int16       m_NumColumnPartitions   = -1;
    kcc_int16       m_NumRowPartitions      = -1;
    kcc_int16       m_OfmapCount            = -1;
    kcc_int16       m_OfmapTileHeight       = -1;
    kcc_int16       m_OfmapTileWidth        = -1;
    DataTypeId      m_OutDtypeId            = DataTypeId::None;
    // previous layers
    kcc_int16       m_PsumBankId            = -1;
    kcc_int16       m_PsumBankOffset        = -1;
    kcc_int16       m_PsumXNum              = -1;
    kcc_int16       m_PsumXStep             = -1;
    kcc_int16       m_PsumYNum              = -1;
    kcc_int16       m_PsumYStep             = -1;
    bool            m_StartTensorCalc       = true;
    bool            m_StopTensorCalc        = true;
    kcc_int16       m_StrideX               = -1;
    kcc_int16       m_StrideY               = -1;
    WaveId          m_WaveId;
    std::string     m_WaveIdFormat          = "";
    // waveop name
    // waveop type
    kcc_int16       m_WeightsAtomId         = -1;
    kcc_int16       m_WeightsOffsetInAtom   = -1;
};

}}


#endif


