#pragma once

#ifndef KCC_SERIALIZE_SERWAVEOP_H
#define KCC_SERIALIZE_SERWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>




#include "utils/inc/debug.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"

#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"


namespace kcc {
using  namespace utils;

namespace serialize {


class SerWaveOp {
public:
    SerWaveOp();


    SerWaveOp(const SerWaveOp&) = default;

public:
    /*
    { ###   SBAtomFile
      "atom_id": 24,
      "ifmaps_fold_idx": 0,
      "ifmaps_replicate": false,
      "length": 1024,
      "offset_in_file": 0,
      "ref_file": "trivnet_input:0_NCHW.npy",

            "layer_name": "input",
            "previous_waveops": [],
            "waveop_name": "input/SBAtomFile_0",
            "waveop_type": "SBAtomFile"
    },

    { ###   MatMul
      "ifmaps_atom_id": 24,
      "ifmaps_offset_in_atom": 0,
      "psum_bank_id": 0,
      "start": true,
      "wave_id": [ 0, 0, 0, 0, 0, 0, 0 ],
      "wave_id_format": "nmhwcrs",
      "weights_atom_id": 0,
      "weights_offset_in_atom": 0

            "waveop_name": "1conv/i1/MatMul_n0_m0_h0_w0_c0_r0_s0",
            "waveop_type": "MatMul",
            "layer_name": "1conv/i1",
            "previous_waveops": [
                "1conv/i1/SBAtomFile_0",
                "input/SBAtomFile_0"
            ],
    },
    */


    template<typename Archive>
    void save(Archive & archive) const;

    template<typename Archive>
    void load(Archive & archive);


    void addPreviousWaveOp(const std::string& prevWaveOp) {
        m_PreviousWaveOps.push_back(prevWaveOp);
    }


#if 0
    // common
    void rWaveOpName(const std::string& waveOpName) {
        m_WaveOpName = waveOpName;
    }
    const std::string& gWaveOpType() const {
        return m_WaveOpType;
    }
    void rWaveOpType(const std::string& waveOpType) {
        m_WaveOpType = waveOpType;
    }
    const std::string& gLayerName() const {
        return m_LayerName;
    }
    void rLayerName(const std::string& layerName) {
        m_LayerName = layerName;
    }

    const std::vector<std::string>& gPreviousWaveOps() const {
        return m_PreviousWaveOps;
    }


    const std::string& gWaveOpName() const {
        return m_WaveOpName;
    }

    // MatMul
    kcc_int32 gIfmapsAtomId() const {
        return m_IfmapsAtomId;
    }
    void rIfmapsAtomId(kcc_int32 ifmapsAtomId) {
        m_IfmapsAtomId = ifmapsAtomId;
    }

    kcc_int32 gIfmapsOffsetInAtom() const {
        return m_IfmapsOffsetInAtom;
    }
    void rIfmapsOffsetInAtom(kcc_int32 ifmapsOffsetInAtom) {
        m_IfmapsOffsetInAtom = ifmapsOffsetInAtom;
    }

    kcc_int32 gPsumBankId() const {
        return m_PsumBankId;
    }
    void rPsumBankId(kcc_int32 psumBankId) {
        m_PsumBankId = psumBankId;
    }

    bool qStartTensorCalc() const {
        return m_StartTensorCalc;
    }
    void rStartTensorCalc(bool start) {
        m_StartTensorCalc = start;
    }

    bool qStopTensorCalc() const {
        return m_StopTensorCalc;
    }
    void rStopTensorCalc(bool stop) {
        m_StopTensorCalc = stop;
    }

    const wave::MatMulWaveOp::WaveId gWaveId() const {
        return m_WaveId;
    }
    void rWaveId(const wave::MatMulWaveOp::WaveId& waveId) {
        m_WaveId = waveId;
    }

    const std::string& gWaveIdFormat() const {
        return m_WaveIdFormat;
    }
    void rWaveIdFormat(const std::string& waveIdFormat) {
        m_WaveIdFormat = waveIdFormat;
    }

    kcc_int32 gWeightsAtomId() const {
        return m_WeightsAtomId;
    }
    void rWeightsAtomId(kcc_int32 weightsAtomId) {
        m_WeightsAtomId = weightsAtomId;
    }

    kcc_int32 gWeightsOffsetInAtom() const {
        return m_WeightsOffsetInAtom;
    }
    void rWeightsOffsetInAtom(kcc_int32 weightsOffsetInAtom) {
        m_WeightsOffsetInAtom = weightsOffsetInAtom;
    }

    kcc_int32 gAtomId() const {
        return m_AtomId;
    }
    void rAtomId(kcc_int32 atomId) {
        m_AtomId = atomId;
    }

    kcc_int32 gAtomSize() const {
        return m_AtomSize;
    }
    void rAtomSize(kcc_int32 atomSize) {
        m_AtomSize = atomSize;
    }

    const std::string& gDataType() const {
        return m_DataType;
    }
    void rDataType(const std::string& dataType) {
        m_DataType = dataType;
    }


    kcc_int32 gIfmapsFoldIdx() const {
        return m_IfmapsFoldIdx;
    }
    void rIfmapsFoldIdx(kcc_int32 ifmapsFoldIdx) {
        m_IfmapsFoldIdx = ifmapsFoldIdx;
    }

    kcc_int32 gOfmapsFoldIdx() const {
        return m_OfmapsFoldIdx;
    }
    void rOfmapsFoldIdx(kcc_int32 ifmapsFoldIdx) {
        m_OfmapsFoldIdx = ifmapsFoldIdx;
    }

    bool qIfmapsReplicate() const {
        return m_IfmapsReplicate;
    }
    void rIfmapsReplicate(bool ifmapsReplicate) {
        m_IfmapsReplicate = ifmapsReplicate;
    }

    kcc_int64 gLength() const {
        return m_Length;
    }
    void rLength(kcc_int64 len) {
        m_Length = len;
    }

    kcc_int32 gOffsetInFile() const {
        return m_OffsetInFile;
    }
    void rOffsetInFile(kcc_int32 off) {
        m_OffsetInFile = off;
    }

    const std::string& gRefFile() const {
        return m_RefFile;
    }
    void rRefFile(const std::string& refFile) {
        m_RefFile = refFile;
    }

    const std::string& gRefFileFormat() const {
        return m_RefFileFormat;
    }
    void rRefFileFormat(const std::string& refFileFormat) {
        m_RefFileFormat = refFileFormat;
    }

    std::vector<int32>& gRefFileShape() const {
        return m_RefFileShape;
    }
    void rRefFileShape(const std::vector<int32>& refFileShape) {
        m_RefFileShape = refFileShape;
    }

    kcc_int32 gBatchFoldIdx () const {
        return m_BatchFoldIdx;
    }
    void rBatchFoldIdx (kcc_int32 batchFoldIdx) {
        m_BatchFoldIdx = batchFoldIdx;
    }

    kcc_int32 gIfmapTileHeight () const {
        return m_IfmapTileHeight;
    }
    void rIfmapTileHeight (kcc_int32 ifmapTileHeight) {
        m_IfmapTileHeight = ifmapTileHeight;
    }

    kcc_int32 gIfmapTileWidth () const {
        return m_IfmapTileWidth;
    }
    void rIfmapTileWidth (kcc_int32 ifmapTileWidth) {
        m_IfmapTileWidth = ifmapTileWidth;
    }

    kcc_int32 gOfmapTileHeight () const {
        return m_OfmapTileHeight;
    }
    void rOfmapTileHeight (kcc_int32 ifmapTileHeight) {
        m_OfmapTileHeight = ifmapTileHeight;
    }

    kcc_int32 gOfmapTileWidth () const {
        return m_OfmapTileWidth;
    }
    void rOfmapTileWidth (kcc_int32 ifmapTileWidth) {
        m_OfmapTileWidth = ifmapTileWidth;
    }

    kcc_int32 gPsumBankOffset () const {
        return m_PsumBankOffset;
    }
    void rPsumBankOffset (kcc_int32 psumBankOffset) {
        m_PsumBankOffset = psumBankOffset;
    }

    kcc_int32 gIfmapCount () const {
        return m_IfmapCount;
    }
    void rIfmapCount (kcc_int32 ifmapCount) {
        m_IfmapCount = ifmapCount;
    }

    kcc_int32 gOfmapCount () const {
        return m_OfmapCount;
    }
    void rOfmapCount (kcc_int32 ofmapCount) {
        m_OfmapCount = ofmapCount;
    }

    kcc_int16 gBatchingInWave () const {
        return m_BatchingInWave;
    }
    void rBatchingInWave (kcc_int16 batchingInWave) {
        m_BatchingInWave = batchingInWave;
    }
#endif

protected:
    bool verify() const;

private:
    bool verifySbAtom () const;
    bool verifySbAtomFile () const;
    bool verifySbAtomSave () const;
    bool verifyMatMul () const;

public:
    // common to all
    std::string                 m_WaveOpType        = "";
    std::string                 m_WaveOpName        = "";
    std::string                 m_LayerName         = "";
    std::vector<std::string>    m_PreviousWaveOps;

    // SBAtom
    kcc_int32                   m_AtomId            = -1;
    kcc_int32                   m_AtomSize          = -1;
    kcc_int32                   m_BatchFoldIdx      = -1;
    std::string                 m_DataType          = "";
    //layer name
    kcc_int64                   m_Length            = -1;
    kcc_int32                   m_OffsetInFile      = -1;
    // previous waveops
    std::string                 m_RefFile           = "";
    std::string                 m_RefFileFormat     = "";
    std::vector<kcc_int32>      m_RefFileShape;
    // waveop name
    // waveop type

    // SBAtomFile
    kcc_int32                   m_IfmapCount       = -1;
    kcc_int32                   m_IfmapsFoldIdx     = -1;
    bool                        m_IfmapsReplicate   = false;

    // SBAtomSave
    kcc_int32                   m_OfmapCount       = -1;
    kcc_int32                   m_OfmapsFoldIdx     = -1;

    // MatMul
    enum {
        WaveIdFormatSize = 7,
    };
    kcc_int32                   m_BatchingInWave        = -1;
    kcc_int32                   m_FmapXNum              = -1;
    kcc_int32                   m_FmapXStep             = -1;
    kcc_int32                   m_FmapYNum              = -1;
    kcc_int32                   m_FmapYStep             = -1;
    kcc_int32                   m_FmapZNum              = -1;
    kcc_int32                   m_FmapZStepAtoms        = -1;
    //kcc_int32                   m_IfmapCount            = -1;
    kcc_int32                   m_IfmapTileHeight       = -1;
    kcc_int32                   m_IfmapTileWidth        = -1;
    kcc_int32                   m_IfmapsAtomId          = -1;
    kcc_int32                   m_IfmapsAtomSize        = -1;
    kcc_int32                   m_IfmapsOffsetInAtom    = -1;
    // layer name
    //kcc_int32                   m_OfmapCount            = -1;
    kcc_int32                   m_OfmapTileHeight       = -1;
    kcc_int32                   m_OfmapTileWidth        = -1;
    // previous waveops
    kcc_int32                   m_PsumBankId            = -1;
    kcc_int32                   m_PsumBankOffset        = -1;
    bool                        m_StartTensorCalc       = true;
    bool                        m_StopTensorCalc        = true;
    wave::MatMulWaveOp::WaveId  m_WaveId;
    std::string                 m_WaveIdFormat          = "";
    // waveop name
    // waveop type
    kcc_int32                   m_WeightsAtomId         = -1;
    kcc_int32                   m_WeightsOffsetInAtom   = -1;

}; // class SerWaveOp



} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERWAVEOP_H

