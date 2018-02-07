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
#include "wave/inc/sbatomwaveop.hpp"


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
    void addPreviousWaveOp(const std::string& prevWaveOp) {
        m_PreviousWaveOps.push_back(prevWaveOp);
    }


    const std::string& gWaveOpName() const {
        return m_WaveOpName;
    }
    void 
    rWaveOpName(const std::string& waveOpName) {
        m_WaveOpName = waveOpName;
    }

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

    bool qStart() const {
        return m_Start;
    }
    void rStart(bool start) {
        m_Start = start;
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

    kcc_int32 gIfmapsFoldIdx() const {
        return m_IfmapsFoldIdx;
    }
    void rIfmapsFoldIdx(kcc_int32 ifmapsFoldIdx) {
        m_IfmapsFoldIdx = ifmapsFoldIdx;
    }

    bool qIfmapsReplicate() const {
        return m_IfmapsReplicate;
    }
    void rIfmapsReplicate(bool ifmapsReplicate) {
        m_IfmapsReplicate = ifmapsReplicate;
    }

    kcc_int32 gLength() const {
        return m_Length;
    }
    void rLength(kcc_int32 len) {
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

protected:
    bool verify() const;

private:
    // common
    std::string                 m_WaveOpType        = "";
    std::string                 m_WaveOpName        = "";
    std::string                 m_LayerName         = "";
    std::vector<std::string>    m_PreviousWaveOps;

    // SBAtomFile
    kcc_int32                   m_AtomId            = -1;
    kcc_int32                   m_IfmapsFoldIdx     = -1;
    bool                        m_IfmapsReplicate   = false;
    kcc_int32                   m_Length            = -1;
    kcc_int32                   m_OffsetInFile      = -1;
    std::string                 m_RefFile           = "";

    // MatMul
    enum {
        WaveIdFormatSize = 7,
    };
    kcc_int32                   m_IfmapsAtomId          = -1;
    kcc_int32                   m_IfmapsOffsetInAtom    = -1;
    kcc_int32                   m_PsumBankId            = -1;
    bool                        m_Start                 = true;
    wave::MatMulWaveOp::WaveId  m_WaveId;
    std::string                 m_WaveIdFormat          = "";
    kcc_int32                   m_WeightsAtomId         = -1;
    kcc_int32                   m_WeightsOffsetInAtom   = -1;
}; // class SerWaveOp



} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERWAVEOP_H

