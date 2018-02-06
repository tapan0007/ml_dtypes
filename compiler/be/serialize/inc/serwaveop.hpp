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
    const std::string& gWaveOpName() const {
        return m_WaveOpName;
    }
    const std::string& gLayerName() const {
        return m_LayerName;
    }
    const std::vector<std::string>& gPreviousWaveOps() const {
        return m_PreviousWaveOps;
    }


    const std::string& gWaveOpName() const {
        return m_WaveOpName;
    }
    void 
    rWaveOpName(const std::string& waveOpName) {
        m_WaveOpName = waveOpName;
    }

    int gIfmapsAtomId() const {
        return m_IfmapsAtomId;
    }
    int gIfmapsOffsetInAtom() const {
        return m_IfmapsOffsetInAtom;
    }
    int gPsumBankId() const {
        return m_PsumBankId;
    }
    bool qStart() const {
        return m_Start;
    }
    const wave::MatMulWaveOp::WaveId gWaveId() const {
        return m_WaveId;
    }
    const std::string& gWaveIdFormat() const {
        return m_WaveIdFormat;
    }
    int gWeightsAtomId() const {
        return m_WeightsAtomId;
    }
    int gWeightsOffsetInAtom() {
        return m_WeightsOffsetInAtom;
    }

private:
    // common
    std::string                 m_WaveOpType;
    std::string                 m_WaveOpName;
    std::string                 m_LayerName;
    std::vector<std::string>    m_PreviousWaveOps;

    // SBAtomFile
    int                         m_AtomId;
    int                         m_IfmapsFoldIdx;
    bool                        m_IfmapsReplicate;
    int                         m_Length;
    int                         m_OffsetInFile;
    std::string                 m_RefFile;

    // MatMul
    enum {
        WaveIdFormatSize = 7,
    };
    int                         m_IfmapsAtomId;
    int                         m_IfmapsOffsetInAtom;
    int                         m_PsumBankId;
    bool                        m_Start;
    wave::MatMulWaveOp::WaveId  m_WaveId;
    std::string                 m_WaveIdFormat;
    int                         m_WeightsAtomId;
    int                         m_WeightsOffsetInAtom;
}; // class SerWaveOp



} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERWAVEOP_H

