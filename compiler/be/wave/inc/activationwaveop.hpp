#pragma once

#ifndef KCC_WAVE_ACTIVATIONWAVEOP_H
#define KCC_WAVE_ACTIVATIONWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>
#include <array>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "layers/inc/activlayer.hpp"

#include "wave/inc/waveop.hpp"


namespace kcc {
namespace layers {
    class ActivLayer;
    class PoolLayer;
}

namespace wave {


class ActivationWaveOp : public WaveOp {
public:
    class Params;
public:
    ActivationWaveOp(const ActivationWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const override;

private:
    ActivationWaveOp() = delete;

public:
    // In the first version (Resnet50 support) for Activation SW supports only 
    // the following combination of IFMAP input, BIAS input, OFMAP output srcs and dsts:
    //      IFMAP-src   BIAS-src   OFMAP-dst
    //      PSUM        SB         PSUM

    ActivationType gActType () const {
        return m_ActType;
    }
    bool qBiasAddEn () const {
        return m_BiasAddEn;
    }
    kcc_int32 gBiasAtomId () const {
        return m_BiasAtomId;
    }
    kcc_int64 gBiasOffsetInAtom () const {
        return m_BiasOffsetInAtom;
    }
    kcc_int32 gPsumBankIdDst () const {
        return m_PsumBankIdDst;
    }
    kcc_int32 gPsumBankIdSrc () const {
        return m_PsumBankIdSrc;
    }
    const std::string& gTileIdFormat () const {
        return m_TileIdFormat;
    }
    const std::array<kcc_int32, 4>& gTileId () const {
        return m_TileId;
    }
    const layers::ActivLayer* gActivLayer() const {
        return dynamic_cast<layers::ActivLayer*>(m_Layer);
    }
    bool qActivationWaveOp() const override {
        return true;
    }
    static std::string gTypeStr() {
        return WaveOpTypeStr_Activation;
    }

private:
    ActivationType              m_ActType;
    bool                        m_BiasAddEn;
    kcc_int32                   m_BiasAtomId;
    kcc_int64                   m_BiasOffsetInAtom;
    kcc_int32                   m_PsumBankIdDst;
    kcc_int32                   m_PsumBankIdSrc;
    std::string                 m_TileIdFormat;
    std::array<kcc_int32, 4>    m_TileId;
}; // class ActivationWaveOp : public WaveOp






class ActivationWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
public:
    ActivationType              m_ActType;
    bool                        m_BiasAddEn;
    kcc_int32                   m_BiasAtomId;
    kcc_int64                   m_BiasOffsetInAtom;
    kcc_int32                   m_PsumBankIdDst;
    kcc_int32                   m_PsumBankIdSrc;
    std::string                 m_TileIdFormat;
    std::array<kcc_int32, 4>    m_TileId;
};


}}

#endif



