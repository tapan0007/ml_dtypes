#pragma once

#ifndef KCC_NETS_NETWORK_SAVE_H
#define KCC_NETS_NETWORK_SAVE_H

#include "nets/inc/network.hpp"

namespace kcc {

namespace nets {

class Network::Save {
public:
    Save(const Network& network);

    void saveMatmul(const wave::MatMulWaveOp* matmulWaveOp,
                    serialize::SerWaveOp& serWaveOp) const;
    void savePool(const wave::PoolWaveOp* poolWaveOp,
                    serialize::SerWaveOp& serWaveOp) const;
    void saveSbAtom(const wave::SbAtomWaveOp* sbatomWaveOp,
                    serialize::SerWaveOp& serWaveOp) const;
    void saveActivation(const wave::ActivationWaveOp* activationWaveOp,
                       serialize::SerWaveOp& serWaveOp) const;
    void saveClipByValue(const wave::ClipByValueWaveOp* activationWaveOp,
                       serialize::SerWaveOp& serWaveOp) const;
    void saveResAdd(const wave::ResAddWaveOp* resAddWaveOp,
                       serialize::SerWaveOp& serWaveOp) const;
    void saveBarrier(const wave::BarrierWaveOp* barrierWaveOp,
                       serialize::SerWaveOp& serWaveOp) const;
    void saveNop(const wave::NopWaveOp* nopWaveOp,
                       serialize::SerWaveOp& serWaveOp) const;

private:
    const Network&    m_Network;
};

}}

#endif


