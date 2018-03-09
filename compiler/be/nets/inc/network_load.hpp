#pragma once

#ifndef KCC_NETS_NETWORK_LOAD_H
#define KCC_NETS_NETWORK_LOAD_H

#include "nets/inc/network.hpp"

namespace kcc {

namespace nets {

class Network::Load {
public:
    Load(Network& network);
    wave::SbAtomFileWaveOp* loadSbAtomFile(const serialize::SerWaveOp& serWaveOp);
    wave::SbAtomSaveWaveOp* loadSbAtomSave(const serialize::SerWaveOp& serWaveOp);
    wave::PoolWaveOp* loadPool(const serialize::SerWaveOp& serWaveOp);
    wave::MatMulWaveOp* loadMatMul(const serialize::SerWaveOp& serWaveOp);
    wave::ActivationWaveOp* loadActivation(const serialize::SerWaveOp& serWaveOp);
    wave::ResAddWaveOp* loadResAdd(const serialize::SerWaveOp& serWaveOp);

private:
    void
    fillWaveOpParams(const serialize::SerWaveOp& serWaveOp,
                     std::vector<wave::WaveOp*>& prevWaveOps,
                     wave::WaveOp::Params& waveOpParams);
private:
    Network&    m_Network;
};

}}

#endif

