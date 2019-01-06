#pragma once

#ifndef KCC_NETS_LOADSPLITTER_H
#define KCC_NETS_LOADSPLITTER_H


#include <string>
#include <vector>
#include <assert.h>



#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "wave/inc/waveop.hpp"


namespace kcc {

namespace arch {
    class Arch;
}


namespace wave {
    class WaveOp;
    class SbAtomWaveOp;
    class SbAtomLoadWaveOp;
    class SbAtomSaveWaveOp;
    class TpbCopyWaveOp;
    class MatMulWaveOp;
    class PoolWaveOp;
    class ReciprocalWaveOp;
    class ActivationWaveOp;
    class ClipByValueWaveOp;
    class TensorWaveOp;
    class TensorTensorWaveOp;
    class TensorScalarWaveOp;
    class NopWaveOp;
}

namespace nets {

using namespace utils;

class Network;




//--------------------------------------------------------
// The whole neural net
//--------------------------------------------------------
class LoadSplitter {
public:
    LoadSplitter(Network& network);

    void SplitReplicatedLoads();
private:

private:
    LoadSplitter() = delete;
    LoadSplitter(const LoadSplitter&) = delete;

    wave::TpbCopyWaveOp* splitOneReplicatedLoad(
            wave::SbAtomLoadWaveOp* prevReplicatedLoad,
            wave::SbAtomLoadWaveOp* loadwaveop);

    wave::SbAtomLoadWaveOp* findPrevReplicatedLoad(wave::SbAtomLoadWaveOp* loadWaveop);

private:
    Network&            m_Network;
}; // LoadSplitter




} // namespace nets
} // namespace kcc

#endif // KCC_NETS_LOADSPLITTER_H

