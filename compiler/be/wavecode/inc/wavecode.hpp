#pragma once

#ifndef KCC_WAVECODE_WAVECODE_H
#define KCC_WAVECODE_WAVECODE_H

#include <string>
#include <cstdio>



#include "tcc.hpp"

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace arch {
    class Arch;
}
namespace nets {
    class Network;
}
namespace layers {
    class Layer;
}
namespace wave {
    class WaveOp;
}

namespace wavecode {



class WaveCode {
public:
    //----------------------------------------------------------------
    WaveCode(const nets::Network* network, const arch::Arch& arch);

    void generate(const char* fileName);

private:
    WaveCode() = delete;
    WaveCode(const WaveCode&) = delete;

private:
    WaveCodeWaveOp& getCodeGen(const wave::WaveOp* waveOp);

private:
    const nets::Network*                    m_Network;
    const arch::Arch&                       m_Arch;
};

}}

#endif // KCC_WAVECODE_WAVECODE_H

