#pragma once

#ifndef KCC_WAVECODE_WAVECODEREGLOAD_H
#define KCC_WAVECODE_WAVECODEREGLOAD_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class PoolWaveOp;
}

namespace wavecode {



class WaveCodeRegLoad : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeRegLoad(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

};


}}

#endif // KCC_WAVECODE_WAVECODEREGLOAD_H

