#pragma once

#ifndef KCC_WAVECODE_WAVECODEPOOL_H
#define KCC_WAVECODE_WAVECODEPOOL_H

#include <string>
#include <cstdio>



#include "tcc/inc/tcc.hpp"

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class PoolWaveOp;
}

namespace wavecode {



class WaveCodePool : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodePool(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

};


}}

#endif // KCC_WAVECODE_WAVECODEPOOL_H

