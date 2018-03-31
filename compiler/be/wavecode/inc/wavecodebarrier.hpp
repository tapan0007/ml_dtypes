#pragma once

#ifndef KCC_WAVECODE_WAVECODEBARRIER_H
#define KCC_WAVECODE_WAVECODEBARRIER_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class BarrierWaveOp;
}

namespace wavecode {



class WaveCodeBarrier : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeBarrier(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

};


}}

#endif // KCC_WAVECODE_WAVECODEBARRIER_H

