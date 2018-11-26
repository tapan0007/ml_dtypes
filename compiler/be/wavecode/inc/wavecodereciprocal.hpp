#pragma once

#ifndef KCC_WAVECODE_WAVECODERECIPROCAL_H
#define KCC_WAVECODE_WAVECODERECIPROCAL_H

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



class WaveCodeReciprocal : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeReciprocal(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

};


}}

#endif // KCC_WAVECODE_WAVECODERECIPROCAL_H

