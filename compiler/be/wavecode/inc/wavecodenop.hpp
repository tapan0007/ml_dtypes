#pragma once

#ifndef KCC_WAVECODE_WAVECODENOP_H
#define KCC_WAVECODE_WAVECODENOP_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class NopWaveOp;
}

namespace wavecode {



class WaveCodeNop : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeNop(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

};


}}

#endif // KCC_WAVECODE_WAVECODENOP_H


