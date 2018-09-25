#pragma once

#ifndef KCC_WAVECODE_WAVECODECLIPBYVALUE_H
#define KCC_WAVECODE_WAVECODECLIPBYVALUE_H

#include <string>
#include <cstdio>




#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace wave {
    class ClipByValueWaveOp;
}

namespace wavecode {



class WaveCodeClipByValue : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeClipByValue(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

};


}}

#endif // KCC_WAVECODE_WAVECODECLIPBYVALUE_H

