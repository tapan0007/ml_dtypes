#pragma once

#ifndef KCC_WAVECODE_WAVECODEACTIVATION_H
#define KCC_WAVECODE_WAVECODEACTIVATION_H

#include <string>
#include <cstdio>



#include "tcc.hpp"

#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"

#include "wavecode/inc/wavecodewaveop.hpp"


namespace kcc {

namespace layers {
    class Layer;
}
namespace wave {
    class ActivationWaveOp;
}

namespace wavecode {



class WaveCodeActivation : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeActivation(WaveCode* wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

};


}}

#endif // KCC_WAVECODE_WAVECODEACTIVATION_H

