#pragma once

#ifndef KCC_WAVECODE_WAVECODEREGSTORE_H
#define KCC_WAVECODE_WAVECODEREGSTORE_H

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



class WaveCodeRegStore : public WaveCodeWaveOp {
public:
    //----------------------------------------------------------------
    WaveCodeRegStore(WaveCodeRef wavecode);

    //----------------------------------------------------------------
    void generate(wave::WaveOp* waveOp) override;

};


}}

#endif // KCC_WAVECODE_WAVECODEREGSTORE_H

